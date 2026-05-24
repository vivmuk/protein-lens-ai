"""
modal_backend.py — Modal cloud GPU backend for ProteinLens AI.

FIRST-TIME SETUP (run once to pre-download model weights into the volume):
    modal run modal_backend.py::setup_weights

Then deploy and keep it running:
    modal deploy modal_backend.py
"""

import modal
import os

app = modal.App("protein-lens-ai")

_PYTORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Persistent volume — SimpleFold writes its downloaded weights here.
# Without this, every cold container re-downloads and risks mid-download corruption.
# Must be a path that does NOT exist in the base image — Modal refuses to mount
# a volume over a non-empty directory.
_MODEL_CACHE = "/model-cache"
model_vol = modal.Volume.from_name("simplefold-weights", create_if_missing=True)

simplefold_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "build-essential", "libxrender1", "libxext6")
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26,<2",
        "biopython==1.85",
        "requests",
        extra_options=f"--extra-index-url {_PYTORCH_INDEX}",
    )
    .run_commands(
        "pip install 'git+https://github.com/apple/ml-simplefold.git'",
        "pip install 'git+https://github.com/facebookresearch/esm.git'",
        "simplefold --help",
    )
    .env({
        "HF_HOME": _MODEL_CACHE,
        "TORCH_HOME": _MODEL_CACHE,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)


# ── One-time weight download ───────────────────────────────────────────────────

@app.function(
    image=simplefold_image,
    gpu="A10G",             # same GPU as inference so pLDDT init exercises the full path
    timeout=1800,
    memory=16384,
    volumes={_MODEL_CACHE: model_vol},
)
def setup_weights(model: str = "simplefold_100M"):
    """
    Run SimpleFold (with --plddt) on a tiny sequence to download BOTH the main
    model and the pLDDT checkpoint into the volume.

        modal run modal_backend.py::setup_weights
    """
    import subprocess, tempfile, os, shutil

    # Wipe any half-finished checkpoints left over from a previous failed run
    # so HF/torch re-downloads them cleanly.
    for sub in ("huggingface", "torch", "simplefold"):
        path = os.path.join(_MODEL_CACHE, sub)
        if os.path.exists(path):
            print(f"Clearing stale cache: {path}")
            shutil.rmtree(path, ignore_errors=True)

    print(f"Downloading {model} + pLDDT weights into volume...")

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        with open(fasta_path, "w") as f:
            f.write(">warmup\nACDEFGHIKLMNPQRSTVWY\n")   # 20-AA test sequence

        output_dir = os.path.join(tmpdir, "out")
        os.makedirs(output_dir)

        result = subprocess.run(
            [
                "simplefold",
                "--simplefold_model", model,
                "--num_steps", "50",
                "--plddt",                     # CRITICAL — without this, pLDDT checkpoint is never downloaded
                "--fasta_path", fasta_path,
                "--output_dir", output_dir,
                "--backend", "torch",
            ],
            capture_output=True,
            text=True,
            timeout=1700,
        )

    if result.returncode == 0:
        print(f"✓ Weights for {model} downloaded and cached in volume.")
    else:
        print(f"✗ Download failed (exit {result.returncode}).")
        print(result.stderr[-1000:])
        raise RuntimeError("Weight download failed — see stderr above.")

    # Flush volume so weights persist immediately
    model_vol.commit()


# ── Inference function ─────────────────────────────────────────────────────────

@app.function(
    image=simplefold_image,
    gpu="A10G",             # 24 GB VRAM — T4 (16 GB) OOMs on simplefold_100M + pLDDT
    timeout=1800,           # 30 min — cold start (ESM + SimpleFold + pLDDT load) can eat 5+ min before steps even begin
    retries=0,              # a 540s timeout that retries just doubles cost with no new info
    memory=16384,
    volumes={_MODEL_CACHE: model_vol},
)
def fold_protein_modal(
    sequence: str,
    model: str = "simplefold_100M",
    num_steps: int = 200,
) -> dict:
    import subprocess, tempfile, os, time, threading

    # Diagnostic: show what's actually cached in the volume so we can tell
    # cold-download from warm-load when something is slow.
    try:
        cached = []
        for sub in ("huggingface", "torch", "simplefold"):
            p = os.path.join(_MODEL_CACHE, sub)
            if os.path.isdir(p):
                total = sum(
                    os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(p) for f in fs
                )
                cached.append(f"{sub}={total / 1e9:.2f} GB")
        print(f"[modal_backend] model-cache state: {', '.join(cached) or '<empty>'}")
    except Exception as e:
        print(f"[modal_backend] could not stat cache: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">protein\n{sequence}\n")

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        cmd = [
            "simplefold",
            "--simplefold_model", model,
            "--num_steps", str(num_steps),
            "--tau", "0.01",
            "--nsample_per_protein", "1",
            "--plddt",
            "--fasta_path", fasta_path,
            "--output_dir", output_dir,
            "--backend", "torch",
        ]
        print(f"[modal_backend] running: {' '.join(cmd)}")

        # Stream stdout+stderr live so Modal logs show progress; otherwise a
        # hang inside SimpleFold is invisible until the subprocess timeout fires.
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=tmpdir,
        )

        tail_lines: list[str] = []  # keep last N lines for the error path
        def _pump():
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                tail_lines.append(line)
                if len(tail_lines) > 200:
                    del tail_lines[: len(tail_lines) - 200]
                print(f"[simplefold] {line}")

        pump_thread = threading.Thread(target=_pump, daemon=True)
        pump_thread.start()

        SUBPROC_TIMEOUT = 1700  # leave ~100s buffer under the 1800s function timeout
        try:
            proc.wait(timeout=SUBPROC_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            pump_thread.join(timeout=5)
            elapsed = time.monotonic() - start
            raise RuntimeError(
                f"SimpleFold timed out after {elapsed:.0f}s "
                f"(num_steps={num_steps}, seq_len={len(sequence)}).\n"
                f"--- last output ---\n" + "\n".join(tail_lines[-50:])
            )

        pump_thread.join(timeout=5)
        elapsed = time.monotonic() - start
        print(f"[modal_backend] simplefold finished in {elapsed:.1f}s, exit={proc.returncode}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"SimpleFold error (code {proc.returncode}):\n"
                + "\n".join(tail_lines[-50:])
            )

        # Walk the whole tmpdir (not just output_dir) — some versions write
        # the structure next to the input FASTA or to CWD. Accept .pdb or .cif.
        candidates = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith((".pdb", ".cif")):
                    candidates.append(os.path.join(root, fname))

        if not candidates:
            # Build a directory tree so we can see what SimpleFold actually wrote.
            listing_lines = []
            for root, _, files in os.walk(tmpdir):
                rel = os.path.relpath(root, tmpdir)
                listing_lines.append(f"  {rel}/")
                for fname in files:
                    try:
                        size = os.path.getsize(os.path.join(root, fname))
                    except OSError:
                        size = -1
                    listing_lines.append(f"    {fname}  ({size} bytes)")
            listing = "\n".join(listing_lines) or "  <empty>"
            raise FileNotFoundError(
                "No PDB/CIF file generated by SimpleFold.\n"
                f"tmpdir contents:\n{listing}\n"
                f"--- last stdout ---\n{(result.stdout or '')[-800:]}\n"
                f"--- last stderr ---\n{(result.stderr or '')[-800:]}"
            )

        # Prefer PDB; fall back to converting CIF.
        candidates.sort(key=lambda p: 0 if p.lower().endswith(".pdb") else 1)
        chosen = candidates[0]
        print(f"Using structure file: {chosen}")

        if chosen.lower().endswith(".pdb"):
            with open(chosen) as pf:
                pdb_string = pf.read()
        else:
            # mmCIF → PDB via Biopython (already in the image).
            from Bio.PDB import MMCIFParser, PDBIO
            import io
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("s", chosen)
            buf = io.StringIO()
            io_writer = PDBIO()
            io_writer.set_structure(structure)
            io_writer.save(buf)
            pdb_string = buf.getvalue()

        scores = []
        for line in pdb_string.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    scores.append(float(line[60:66].strip()))
                except (ValueError, IndexError):
                    pass
        confidence = (sum(scores) / len(scores) / 100.0) if scores else 0.75

        return {"pdb_string": pdb_string, "confidence": confidence, "model": model}


# ── Convenience wrapper ────────────────────────────────────────────────────────

def fold_on_modal(
    sequence: str,
    model: str = "simplefold_100M",
    num_steps: int = 500,
) -> tuple[str, float]:
    result = fold_protein_modal.remote(sequence, model=model, num_steps=num_steps)
    return result["pdb_string"], result["confidence"]


# ── Local smoke test ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def test():
    test_seq = "ACDEFGHIKLMNPQRSTVWY"
    print(f"Testing Modal fold ({len(test_seq)} AA)...")
    result = fold_protein_modal.remote(test_seq, model="simplefold_100M", num_steps=100)
    print(f"Confidence: {result['confidence']:.1%}  |  PDB lines: {len(result['pdb_string'].splitlines())}")
