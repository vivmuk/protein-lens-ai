"""
modal_backend.py — Modal cloud GPU backend for ProteinLens AI.

Uses the `simplefold` CLI in a subprocess and points its checkpoint cache at
a persistent Modal volume so the ~6 GB weight download only happens once.

FIRST-TIME SETUP — pre-download weights into the volume:
    python -m modal run modal_backend.py::setup_weights

Then deploy:
    python -m modal deploy modal_backend.py
"""

import modal

app = modal.App("protein-lens-ai")

_PYTORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Persistent volume — SimpleFold writes its weights here. Without this, every
# cold container re-downloads ~6 GB (~15 min). Must be a path that does NOT
# exist in the base image — Modal refuses to mount over a non-empty dir.
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
        "simplefold --help",  # bake the CLI help into the image to surface install errors at build time
    )
    .env({
        # ESM uses TORCH_HOME for its hub cache — point it at the volume too.
        "TORCH_HOME": _MODEL_CACHE,
        "HF_HOME": _MODEL_CACHE,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)


def _run_simplefold(
    *,
    sequence: str,
    model: str,
    num_steps: int,
    tmpdir: str,
    output_dir: str,
) -> tuple[int, list[str]]:
    """
    Run the simplefold CLI with weights cached to the persistent volume.
    Streams stdout/stderr live to Modal logs and returns (exit_code, tail_lines).
    """
    import os, subprocess, threading, time

    fasta_path = os.path.join(tmpdir, "input.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">protein\n{sequence}\n")

    # The pLDDT path internally uses the `boltz` library, which downloads CCD
    # dict + boltz1_conf.ckpt to {output_dir}/cache/ — hardcoded, no CLI flag
    # exposes it. Trick: symlink that subdir to a persistent location on the
    # volume so boltz's writes land in the cache and subsequent runs hit it.
    aux_cache = os.path.join(_MODEL_CACHE, "aux", "boltz_cache")
    os.makedirs(aux_cache, exist_ok=True)
    cache_link = os.path.join(output_dir, "cache")
    if not os.path.lexists(cache_link):
        os.symlink(aux_cache, cache_link)

    cmd = [
        "simplefold",
        "--simplefold_model", model,
        "--num_steps", str(num_steps),
        "--tau", "0.05",
        "--nsample_per_protein", "1",
        "--plddt",
        "--ckpt_dir", _MODEL_CACHE,        # main SimpleFold + pLDDT weights → volume
        "--fasta_path", fasta_path,
        "--output_dir", output_dir,
        "--output_format", "pdb",          # skip CIF→PDB conversion
        "--backend", "torch",
    ]
    print(f"[modal_backend] running: {' '.join(cmd)}")

    start = time.monotonic()
    # Run from _MODEL_CACHE so any relative paths SimpleFold uses also land on the volume.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=_MODEL_CACHE,
    )

    tail: list[str] = []
    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            if len(tail) > 200:
                del tail[: len(tail) - 200]
            print(f"[simplefold] {line}")
    t = threading.Thread(target=_pump, daemon=True)
    t.start()

    try:
        proc.wait(timeout=1700)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        t.join(timeout=5)
        elapsed = time.monotonic() - start
        raise RuntimeError(
            f"simplefold timed out after {elapsed:.0f}s "
            f"(num_steps={num_steps}, seq_len={len(sequence)})\n"
            + "\n".join(tail[-50:])
        )

    t.join(timeout=5)
    elapsed = time.monotonic() - start
    print(f"[modal_backend] simplefold finished in {elapsed:.1f}s, exit={proc.returncode}")
    return proc.returncode, tail


# ── One-time weight download ───────────────────────────────────────────────────

@app.function(
    image=simplefold_image,
    gpu="A10G",
    timeout=1800,
    memory=16384,
    volumes={_MODEL_CACHE: model_vol},
)
def setup_weights(model: str = "simplefold_100M"):
    """
    Pre-download SimpleFold + pLDDT + ESM weights into the volume.

        python -m modal run modal_backend.py::setup_weights
    """
    import os, tempfile

    os.makedirs(_MODEL_CACHE, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "out")
        os.makedirs(output_dir)
        rc, tail = _run_simplefold(
            sequence="ACDEFGHIKLMNPQRSTVWY",
            model=model,
            num_steps=20,
            tmpdir=tmpdir,
            output_dir=output_dir,
        )

    if rc != 0:
        raise RuntimeError(
            f"setup_weights: simplefold failed (exit {rc}):\n" + "\n".join(tail[-50:])
        )

    total = 0
    for root, _, files in os.walk(_MODEL_CACHE):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    print(f"[setup_weights] volume now holds {total / 1e9:.2f} GB")
    model_vol.commit()


# ── Inference function ─────────────────────────────────────────────────────────

@app.function(
    image=simplefold_image,
    gpu="A10G",             # 24 GB VRAM — T4 (16 GB) OOMs on 100M + pLDDT
    timeout=1800,           # generous for first-ever cold container that has to download
    retries=0,
    memory=16384,
    volumes={_MODEL_CACHE: model_vol},
    scaledown_window=300,   # keep container warm for 5 min after a fold
)
def fold_protein_modal(
    sequence: str,
    model: str = "simplefold_100M",
    num_steps: int = 200,
) -> dict:
    import os, tempfile

    # Log cache state so we can tell warm vs cold downloads.
    if os.path.isdir(_MODEL_CACHE):
        try:
            total = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(_MODEL_CACHE) for f in fs
            )
            print(f"[modal_backend] /model-cache contains {total / 1e9:.2f} GB")
        except OSError as e:
            print(f"[modal_backend] could not stat cache: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)
        rc, tail = _run_simplefold(
            sequence=sequence,
            model=model,
            num_steps=num_steps,
            tmpdir=tmpdir,
            output_dir=output_dir,
        )

        if rc != 0:
            raise RuntimeError(
                f"SimpleFold error (exit {rc}):\n" + "\n".join(tail[-50:])
            )

        # SimpleFold writes either .pdb or .cif; look broadly.
        candidates = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith((".pdb", ".cif")):
                    candidates.append(os.path.join(root, fname))

        if not candidates:
            listing = "\n".join(
                f"  {os.path.relpath(os.path.join(r, f), tmpdir)}"
                for r, _, fs in os.walk(tmpdir) for f in fs
            ) or "  <empty>"
            raise FileNotFoundError(
                "No PDB/CIF file produced by SimpleFold.\n"
                f"tmpdir contents:\n{listing}\n--- last output ---\n"
                + "\n".join(tail[-30:])
            )

        candidates.sort(key=lambda p: 0 if p.lower().endswith(".pdb") else 1)
        chosen = candidates[0]
        print(f"[modal_backend] using structure file: {chosen}")

        if chosen.lower().endswith(".pdb"):
            with open(chosen) as pf:
                pdb_string = pf.read()
        else:
            from Bio.PDB import MMCIFParser, PDBIO
            import io
            parser = MMCIFParser(QUIET=True)
            struct = parser.get_structure("s", chosen)
            buf = io.StringIO()
            w = PDBIO()
            w.set_structure(struct)
            w.save(buf)
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


# ── Convenience wrapper (called from app.py) ──────────────────────────────────

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
