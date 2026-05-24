"""
folding.py — Protein folding backend for ProteinLens AI.

Supports three backends (auto-detected in priority order):
  1. SimpleFold (Apple, local / MLX) — best on Mac Apple Silicon.
  2. Modal cloud GPU                  — used on Railway / cloud deployments.
  3. ESMFold API (Meta, public REST)  — zero-setup fallback, always works.

Backend priority:
  - If `simplefold` CLI is on PATH → use SimpleFold (local)
  - Else if MODAL_TOKEN_ID env var is set → use Modal cloud GPU
  - Else → use ESMFold public API
"""

import subprocess
import tempfile
import os
import shutil
import requests

# Load .env if present (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Backend detection ──────────────────────────────────────────────────────────

def _is_simplefold_available() -> bool:
    """Return True if the `simplefold` CLI is on PATH."""
    return shutil.which("simplefold") is not None


def _is_modal_available() -> bool:
    """Return True if Modal credentials are configured."""
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


if _is_simplefold_available():
    FOLDING_BACKEND = "simplefold"
elif _is_modal_available():
    FOLDING_BACKEND = "modal"
else:
    FOLDING_BACKEND = "esm"


# ══════════════════════════════════════════════════════════════════════════════
# Primary backend — SimpleFold (Apple, local MLX)
# ══════════════════════════════════════════════════════════════════════════════

def _fold_simplefold(sequence: str, model: str = "simplefold_100M", num_steps: int = 500) -> tuple[str, float]:
    """
    Run SimpleFold locally and return (pdb_string, mean_plddt).

    Args:
        sequence:  Amino acid sequence (uppercase, single-letter codes).
        model:     SimpleFold model variant (100M / 360M / 700M / 1.1B / 1.6B / 3B).
        num_steps: Number of flow-matching inference steps (higher = better, slower).

    Returns:
        Tuple of (PDB file contents as string, mean pLDDT confidence score 0–1).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write FASTA input
        fasta_path = os.path.join(tmpdir, "input.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">protein\n{sequence}\n")

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        # Detect backend (MLX on Apple Silicon, else torch)
        import platform
        backend = "mlx" if platform.processor() == "arm" else "torch"

        cmd = [
            "simplefold",
            "--simplefold_model", model,
            "--num_steps", str(num_steps),
            "--tau", "0.01",
            "--nsample_per_protein", "1",
            "--plddt",
            "--fasta_path", fasta_path,
            "--output_dir", output_dir,
            "--backend", backend,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=360,   # 6-min hard timeout
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"SimpleFold exited with code {result.returncode}.\n"
                f"STDERR: {result.stderr[-1000:]}"
            )

        # Find generated PDB
        pdb_path = None
        for root, _, files in os.walk(output_dir):
            for fname in files:
                if fname.endswith(".pdb"):
                    pdb_path = os.path.join(root, fname)
                    break

        if pdb_path is None:
            raise FileNotFoundError(
                f"No PDB file found in SimpleFold output directory: {output_dir}\n"
                f"STDOUT: {result.stdout[-500:]}"
            )

        with open(pdb_path) as f:
            pdb_string = f.read()

        confidence = _extract_mean_plddt(pdb_string)
        return pdb_string, confidence


# ══════════════════════════════════════════════════════════════════════════════
# Fallback backend — ESMFold (Meta, public API)
# ══════════════════════════════════════════════════════════════════════════════

ESM_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def _fold_esm(sequence: str, **_) -> tuple[str, float]:
    """
    Fold via Meta's public ESMFold API endpoint.

    No API key required. Best for sequences < 400 AA for speed.
    Returns (pdb_string, mean_plddt).
    """
    response = requests.post(
        ESM_API_URL,
        data=sequence,
        headers={"Content-Type": "text/plain"},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"ESMFold API returned HTTP {response.status_code}: {response.text[:300]}"
        )
    pdb_string = response.text
    confidence = _extract_mean_plddt(pdb_string)
    return pdb_string, confidence


# ══════════════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════════════

def _fold_modal(sequence: str, model: str = "simplefold_100M", num_steps: int = 500) -> tuple[str, float]:
    """
    Fold via Modal cloud GPU (SimpleFold running on T4).
    Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
    """
    import modal

    # Set credentials from env (Modal SDK picks these up automatically)
    token_id = os.environ["MODAL_TOKEN_ID"]
    token_secret = os.environ["MODAL_TOKEN_SECRET"]

    # Import the deployed function
    fold_fn = modal.Function.lookup("protein-lens-ai", "fold_protein_modal")
    result = fold_fn.remote(sequence, model=model, num_steps=num_steps)
    return result["pdb_string"], result["confidence"]


def fold_sequence(
    sequence: str,
    model: str = "simplefold_100M",
    num_steps: int = 500,
) -> tuple[str, float]:
    """
    Fold a protein sequence and return (pdb_string, confidence).

    Backend priority: SimpleFold (local) → Modal (cloud GPU) → ESMFold API.
    """
    if FOLDING_BACKEND == "simplefold":
        return _fold_simplefold(sequence, model=model, num_steps=num_steps)
    elif FOLDING_BACKEND == "modal":
        return _fold_modal(sequence, model=model, num_steps=num_steps)
    else:
        return _fold_esm(sequence)


# ══════════════════════════════════════════════════════════════════════════════
# UniProt sequence lookup
# ══════════════════════════════════════════════════════════════════════════════

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def fetch_uniprot_sequence(query: str) -> tuple[str | None, str | None]:
    """
    Search UniProt by protein name and return (sequence, protein_name).

    Returns (None, None) if not found.
    """
    try:
        resp = requests.get(
            UNIPROT_SEARCH,
            params={
                "query": query,
                "format": "json",
                "size": 1,
                "fields": "sequence,protein_name,organism_name",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("results"):
            return None, None

        hit = data["results"][0]
        sequence = hit["sequence"]["value"]

        # Extract human-readable name
        try:
            name = hit["proteinDescription"]["recommendedName"]["fullName"]["value"]
        except (KeyError, IndexError):
            try:
                name = hit["proteinDescription"]["submittedNames"][0]["fullName"]["value"]
            except (KeyError, IndexError):
                name = query.title()

        return sequence, name

    except Exception as e:
        print(f"UniProt lookup failed: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def _extract_mean_plddt(pdb_string: str) -> float:
    """
    Extract mean pLDDT from PDB B-factor column.
    Both SimpleFold and ESMFold store pLDDT in the B-factor field.
    Returns a value between 0.0 and 1.0.
    """
    scores = []
    for line in pdb_string.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                b_factor = float(line[60:66].strip())
                scores.append(b_factor)
            except (ValueError, IndexError):
                pass
    if not scores:
        return 0.75  # Reasonable default if parsing fails
    mean = sum(scores) / len(scores)
    # pLDDT is stored as 0–100; normalise to 0–1
    return min(mean / 100.0, 1.0) if mean > 1.0 else mean
