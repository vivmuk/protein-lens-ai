# 🔬 ProteinLens AI

> AI-powered protein folding · Interactive 3D visualization · Scientific explanation · Medical Affairs mode

ProteinLens AI lets you fold a protein, visualize its 3D structure, and instantly generate AI-powered scientific explanations — including Medical Science Liaison (MSL) talking points for HCP communication.

---

## Features

- **Protein folding** — Apple SimpleFold (local, MLX-accelerated on Apple Silicon) with ESMFold API as automatic fallback
- **3D visualization** — Interactive py3Dmol viewer with multiple color schemes (spectrum, pLDDT confidence, hydrophobicity, secondary structure)
- **AI scientific explanation** — Powered by Venice AI, covering function, structure, disease relevance, mechanism, and therapeutic implications
- **MSL / Medical Affairs mode** — HCP-ready talking points, mechanism summaries, anticipated questions, and clinical framing
- **UniProt integration** — Search proteins by name (insulin, EGFR, BRCA1) and auto-fetch sequence
- **pLDDT confidence scoring** — Visual confidence bar and per-residue coloring
- **PDB download** — Export any predicted structure

---

## Quick Start

### 1. Run setup (installs everything)

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Launch the app

```bash
conda activate simplefold
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## Manual Setup

If you prefer to install manually:

```bash
# 1. Create environment
conda create -n simplefold python=3.10
conda activate simplefold

# 2. Install SimpleFold (Apple)
git clone https://github.com/apple/ml-simplefold.git
cd ml-simplefold
pip install -e .

# 3. Install MLX backend (Apple Silicon only)
pip install mlx==0.28.0
pip install git+https://github.com/facebookresearch/esm.git
cd ..

# 4. Install app dependencies
pip install -r requirements.txt

# 5. Launch
streamlit run app.py
```

---

## Optional: Modal Cloud GPU Backend

For larger models (360M, 700M, 1.1B) or when SimpleFold isn't installed locally:

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal token new

# Deploy the inference function
modal deploy modal_backend.py

# Test it
modal run modal_backend.py
```

Then in `app.py`, swap the folding call to use `fold_on_modal()` from `modal_backend.py`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│   Sidebar: input + controls   │   Main: viewer + AI output  │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  folding.py     │
        │  ┌────────────┐ │
        │  │ SimpleFold │ │ ← Primary (local MLX, Apple Silicon)
        │  │   (Apple)  │ │
        │  └────────────┘ │
        │  ┌────────────┐ │
        │  │  ESMFold   │ │ ← Automatic fallback (public API)
        │  │    API     │ │
        │  └────────────┘ │
        └────────┬────────┘
                 │ PDB file
        ┌────────▼────────┐
        │  py3Dmol / stmol │ ← 3D interactive visualization
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  explanation.py │
        │  Venice AI API  │ ← Scientific + MSL explanations
        └─────────────────┘
```

---

## Folding Backends

| Backend | When used | Speed | Accuracy | Setup |
|---------|-----------|-------|----------|-------|
| SimpleFold 100M | Primary (if installed) | ~30–90s | Good | conda + pip |
| SimpleFold 360M | Optional (larger) | ~2–5min | Better | same |
| ESMFold API | Automatic fallback | ~15–30s | Very good | None |

---

## Example Proteins to Try

| Protein | Why it's interesting |
|---------|---------------------|
| `insulin` | Short, well-characterized, 51 AA |
| `GLP-1` | Hot drug target (Ozempic/Wegovy) |
| `EGFR` | Classic oncology target |
| `p53` | Tumor suppressor, cancer biology |
| `hemoglobin` | Classic structural biology teaching case |
| `BRCA1` | Breast cancer, DNA repair |

---

## File Structure

```
protein-lens-ai/
├── app.py              # Streamlit application (UI, layout, orchestration)
├── folding.py          # Protein folding (SimpleFold + ESMFold fallback)
├── explanation.py      # Venice AI explanation engine (scientific + MSL)
├── modal_backend.py    # Optional Modal cloud GPU deployment
├── requirements.txt    # Python dependencies
├── setup.sh            # One-command setup script
└── README.md           # This file
```

---

## Environment Variables

You can optionally set these in a `.env` file:

```bash
VENICE_API_KEY=your_venice_key_here   # Optional: uses pre-configured key by default
```

---

## Cost Estimate (Monthly)

| Item | Estimated cost |
|------|---------------|
| Venice AI (explanations) | ~$5–15 |
| Modal GPU (optional) | ~$5–20 |
| ESMFold API (fallback) | Free |
| Streamlit hosting | Free |
| **Total** | **~$10–35** |

---

## Roadmap

- [ ] Mutation comparison (side-by-side structures)
- [ ] PubMed literature retrieval
- [ ] Agentic protein Q&A copilot
- [ ] Multi-chain / complex folding
- [ ] Structure alignment viewer
- [ ] Pathway reasoning integration

---

Built with 🍎 SimpleFold · 🤖 Venice AI · ⚡ Streamlit · 🧊 py3Dmol
