#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# ProteinLens AI — One-command setup script
# ════════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script:
#   1. Creates a conda environment (simplefold)
#   2. Installs Apple SimpleFold with MLX backend (Apple Silicon)
#   3. Installs Streamlit and all app dependencies
#   4. Runs a quick smoke test
#   5. Prints launch instructions
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3/M4)
#   - conda or miniconda installed
#   - Python 3.10
# ════════════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ── Colors ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Banner ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}${BOLD}║           🔬  ProteinLens AI Setup               ║${NC}"
echo -e "${BLUE}${BOLD}║   AI Protein Folding · Visualization · MSL Mode  ║${NC}"
echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check conda ────────────────────────────────────────────────────────────────
echo -e "${CYAN}▶ Checking prerequisites...${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ conda not found. Please install Miniconda:${NC}"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✓ conda found${NC}"

# Check Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo -e "${GREEN}✓ Apple Silicon detected (MLX backend will be used)${NC}"
    USE_MLX=true
else
    echo -e "${YELLOW}⚠ Intel Mac detected. MLX unavailable; falling back to torch/CPU${NC}"
    USE_MLX=false
fi

# ── Create conda environment ───────────────────────────────────────────────────
echo ""
echo -e "${CYAN}▶ Creating conda environment 'simplefold' (Python 3.10)...${NC}"

if conda env list | grep -q "^simplefold "; then
    echo -e "${YELLOW}  Environment 'simplefold' already exists — skipping creation${NC}"
else
    conda create -n simplefold python=3.10 -y
    echo -e "${GREEN}✓ Environment created${NC}"
fi

# Activate for subsequent pip commands
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate simplefold

# ── Clone & install SimpleFold ─────────────────────────────────────────────────
echo ""
echo -e "${CYAN}▶ Installing Apple SimpleFold...${NC}"

if [ ! -d "ml-simplefold" ]; then
    git clone https://github.com/apple/ml-simplefold.git
    echo -e "${GREEN}✓ Cloned ml-simplefold${NC}"
else
    echo -e "${YELLOW}  ml-simplefold directory already exists — skipping clone${NC}"
fi

cd ml-simplefold
pip install -U pip build -q
pip install -e . -q
echo -e "${GREEN}✓ SimpleFold core installed${NC}"

# Install MLX backend (Apple Silicon only)
if [ "$USE_MLX" = true ]; then
    echo -e "${CYAN}▶ Installing MLX backend (Apple Silicon)...${NC}"
    pip install mlx==0.28.0 -q
    pip install git+https://github.com/facebookresearch/esm.git -q
    echo -e "${GREEN}✓ MLX backend installed${NC}"
fi

cd ..

# ── Install app dependencies ───────────────────────────────────────────────────
echo ""
echo -e "${CYAN}▶ Installing ProteinLens AI dependencies...${NC}"
pip install -r requirements.txt -q
echo -e "${GREEN}✓ App dependencies installed${NC}"

# ── Quick smoke test ───────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}▶ Running smoke test (SimpleFold 100M, tiny peptide)...${NC}"

BACKEND_ARG="torch"
if [ "$USE_MLX" = true ]; then
    BACKEND_ARG="mlx"
fi

SMOKE_DIR=$(mktemp -d)
echo ">test_peptide" > "$SMOKE_DIR/test.fasta"
echo "ACDEFGHIKLM" >> "$SMOKE_DIR/test.fasta"

if simplefold \
    --simplefold_model simplefold_100M \
    --num_steps 50 \
    --tau 0.01 \
    --nsample_per_protein 1 \
    --fasta_path "$SMOKE_DIR/test.fasta" \
    --output_dir "$SMOKE_DIR/out" \
    --backend "$BACKEND_ARG" \
    &> /dev/null; then
    echo -e "${GREEN}✓ SimpleFold smoke test passed!${NC}"
else
    echo -e "${YELLOW}⚠ SimpleFold smoke test failed (the app will fall back to ESMFold API)${NC}"
fi

rm -rf "$SMOKE_DIR"

# ── Optional: Modal setup ──────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}▶ Setting up Modal (optional cloud GPU backend)...${NC}"
if pip show modal &>/dev/null; then
    echo -e "${YELLOW}  Modal already installed${NC}"
else
    pip install modal -q
fi

echo ""
echo -e "${YELLOW}  To authenticate Modal, run:${NC}"
echo -e "${BOLD}    modal token new${NC}"
echo -e "${YELLOW}  This opens a browser for one-click OAuth auth.${NC}"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✅  Setup complete! ProteinLens AI is ready.${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Launch the app:${NC}"
echo ""
echo -e "  ${CYAN}conda activate simplefold${NC}"
echo -e "  ${CYAN}streamlit run app.py${NC}"
echo ""
echo -e "${BOLD}Optional — deploy to Modal cloud:${NC}"
echo ""
echo -e "  ${CYAN}modal token new             ${NC}${YELLOW}# one-time auth${NC}"
echo -e "  ${CYAN}modal deploy modal_backend.py${NC}"
echo ""
echo -e "The app will open at ${BOLD}http://localhost:8501${NC}"
echo ""
