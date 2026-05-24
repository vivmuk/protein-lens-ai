#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# push_to_github.sh — Initialize git repo and push to GitHub
# ════════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
#
# Requirements:
#   - GitHub CLI installed: brew install gh
#   - OR create the repo manually at github.com and paste the URL when prompted
# ════════════════════════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}🔬 ProteinLens AI — GitHub Setup${NC}"
echo ""

# ── Init git ───────────────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✓ Git initialized${NC}"
else
    echo -e "${YELLOW}  Git already initialized${NC}"
fi

git add .
git commit -m "feat: initial ProteinLens AI MVP

- Streamlit UI with dark scientific theme
- SimpleFold (Apple) local folding via MLX backend
- ESMFold API fallback (zero setup)
- Venice AI scientific explanations + MSL mode
- py3Dmol 3D visualization
- UniProt protein name search
- Modal cloud GPU backend
- Railway deployment config" 2>/dev/null || echo -e "${YELLOW}  Nothing new to commit${NC}"

# ── Create GitHub repo ─────────────────────────────────────────────────────────
echo ""
if command -v gh &> /dev/null; then
    echo -e "${CYAN}▶ GitHub CLI detected — creating repo...${NC}"
    gh repo create protein-lens-ai \
        --public \
        --description "🔬 AI-powered protein folding, 3D visualization, and Medical Affairs explanations" \
        --push \
        --source . \
        --remote origin
    echo -e "${GREEN}✓ Repo created and pushed!${NC}"
    echo ""
    REPO_URL=$(gh repo view --json url -q .url 2>/dev/null || echo "https://github.com/$(gh api user -q .login)/protein-lens-ai")
    echo -e "${BOLD}GitHub repo:${NC} $REPO_URL"
else
    echo -e "${YELLOW}GitHub CLI (gh) not found.${NC}"
    echo ""
    echo "Two options:"
    echo ""
    echo -e "  ${BOLD}Option A — Install gh and rerun:${NC}"
    echo "    brew install gh"
    echo "    gh auth login"
    echo "    ./push_to_github.sh"
    echo ""
    echo -e "  ${BOLD}Option B — Manual:${NC}"
    echo "    1. Go to https://github.com/new"
    echo "    2. Create repo named: protein-lens-ai"
    echo "    3. Paste the repo URL below"
    echo ""
    read -p "GitHub repo URL (e.g. https://github.com/vivgatesai/protein-lens-ai): " REPO_URL
    if [ -n "$REPO_URL" ]; then
        git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
        git branch -M main
        git push -u origin main
        echo -e "${GREEN}✓ Pushed to $REPO_URL${NC}"
    fi
fi

# ── Railway instructions ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  🚀  Next: Deploy to Railway${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo ""
echo "1. Go to https://railway.app/new"
echo "2. Click 'Deploy from GitHub repo'"
echo "3. Select: protein-lens-ai"
echo "4. Railway auto-detects the Procfile and deploys"
echo ""
echo -e "${BOLD}Set these environment variables in Railway dashboard:${NC}"
echo ""
echo "  VENICE_API_KEY      = (your Venice key — already in code as default)"
echo "  MODAL_TOKEN_ID      = (from ~/.modal.toml after running: modal token new)"
echo "  MODAL_TOKEN_SECRET  = (from ~/.modal.toml)"
echo ""
echo -e "Your app will be live at: ${CYAN}https://protein-lens-ai.up.railway.app${NC}"
echo ""
