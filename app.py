"""
ProteinLens AI — Main Streamlit Application
AI-powered protein folding, visualization, and scientific explanation platform.
"""

import html
import streamlit as st
import requests
import json
import time
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProteinLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS theme (dark, scientific) ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background-color: #0d1117; color: #e6edf3; }
  .main .block-container { padding-top: 1rem; max-width: 100%; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; }

  /* Header banner */
  .header-banner {
    background: linear-gradient(135deg, #0d1b2e 0%, #1a3a5c 50%, #0d1b2e 100%);
    border: 1px solid #1f6feb40;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .header-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; color: #ffffff; }
  .header-banner p  { margin: 0; color: #8b949e; font-size: 0.9rem; }
  .header-icon { font-size: 3rem; }

  /* Cards */
  .protein-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
  }
  .protein-card h3 { color: #58a6ff; margin-top: 0; font-size: 1rem; font-weight: 600; }
  .protein-card p  { color: #c9d1d9; line-height: 1.7; margin: 0; font-size: 0.9rem; }

  /* Stat pills */
  .stat-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.75rem 0; }
  .stat-pill {
    background: #1f2d40;
    border: 1px solid #1f6feb50;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: #58a6ff;
    font-weight: 500;
  }

  /* Mode badge */
  .mode-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  .mode-badge-msl {
    background: linear-gradient(135deg, #238636, #2ea043);
  }

  /* Fold button */
  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white !important;
    border: none !important;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s;
  }
  div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px #1f6feb50;
  }

  /* ── Readable text (sidebar + main) ───────────────────────────────────── */
  .stApp, .main, section[data-testid="stSidebar"] {
    color: #e6edf3;
  }

  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] label p,
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] .stMarkdown li,
  section[data-testid="stSidebar"] .stRadio label {
    color: #e6edf3 !important;
  }

  section[data-testid="stSidebar"] .stMarkdown strong {
    color: #ffffff !important;
  }

  /* Captions, help, tooltips */
  .stCaption,
  [data-testid="stCaptionContainer"],
  [data-testid="stCaptionContainer"] p,
  [data-testid="stTooltipIcon"],
  small {
    color: #b1bac4 !important;
  }

  /* Widget labels */
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label {
    color: #e6edf3 !important;
    font-weight: 500 !important;
  }

  /* Preset description chip */
  .preset-description {
    color: #c9d1d9 !important;
    background: #21262d;
    border: 1px solid #484f58;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin: 0.25rem 0 0.75rem 0;
    font-size: 0.85rem;
    line-height: 1.5;
  }

  /* Sequence code block */
  .sequence-display {
    background: #21262d !important;
    border: 1px solid #484f58 !important;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #79c0ff !important;
    word-break: break-all;
    line-height: 1.6;
    margin-bottom: 0.75rem;
  }

  /* Text areas & inputs */
  .stTextArea textarea,
  .stTextInput input {
    background-color: #21262d !important;
    border: 1px solid #484f58 !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace;
    border-radius: 6px;
    caret-color: #58a6ff;
  }

  /* Disabled / read-only — Streamlit dims these by default */
  .stTextArea textarea:disabled,
  .stTextArea textarea[disabled],
  .stTextInput input:disabled {
    color: #e6edf3 !important;
    -webkit-text-fill-color: #e6edf3 !important;
    background-color: #21262d !important;
    opacity: 1 !important;
  }

  /* Selectbox */
  [data-baseweb="select"] > div,
  [data-baseweb="select"] div[role="combobox"],
  [data-baseweb="select"] span {
    color: #e6edf3 !important;
    background-color: #21262d !important;
  }

  [data-baseweb="popover"] li,
  [data-baseweb="menu"] li {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
  }

  /* Placeholders */
  .stTextArea textarea::placeholder,
  .stTextInput input::placeholder {
    color: #8b949e !important;
    opacity: 1 !important;
  }

  /* Code blocks (st.code) */
  .stCode, .stCode pre, .stCode code,
  [data-testid="stCode"] pre,
  [data-testid="stCode"] code {
    background-color: #21262d !important;
    color: #79c0ff !important;
    border: 1px solid #484f58 !important;
  }

  /* Status line under header */
  .engine-status {
    color: #b1bac4 !important;
    font-size: 0.9rem;
  }
  .engine-status strong {
    color: #58a6ff !important;
  }

  /* Radio circles */
  .stRadio [data-baseweb="radio"] div {
    color: #e6edf3 !important;
  }

  /* Slider labels */
  .stSlider label, .stSlider [data-testid="stMarkdownContainer"] p {
    color: #e6edf3 !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; gap: 0.25rem; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; background: transparent; border-radius: 6px; }
  .stTabs [aria-selected="true"] { background: #1f6feb !important; color: white !important; }

  /* Success / warning / info */
  .stSuccess { background: #0f2b1a !important; border-color: #238636 !important; }
  .stInfo    { background: #0d2137 !important; border-color: #1f6feb !important; }
  .stWarning { background: #2b1f0a !important; border-color: #9e6a03 !important; }

  /* Divider */
  hr { border-color: #30363d; }

  /* Scrollable explanation box */
  .explanation-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    font-size: 0.88rem;
    line-height: 1.75;
    color: #c9d1d9;
    white-space: pre-wrap;
  }

  /* Confidence bar */
  .confidence-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    margin: 0.25rem 0;
  }
  .confidence-bar-fill {
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    border-radius: 4px;
    height: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Imports (with graceful fallback) ──────────────────────────────────────────
try:
    from stmol import showmol
    import py3Dmol
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

from folding import fold_sequence, fetch_uniprot_sequence, FOLDING_BACKEND
from explanation import get_scientific_explanation, get_msl_summary
from presets import (
    EXAMPLE_PROTEINS,
    find_preset_by_label,
    get_default_preset,
    preset_select_label,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def render_protein_3d(pdb_string: str, color_scheme: str = "spectrum", bg_color: str = "#0d1117"):
    """Render interactive 3D protein structure using py3Dmol."""
    view = py3Dmol.view(width=700, height=500)
    view.addModel(pdb_string, "pdb")

    if color_scheme == "Spectrum (Rainbow)":
        view.setStyle({"cartoon": {"color": "spectrum"}})
    elif color_scheme == "Confidence (pLDDT)":
        view.setStyle({"cartoon": {
            "colorscheme": {"prop": "b", "gradient": "roygb", "min": 50, "max": 100}
        }})
    elif color_scheme == "Secondary Structure":
        view.setStyle({"cartoon": {"color": "ssJmol"}})
    elif color_scheme == "Hydrophobicity":
        view.setStyle({"cartoon": {"color": "hydrophobicity"}})
    elif color_scheme == "Monochrome Blue":
        view.setStyle({"cartoon": {"color": "#388bfd"}})

    view.setBackgroundColor(bg_color)
    view.zoomTo()
    view.spin(False)
    showmol(view, height=500, width=700)


def validate_sequence(seq: str) -> tuple[bool, str]:
    """Validate amino acid sequence."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")
    seq_clean = seq.replace(" ", "").replace("\n", "").replace("\r", "")
    # Strip FASTA header if present
    if seq_clean.startswith(">"):
        lines = seq_clean.split("\n")
        seq_clean = "".join(lines[1:])
    invalid = set(seq_clean) - valid_aa
    if invalid:
        return False, f"Invalid characters found: {', '.join(invalid)}"
    if len(seq_clean) < 10:
        return False, "Sequence too short (minimum 10 amino acids)"
    if len(seq_clean) > 1400:
        return False, f"Sequence too long ({len(seq_clean)} AA). For MVP, max is 1400 AA."
    return True, seq_clean.upper()


# ══════════════════════════════════════════════════════════════════════════════
# Session state init
# ══════════════════════════════════════════════════════════════════════════════

for key in ["pdb_string", "explanation", "msl_summary", "protein_name", "sequence", "confidence"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Pre-load a default example so Railway users can fold immediately
if st.session_state["sequence"] is None:
    _default = get_default_preset()
    st.session_state["sequence"] = _default["sequence"]
    st.session_state["protein_name"] = _default["name"]
    st.session_state["selected_preset_id"] = _default["id"]


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

_SUNDAI_LOGO = os.path.join(os.path.dirname(__file__), "assets", "sundai_club.png")
_SUNDAI_URL = "https://sundai.club"

import base64
def _img_data_uri(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

_sundai_uri = _img_data_uri(_SUNDAI_LOGO)
_sundai_badge_html = (
    f'<a href="{_SUNDAI_URL}" target="_blank" style="text-decoration:none;display:flex;'
    f'align-items:center;gap:0.5rem;color:#8b949e;font-size:0.8rem;">'
    f'<img src="{_sundai_uri}" alt="Sundai Club" style="height:32px;"/>'
    f'<span>Developed at <strong style="color:#f5b400;">Sundai Club</strong></span></a>'
    if _sundai_uri else
    f'<a href="{_SUNDAI_URL}" target="_blank" style="text-decoration:none;color:#8b949e;'
    f'font-size:0.85rem;">🍦 Developed at <strong style="color:#f5b400;">Sundai Club</strong></a>'
)

st.markdown(f"""
<div class="header-banner">
  <div class="header-icon">🔬</div>
  <div style="flex:1;">
    <h1>ProteinLens AI</h1>
    <p>AI-powered protein folding · 3D visualization · Scientific explanation · Medical Affairs mode</p>
  </div>
  <div style="margin-left:auto;">{_sundai_badge_html}</div>
</div>
""", unsafe_allow_html=True)

# Backend status
backend_label = {
    "simplefold": "🍎 SimpleFold (Local MLX)",
    "modal": "☁️ Modal GPU (SimpleFold)",
    "esm": "🌐 ESMFold API (Fallback)",
}.get(FOLDING_BACKEND, FOLDING_BACKEND)
st.markdown(
    f'<p class="engine-status">Folding engine: <strong>{backend_label}</strong> &nbsp;|&nbsp; '
    f'Explanation: <strong>Venice AI</strong> &nbsp;|&nbsp; Visualization: <strong>py3Dmol</strong></p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Input controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧬 Protein Input")

    input_mode = st.radio(
        "Input method",
        ["Quick examples", "Paste sequence", "Search by name (UniProt)"],
        label_visibility="collapsed",
    )

    if input_mode == "Quick examples":
        preset_labels = [preset_select_label(p) for p in EXAMPLE_PROTEINS]
        default_idx = next(
            (i for i, p in enumerate(EXAMPLE_PROTEINS) if p["id"] == st.session_state.get("selected_preset_id")),
            0,
        )
        selected_label = st.selectbox(
            "Choose a protein",
            preset_labels,
            index=default_idx,
            help="Curated sequences — pick one and click FOLD & ANALYZE",
        )
        preset = find_preset_by_label(selected_label)
        if preset:
            st.session_state["sequence"] = preset["sequence"]
            st.session_state["protein_name"] = preset["name"]
            st.session_state["selected_preset_id"] = preset["id"]
            st.markdown(
                f'<p class="preset-description">{preset["description"]}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("**Sequence**")
        seq_preview = st.session_state["sequence"] or ""
        st.markdown(
            f'<pre class="sequence-display">{html.escape(seq_preview)}</pre>',
            unsafe_allow_html=True,
        )

        st.markdown("**Try another quickly**")
        quick_cols = st.columns(2)
        for i, p in enumerate(EXAMPLE_PROTEINS):
            col = quick_cols[i % 2]
            short = p["name"].split("(")[0].strip()[:22]
            if col.button(short, key=f"preset_quick_{p['id']}", use_container_width=True):
                st.session_state["sequence"] = p["sequence"]
                st.session_state["protein_name"] = p["name"]
                st.session_state["selected_preset_id"] = p["id"]
                st.rerun()

    elif input_mode == "Search by name (UniProt)":
        protein_query = st.text_input(
            "Protein name",
            placeholder="e.g. insulin, hemoglobin, BRCA1",
        )
        if st.button("🔍 Fetch from UniProt") and protein_query:
            with st.spinner("Fetching from UniProt..."):
                seq, name = fetch_uniprot_sequence(protein_query)
            if seq:
                st.session_state["sequence"] = seq
                st.session_state["protein_name"] = name
                st.success(f"Found: **{name}** ({len(seq)} AA)")
            else:
                st.error("Protein not found. Try a different name.")

        if st.session_state["sequence"]:
            st.markdown("**Fetched sequence**")
            st.markdown(
                f'<pre class="sequence-display">{html.escape(st.session_state["sequence"])}</pre>',
                unsafe_allow_html=True,
            )
    else:
        sequence_input = st.text_area(
            "Amino acid sequence (FASTA or raw)",
            height=140,
            placeholder=">MyProtein\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF...",
        )
        protein_name_input = st.text_input(
            "Protein name (optional)",
            placeholder="e.g. Hemoglobin alpha chain",
        )

    st.markdown("---")
    st.markdown("## ⚙️ Folding Settings")

    if FOLDING_BACKEND in ("simplefold", "modal"):
        # Only the 100M model is pre-loaded in the Modal volume. Larger sizes
        # would need separate downloads and don't fit on an A10G anyway.
        model_size = "simplefold_100M"
        st.markdown(
            '<p style="color:#8b949e;font-size:0.85rem;margin:0 0 0.5rem 0;">'
            'Model: <strong style="color:#58a6ff;">SimpleFold 100M</strong> '
            '· diffusion flow-matching</p>',
            unsafe_allow_html=True,
        )
        num_steps = st.slider(
            "Denoising steps",
            50,
            300,
            200,
            50,
            help=(
                "SimpleFold is a diffusion model — it starts from noise and "
                "iteratively denoises into a 3D structure. More steps = slightly "
                "higher quality, linearly slower. 200 works well for the 100M model."
            ),
        )
    else:
        model_size = "esm_default"
        num_steps = 0

    st.markdown("---")
    st.markdown("## 🎨 Visualization")

    color_scheme = st.selectbox(
        "Color scheme",
        ["Spectrum (Rainbow)", "Confidence (pLDDT)", "Secondary Structure", "Hydrophobicity", "Monochrome Blue"],
    )

    st.markdown("---")
    st.markdown("## 🤖 AI Mode")

    ai_mode = st.radio(
        "Explanation mode",
        ["🔬 Scientific Explanation", "🏥 MSL / Medical Affairs"],
        help="Scientific: deep structural biology. MSL: HCP-ready talking points.",
    )

    st.markdown("---")

    # Main fold button
    fold_clicked = st.button("⚡ FOLD & ANALYZE", type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# Main logic — Fold on button click
# ══════════════════════════════════════════════════════════════════════════════

if fold_clicked:
    # Get sequence
    if input_mode == "Paste sequence":
        raw_seq = sequence_input if sequence_input else ""
        pname = protein_name_input or "Unknown protein"
    else:
        # Quick examples + UniProt search both use session state
        raw_seq = st.session_state.get("sequence") or ""
        pname = st.session_state.get("protein_name") or "Unknown protein"

    if not raw_seq.strip():
        st.error("Please enter or fetch a protein sequence before folding.")
        st.stop()

    is_valid, result = validate_sequence(raw_seq)
    if not is_valid:
        st.error(f"Sequence validation failed: {result}")
        st.stop()

    clean_seq = result
    st.session_state["sequence"] = clean_seq
    st.session_state["protein_name"] = pname

    # Run folding
    progress_bar = st.progress(0, text="⚡ Initializing folding engine...")
    status_text = st.empty()

    try:
        status_text.info(f"🧬 Folding **{pname}** ({len(clean_seq)} amino acids)…  \n"
                         f"Engine: {backend_label}")
        progress_bar.progress(20, text="Running structure prediction...")

        pdb_string, confidence = fold_sequence(
            clean_seq,
            model=model_size,
            num_steps=num_steps,
        )

        progress_bar.progress(60, text="Structure predicted — generating AI explanation...")
        st.session_state["pdb_string"] = pdb_string
        st.session_state["confidence"] = confidence

        # Generate explanations (both modes pre-generated for instant tab switching)
        status_text.info("🤖 Generating scientific explanation via Venice AI...")
        explanation = get_scientific_explanation(clean_seq, pname)
        st.session_state["explanation"] = explanation
        progress_bar.progress(80, text="Generating MSL summary...")

        msl = get_msl_summary(clean_seq, pname)
        st.session_state["msl_summary"] = msl
        progress_bar.progress(100, text="Done!")

        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Structure predicted successfully!  Confidence: **{confidence:.1%}** &nbsp;|&nbsp; Length: **{len(clean_seq)} AA**")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Folding failed: {str(e)}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Main content area
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["pdb_string"]:
    pdb = st.session_state["pdb_string"]
    confidence = st.session_state["confidence"] or 0.0
    seq = st.session_state["sequence"] or ""
    pname = st.session_state["protein_name"] or "Unknown"

    col_viewer, col_explain = st.columns([6, 4], gap="large")

    # ── 3D Viewer ──────────────────────────────────────────────────────────────
    with col_viewer:
        st.markdown("### 🔬 3D Structure Viewer")
        st.markdown(f"""
        <div class="stat-row">
            <span class="stat-pill">📏 {len(seq)} amino acids</span>
            <span class="stat-pill">🎯 Confidence: {confidence:.1%}</span>
            <span class="stat-pill">🎨 {color_scheme}</span>
        </div>
        """, unsafe_allow_html=True)

        if VIZ_AVAILABLE:
            render_protein_3d(pdb, color_scheme=color_scheme)
            st.caption("🖱️ Drag to rotate · Scroll to zoom · Right-click to pan")
        else:
            st.warning("**py3Dmol / stmol not installed.** Run `pip install stmol py3Dmol` to enable 3D viewer.")
            st.download_button(
                "⬇️ Download PDB file",
                data=pdb,
                file_name=f"{pname.replace(' ', '_')}.pdb",
                mime="text/plain",
            )

        # Confidence meter
        st.markdown(f"""
        <div style="margin-top: 1rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e; margin-bottom:4px;">
                <span>Model confidence (pLDDT)</span>
                <span>{confidence:.1%}</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{min(confidence*100, 100):.0f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download PDB
        st.download_button(
            "⬇️ Download PDB",
            data=pdb,
            file_name=f"{pname.replace(' ', '_')}.pdb",
            mime="text/plain",
            use_container_width=True,
        )

    # ── AI Explanations ────────────────────────────────────────────────────────
    with col_explain:
        st.markdown(f"### 🤖 AI Analysis — *{pname}*")

        tab_sci, tab_msl = st.tabs(["🔬 Scientific", "🏥 MSL Mode"])

        with tab_sci:
            st.markdown('<div class="mode-badge">🔬 Scientific Explanation</div>', unsafe_allow_html=True)
            if st.session_state["explanation"]:
                st.markdown(f'<div class="explanation-box">{st.session_state["explanation"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.info("Run folding to generate explanation.")

        with tab_msl:
            st.markdown('<div class="mode-badge mode-badge-msl">🏥 Medical Affairs / MSL Mode</div>',
                        unsafe_allow_html=True)
            if st.session_state["msl_summary"]:
                st.markdown(f'<div class="explanation-box">{st.session_state["msl_summary"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.info("Run folding to generate MSL summary.")

        # Sequence info card
        with st.expander("🧬 Sequence details"):
            st.code(seq[:200] + ("..." if len(seq) > 200 else ""), language=None)
            st.caption(f"Full length: {len(seq)} amino acids")

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #8b949e;">
        <div style="font-size:4rem; margin-bottom:1rem;">🔬</div>
        <h2 style="color:#58a6ff; font-weight:600;">Ready to fold</h2>
        <p style="max-width:500px; margin:0 auto; line-height:1.7;">
            <strong style="color:#e6edf3;">Insulin</strong> is already loaded in the sidebar under
            <em>Quick examples</em>. Click
            <strong style="color:#e6edf3;">⚡ FOLD & ANALYZE</strong> to run on your deployed backend.
        </p>
        <br>
        <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; margin-top:1rem;">
            <span style="background:#1f2d40; border:1px solid #1f6feb50; padding:0.4rem 1rem; border-radius:20px; font-size:0.85rem; color:#58a6ff;">☁️ Modal GPU</span>
            <span style="background:#1f2d40; border:1px solid #1f6feb50; padding:0.4rem 1rem; border-radius:20px; font-size:0.85rem; color:#58a6ff;">🤖 Venice AI</span>
            <span style="background:#1f2d40; border:1px solid #1f6feb50; padding:0.4rem 1rem; border-radius:20px; font-size:0.85rem; color:#58a6ff;">🧊 3D py3Dmol</span>
            <span style="background:#1f2d40; border:1px solid #1f6feb50; padding:0.4rem 1rem; border-radius:20px; font-size:0.85rem; color:#58a6ff;">🏥 MSL Mode</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🧪 Example proteins (sidebar → Quick examples)")
    example_cols = st.columns(3)
    for i, preset in enumerate(EXAMPLE_PROTEINS):
        col = example_cols[i % 3]
        with col:
            if st.button(
                f"**{preset['name'].split('(')[0].strip()}**\n\n{len(preset['sequence'])} AA",
                key=f"home_preset_{preset['id']}",
                use_container_width=True,
            ):
                st.session_state["sequence"] = preset["sequence"]
                st.session_state["protein_name"] = preset["name"]
                st.session_state["selected_preset_id"] = preset["id"]
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Footer — Sundai Club attribution
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
_footer_inner = (
    f'<img src="{_sundai_uri}" alt="Sundai Club" style="height:56px;"/>'
    if _sundai_uri else '<div style="font-size:2rem;">🍦</div>'
)
st.markdown(f"""
<div style="text-align:center; padding: 2rem 1rem 3rem 1rem; color:#8b949e;">
  <a href="{_SUNDAI_URL}" target="_blank" style="text-decoration:none; display:inline-flex;
     flex-direction:column; align-items:center; gap:0.5rem; color:#8b949e;">
    {_footer_inner}
    <div style="font-size:0.95rem;">
      Developed at <strong style="color:#f5b400; letter-spacing:0.5px;">SUNDAI CLUB</strong>
    </div>
    <div style="font-size:0.75rem; color:#6e7681;">
      A community of builders shipping AI projects · sundai.club
    </div>
  </a>
</div>
""", unsafe_allow_html=True)
