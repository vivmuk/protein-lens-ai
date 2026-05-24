"""
ProteinLens AI — Main Streamlit Application
AI-powered protein folding, visualization, and scientific explanation.

Atelier-style light UI with: 3D viewer (direct py3Dmol), sequence builder,
always-on sequence insights, per-residue pLDDT heatmap, and unknown-peptide
AI explanations grounded in computed stats.
"""

from __future__ import annotations

import base64
import html
import os
import time
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from aa_insights import (
    comparative_framing,
    compute_basic_stats,
    format_for_prompt,
    kyte_doolittle_profile,
    scan_motifs,
)
from explanation import get_msl_summary, get_scientific_explanation
from folding import FOLDING_BACKEND, fetch_uniprot_sequence, fold_sequence
from presets import (
    EXAMPLE_PROTEINS,
    find_preset_by_label,
    get_default_preset,
    preset_select_label,
)

# ── Optional 3D viewer (graceful degradation if py3Dmol is missing) ──────────
try:
    import py3Dmol  # type: ignore

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# ══════════════════════════════════════════════════════════════════════════════
# Page config + theme
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ProteinLens",
    page_icon="🜂",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#D9583C"        # the "one fold" italic orange
INK = "#1a1a1a"           # primary text
INK_2 = "#52524c"         # secondary text
INK_3 = "#8a8a82"         # muted
PAPER = "#F4F1EC"         # background cream
PAPER_2 = "#FFFFFF"       # card surface
RULE = "#E5E1DA"          # subtle dividers / pills
PLDDT_HI = "#1d6b3a"      # green (>90)
PLDDT_MID = "#9b7c1f"     # amber (70-90)
PLDDT_LO = "#a83d2c"      # red (<70)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=JetBrains+Mono:wght@400;500&display=swap');

      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
      .stApp {{ background-color: {PAPER}; }}
      .main .block-container {{ padding-top: 1rem; max-width: 1280px; }}

      /* Hide Streamlit chrome we don't need */
      #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}

      /* ── Top nav (decorative) ── */
      .pl-nav {{
        display:flex; align-items:center; padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid {RULE}; margin-bottom: 2.5rem;
      }}
      .pl-nav .brand {{
        font-family: 'EB Garamond', serif; font-style: italic;
        font-size: 1.4rem; color: {INK}; font-weight: 500;
        display:flex; align-items:center; gap: 0.6rem;
      }}
      .pl-nav .dot {{
        width: 14px; height: 14px; border-radius: 50%; background: {INK};
        display:inline-block;
      }}
      .pl-nav .links {{ margin: 0 auto; display:flex; gap: 2.2rem; }}
      .pl-nav .links a {{
        color: {INK_2}; text-decoration: none; font-size: 0.92rem;
      }}
      .pl-nav .links a.active {{ color: {INK}; font-weight: 500; }}
      .pl-nav .meta {{ color: {INK_3}; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; }}

      /* ── Hero typography ── */
      .pl-eyebrow {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        color: {INK_3}; letter-spacing: 0.12em; text-transform: uppercase;
        margin-bottom: 1.5rem;
      }}
      .pl-headline {{
        font-family: 'EB Garamond', serif; font-weight: 500;
        font-size: 4rem; line-height: 1.05; color: {INK};
        margin: 0 0 1.5rem 0;
      }}
      .pl-headline .accent {{ color: {ACCENT}; font-style: italic; font-weight: 500; }}
      .pl-tagline {{
        font-size: 1rem; color: {INK_2}; line-height: 1.6;
        max-width: 460px; margin: 0 0 2rem 0;
      }}

      /* ── Card (white surface, hairline border) ── */
      .pl-card {{
        background: {PAPER_2};
        border: 1px solid {RULE};
        border-radius: 14px;
        padding: 1.75rem 1.75rem;
        box-shadow: 0 1px 0 rgba(0,0,0,0.02);
      }}
      .pl-card-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; color: {INK_3};
        letter-spacing: 0.08em; text-transform: uppercase;
      }}
      .pl-card-title {{
        font-family: 'EB Garamond', serif; font-weight: 500;
        font-size: 1.6rem; line-height: 1.2; color: {INK};
        margin: 0.4rem 0 0.9rem 0;
      }}
      .pl-card-body {{
        font-size: 0.93rem; color: {INK_2}; line-height: 1.6;
      }}

      /* ── Pill chips ── */
      .pl-pill-row {{ display:flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }}
      .pl-pill {{
        background: {RULE}50;
        border: 1px solid {RULE};
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem; color: {INK_2};
        font-family: 'JetBrains Mono', monospace;
      }}

      /* ── pLDDT bar ── */
      .pl-plddt-row {{
        display:flex; justify-content:space-between; align-items:center;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        color: {INK_2}; margin-top: 1rem;
      }}
      .pl-plddt-bar {{
        height: 4px; background: {RULE}; border-radius: 2px;
        margin: 0.35rem 0; overflow: hidden;
      }}
      .pl-plddt-fill {{
        height: 100%; background: {ACCENT}; border-radius: 2px;
      }}

      /* ── Per-residue heatmap strip ── */
      .pl-heatmap {{
        display:flex; gap: 1px; height: 22px; width: 100%; margin: 0.6rem 0 0.4rem 0;
        border-radius: 3px; overflow: hidden;
      }}
      .pl-heatmap-cell {{ flex: 1 1 0; min-width: 2px; }}
      .pl-heatmap-axis {{
        display:flex; justify-content:space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: {INK_3};
      }}

      /* ── Composition bar ── */
      .pl-comp-row {{ display:flex; align-items:center; gap: 0.6rem; margin: 0.3rem 0; }}
      .pl-comp-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; width: 90px; color: {INK_2};
      }}
      .pl-comp-bar {{
        flex: 1; height: 8px; background: {RULE}; border-radius: 4px; overflow: hidden;
      }}
      .pl-comp-fill {{ height: 100%; background: {INK}; }}
      .pl-comp-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; width: 48px; text-align: right; color: {INK_3};
      }}

      /* ── Sidebar ── */
      section[data-testid="stSidebar"] {{
        background-color: {PAPER}; border-right: 1px solid {RULE};
      }}
      section[data-testid="stSidebar"] .stMarkdown h2,
      section[data-testid="stSidebar"] .stMarkdown h3 {{
        font-family: 'EB Garamond', serif; font-weight: 500;
        color: {INK}; font-size: 1.25rem;
      }}
      section[data-testid="stSidebar"] .stMarkdown p,
      section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stRadio label {{
        color: {INK_2} !important;
      }}

      /* ── Inputs ── */
      .stTextArea textarea, .stTextInput input {{
        background: {PAPER_2} !important;
        border: 1px solid {RULE} !important;
        color: {INK} !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 8px;
        caret-color: {ACCENT};
      }}
      .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {INK_3} !important; outline: none !important;
      }}
      .stTextArea textarea::placeholder, .stTextInput input::placeholder {{
        color: {INK_3} !important; opacity: 1 !important;
      }}

      /* Selectbox */
      [data-baseweb="select"] > div {{
        background: {PAPER_2} !important;
        border: 1px solid {RULE} !important;
        color: {INK} !important;
      }}
      [data-baseweb="popover"] li {{
        background: {PAPER_2} !important; color: {INK} !important;
      }}
      [data-baseweb="popover"] li:hover {{ background: {RULE} !important; }}

      /* Generic button — quieter than the old gradient */
      div[data-testid="stButton"] > button {{
        background: {PAPER_2}; color: {INK} !important;
        border: 1px solid {RULE} !important;
        border-radius: 8px; padding: 0.5rem 1.1rem; font-weight: 500;
        transition: all 0.15s;
      }}
      div[data-testid="stButton"] > button:hover {{
        border-color: {INK} !important;
        background: {PAPER_2};
      }}
      /* Primary button — the FOLD action */
      div[data-testid="stButton"] > button[kind="primary"] {{
        background: {INK} !important; color: {PAPER_2} !important;
        border-color: {INK} !important;
      }}
      div[data-testid="stButton"] > button[kind="primary"]:hover {{
        background: {ACCENT} !important; border-color: {ACCENT} !important;
      }}

      /* Sequence builder AA chip buttons */
      .pl-aa-grid div[data-testid="stButton"] > button {{
        padding: 0.45rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
      }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{
        background: transparent; gap: 0.25rem; border-bottom: 1px solid {RULE};
      }}
      .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {INK_3}; padding: 0.5rem 0.9rem;
        font-size: 0.92rem;
      }}
      .stTabs [aria-selected="true"] {{
        color: {INK} !important; border-bottom: 2px solid {ACCENT} !important;
      }}

      /* Code & sequence display */
      .pl-seq {{
        background: {PAPER_2}; border: 1px solid {RULE};
        border-radius: 8px; padding: 0.7rem 0.9rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; color: {INK}; word-break: break-all;
        line-height: 1.6; max-height: 130px; overflow-y: auto;
      }}

      /* Markdown headings in explanation panels */
      .pl-explain h1, .pl-explain h2, .pl-explain h3 {{
        font-family: 'EB Garamond', serif; font-weight: 500; color: {INK};
        margin-top: 1.2rem; margin-bottom: 0.4rem;
      }}
      .pl-explain h2 {{ font-size: 1.25rem; }}
      .pl-explain h3 {{ font-size: 1.05rem; }}
      .pl-explain p, .pl-explain li {{
        color: {INK_2}; line-height: 1.65; font-size: 0.92rem;
      }}
      .pl-explain code {{
        background: {RULE}80; padding: 1px 5px; border-radius: 3px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.84em;
      }}

      /* Toast-y status messages */
      .stSuccess {{ background: #eaf3ec !important; border: 1px solid #c7decd !important; color: #1d6b3a !important; }}
      .stInfo    {{ background: #eef0f3 !important; border: 1px solid {RULE} !important; color: {INK} !important; }}
      .stWarning {{ background: #fbf2e6 !important; border: 1px solid #e6d0a8 !important; color: #7a5212 !important; }}
      .stError   {{ background: #f9e9e6 !important; border: 1px solid #e9c2b9 !important; color: #8a2a1a !important; }}

      hr {{ border-color: {RULE}; }}

      /* Slider */
      .stSlider [role="slider"] {{ background: {INK} !important; }}

      /* Footer */
      .pl-footer {{
        text-align:center; padding: 3rem 1rem 1.5rem 1rem;
        color: {INK_3}; font-size: 0.82rem;
        border-top: 1px solid {RULE}; margin-top: 4rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def render_protein_3d(pdb_string: str, color_scheme: str, bg_color: str = "#FFFFFF") -> None:
    """Render py3Dmol view directly via components.html — no stmol dependency."""
    if not HAS_VIZ:
        st.warning(
            "**py3Dmol not installed.** Run `pip install py3Dmol` to enable the 3D viewer."
        )
        return

    view = py3Dmol.view(width=720, height=460)
    view.addModel(pdb_string, "pdb")

    style_map = {
        "Spectrum (Rainbow)": {"cartoon": {"color": "spectrum"}},
        "Confidence (pLDDT)": {
            "cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": 50, "max": 100}}
        },
        "Secondary Structure": {"cartoon": {"color": "ssJmol"}},
        "Hydrophobicity": {"cartoon": {"color": "hydrophobicity"}},
        "Monochrome": {"cartoon": {"color": "#1a1a1a"}},
    }
    view.setStyle(style_map.get(color_scheme, {"cartoon": {"color": "spectrum"}}))
    view.setBackgroundColor(bg_color)
    view.zoomTo()
    components.html(view._make_html(), height=480, scrolling=False)


def validate_sequence(seq: str) -> tuple[bool, str]:
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")
    seq_clean = seq.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if seq_clean.startswith(">"):
        seq_clean = "".join(seq_clean.split("\n")[1:])
    invalid = set(seq_clean) - valid_aa
    if invalid:
        return False, f"Invalid characters: {', '.join(sorted(invalid))}"
    if len(seq_clean) < 10:
        return False, "Sequence too short (minimum 10 amino acids)."
    if len(seq_clean) > 1400:
        return False, f"Sequence too long ({len(seq_clean)} AA). Max is 1400."
    return True, seq_clean.upper()


def extract_per_residue_plddt(pdb_string: str) -> list[float]:
    """Average the B-factor column across all atoms of each residue. pLDDT lives there."""
    per_res: dict[tuple[str, int], list[float]] = {}
    order: list[tuple[str, int]] = []
    for line in pdb_string.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            chain = line[21:22]
            res_id = int(line[22:26].strip())
            b = float(line[60:66].strip())
        except (ValueError, IndexError):
            continue
        key = (chain, res_id)
        if key not in per_res:
            per_res[key] = []
            order.append(key)
        per_res[key].append(b)
    return [sum(vals) / len(vals) for k in order for vals in [per_res[k]]]


def plddt_color(v: float) -> str:
    if v >= 90:
        return PLDDT_HI
    if v >= 70:
        return PLDDT_MID
    if v >= 50:
        return "#c08a3a"
    return PLDDT_LO


def render_plddt_heatmap(plddt: list[float]) -> str:
    if not plddt:
        return f'<div class="pl-card-body">No per-residue confidence available.</div>'
    cells = "".join(
        f'<div class="pl-heatmap-cell" style="background:{plddt_color(v)};" '
        f'title="Residue {i+1}: pLDDT {v:.1f}"></div>'
        for i, v in enumerate(plddt)
    )
    n = len(plddt)
    mid = n // 2
    return (
        f'<div class="pl-heatmap">{cells}</div>'
        f'<div class="pl-heatmap-axis"><span>1</span><span>{mid+1}</span><span>{n}</span></div>'
    )


def render_composition_bars(comp_pct: dict[str, float], top_n: int = 8) -> str:
    items = sorted(comp_pct.items(), key=lambda kv: -kv[1])[:top_n]
    max_pct = max((v for _, v in items), default=1) or 1
    rows = []
    for aa, pct in items:
        width = max(2, (pct / max_pct) * 100)
        rows.append(
            f'<div class="pl-comp-row">'
            f'<div class="pl-comp-label">{aa}</div>'
            f'<div class="pl-comp-bar"><div class="pl-comp-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="pl-comp-value">{pct:.1f}%</div>'
            f"</div>"
        )
    return "".join(rows)


def _img_data_uri(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("pdb_string", None),
    ("explanation", None),
    ("msl_summary", None),
    ("protein_name", None),
    ("sequence", None),
    ("confidence", None),
    ("build_seq", ""),
    ("selected_preset_id", None),
    ("input_mode", "Quick examples"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state["sequence"]:
    _default = get_default_preset()
    st.session_state["sequence"] = _default["sequence"]
    st.session_state["protein_name"] = _default["name"]
    st.session_state["selected_preset_id"] = _default["id"]


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — auxiliary controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f'<div style="font-family:\'EB Garamond\',serif;font-style:italic;'
        f'font-size:1.4rem;color:{INK};margin-bottom:0.2rem;">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{INK};'
        f'border-radius:50%;margin-right:0.5rem;vertical-align:middle;"></span>'
        f"proteinlens</div>"
        f'<div style="font-size:0.78rem;color:{INK_3};margin-bottom:1.5rem;'
        f'font-family:\'JetBrains Mono\',monospace;letter-spacing:0.05em;">'
        f"controls · settings</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## Input source")
    input_mode = st.radio(
        "Input method",
        ["Quick examples", "Paste sequence", "Search UniProt", "Build sequence"],
        label_visibility="collapsed",
        key="input_mode",
    )

    sequence_input = ""
    protein_name_input = ""

    if input_mode == "Quick examples":
        preset_labels = [preset_select_label(p) for p in EXAMPLE_PROTEINS]
        default_idx = next(
            (i for i, p in enumerate(EXAMPLE_PROTEINS)
             if p["id"] == st.session_state.get("selected_preset_id")),
            0,
        )
        selected_label = st.selectbox("Choose a protein", preset_labels, index=default_idx)
        preset = find_preset_by_label(selected_label)
        if preset:
            st.session_state["sequence"] = preset["sequence"]
            st.session_state["protein_name"] = preset["name"]
            st.session_state["selected_preset_id"] = preset["id"]
            st.markdown(
                f'<p style="font-size:0.83rem;color:{INK_2};line-height:1.5;">{preset["description"]}</p>',
                unsafe_allow_html=True,
            )

    elif input_mode == "Paste sequence":
        sequence_input = st.text_area(
            "FASTA or raw sequence",
            height=140,
            placeholder=">MyProtein\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF...",
        )
        protein_name_input = st.text_input(
            "Protein name (optional)",
            placeholder="e.g. Hemoglobin alpha",
        )
        if sequence_input.strip():
            ok, cleaned = validate_sequence(sequence_input)
            if ok:
                st.session_state["sequence"] = cleaned
                st.session_state["protein_name"] = protein_name_input or "Unknown peptide"

    elif input_mode == "Search UniProt":
        query = st.text_input("Protein name", placeholder="e.g. insulin, BRCA1")
        if st.button("Fetch from UniProt", use_container_width=True) and query:
            with st.spinner("Querying UniProt..."):
                seq, name = fetch_uniprot_sequence(query)
            if seq:
                st.session_state["sequence"] = seq
                st.session_state["protein_name"] = name
                st.success(f"Loaded **{name}** ({len(seq)} AA)")
            else:
                st.error("Not found. Try a different name.")

    else:  # Build sequence
        st.markdown(
            f'<p style="font-size:0.83rem;color:{INK_2};line-height:1.5;">'
            f"Click amino acids to append. Minimum 10 to fold.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="pl-seq" style="min-height:60px;">{html.escape(st.session_state["build_seq"]) or "&nbsp;"}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Length: {len(st.session_state['build_seq'])} AA")

        # 20 standard AAs, grouped visually by category
        AA_GROUPS = [
            ("A", "Ala"), ("V", "Val"), ("L", "Leu"), ("I", "Ile"), ("M", "Met"),
            ("F", "Phe"), ("Y", "Tyr"), ("W", "Trp"), ("S", "Ser"), ("T", "Thr"),
            ("N", "Asn"), ("Q", "Gln"), ("C", "Cys"), ("G", "Gly"), ("P", "Pro"),
            ("K", "Lys"), ("R", "Arg"), ("H", "His"), ("D", "Asp"), ("E", "Glu"),
        ]
        st.markdown('<div class="pl-aa-grid">', unsafe_allow_html=True)
        for row_start in range(0, 20, 5):
            cols = st.columns(5)
            for i, (code, name) in enumerate(AA_GROUPS[row_start:row_start + 5]):
                if cols[i].button(code, key=f"aa_{code}", help=name, use_container_width=True):
                    st.session_state["build_seq"] += code
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        bcol1, bcol2 = st.columns(2)
        if bcol1.button("⌫ Backspace", use_container_width=True):
            st.session_state["build_seq"] = st.session_state["build_seq"][:-1]
            st.rerun()
        if bcol2.button("Clear", use_container_width=True):
            st.session_state["build_seq"] = ""
            st.rerun()

        protein_name_input = st.text_input(
            "Name (optional)", placeholder="e.g. Designed peptide #1"
        )
        if len(st.session_state["build_seq"]) >= 10:
            st.session_state["sequence"] = st.session_state["build_seq"].upper()
            st.session_state["protein_name"] = protein_name_input or "Unknown peptide"

    st.markdown("---")
    st.markdown("## Folding")
    if FOLDING_BACKEND in ("simplefold", "modal"):
        model_size = "simplefold_100M"
        st.markdown(
            f'<p style="font-size:0.83rem;color:{INK_2};margin-bottom:0.6rem;">'
            f"Model: <strong>SimpleFold 100M</strong> · diffusion flow-matching</p>",
            unsafe_allow_html=True,
        )
        num_steps = st.slider("Denoising steps", 50, 300, 200, 50,
                              help="More steps = marginally higher quality, linearly slower.")
    else:
        model_size = "esm_default"
        num_steps = 0

    st.markdown("## Visualization")
    color_scheme = st.selectbox(
        "Color scheme",
        ["Spectrum (Rainbow)", "Confidence (pLDDT)", "Secondary Structure",
         "Hydrophobicity", "Monochrome"],
    )

    st.markdown("---")
    fold_clicked = st.button("⚡  Fold & analyze", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Top nav (decorative)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="pl-nav">
      <div class="brand"><span class="dot"></span>proteinlens</div>
      <div class="links">
        <a class="active" href="#fold">Fold</a>
        <a href="#library">Library</a>
        <a href="#docs">Docs</a>
        <a href="#api">API</a>
      </div>
      <div class="meta">⌘K  ·  {html.escape((st.session_state["protein_name"] or "unknown")[:24])}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Hero row
# ══════════════════════════════════════════════════════════════════════════════

current_seq: str = (st.session_state["sequence"] or "").upper()
current_name: str = st.session_state["protein_name"] or "Unknown peptide"
has_fold: bool = bool(st.session_state["pdb_string"])

hero_left, hero_right = st.columns([5, 6], gap="large")

with hero_left:
    eyebrow = (
        f"{current_name[:32]} · {len(current_seq)} AA"
        if current_seq else "no sequence loaded"
    )
    st.markdown(f'<div class="pl-eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 class="pl-headline">One sequence.<br>'
        f'<span class="accent">One fold.</span><br>One story.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="pl-tagline">Paste a sequence. Get a structure, a confidence map, and a '
        "science-grade explanation — all in under a few minutes on cloud GPU.</p>",
        unsafe_allow_html=True,
    )

    # Inline FASTA preview + fold button (the mockup-style input bar)
    preview = current_seq[:48] + ("…" if len(current_seq) > 48 else "")
    st.markdown(
        f'<div style="display:flex;align-items:stretch;border:1px solid {RULE};'
        f"background:{PAPER_2};border-radius:10px;overflow:hidden;margin-bottom:0.75rem;\">"
        f"<div style=\"flex:1;padding:0.85rem 1rem;font-family:'JetBrains Mono',monospace;"
        f'font-size:0.85rem;color:{INK};overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">'
        f'<span style="color:{INK_3};margin-right:0.6rem;">&gt;</span>{html.escape(preview) or "<em>empty</em>"}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "paste fasta  ·  search uniprot  ·  build sequence  ·  use the sidebar →"
    )

with hero_right:
    st.markdown('<div class="pl-card" style="padding:1.5rem;">', unsafe_allow_html=True)

    # Stat pills
    backend_label = {
        "simplefold": "SimpleFold 100M · local MLX",
        "modal": "SimpleFold 100M · Modal GPU",
        "esm": "ESMFold · API",
    }.get(FOLDING_BACKEND, FOLDING_BACKEND)
    conf_pct = (st.session_state["confidence"] or 0) * 100
    pills_html = (
        f'<span class="pl-pill">{len(current_seq)} AA</span>'
        f'<span class="pl-pill">{html.escape(backend_label)}</span>'
        f'<span class="pl-pill">{html.escape(color_scheme.lower())}</span>'
    )
    st.markdown(f'<div class="pl-pill-row">{pills_html}</div>', unsafe_allow_html=True)

    if has_fold:
        render_protein_3d(st.session_state["pdb_string"], color_scheme=color_scheme)
        st.markdown(
            f'<div class="pl-plddt-row"><span>pLDDT</span><span>{conf_pct:.1f}%</span></div>'
            f'<div class="pl-plddt-bar"><div class="pl-plddt-fill" style="width:{min(conf_pct,100):.0f}%"></div></div>',
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns([1, 1])
        with d1:
            st.download_button(
                "↓  PDB",
                data=st.session_state["pdb_string"],
                file_name=f"{current_name.replace(' ', '_')}.pdb",
                mime="text/plain",
                use_container_width=True,
            )
        with d2:
            st.markdown(
                f'<div style="text-align:right;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:0.78rem;color:{INK_3};padding-top:0.4rem;">'
                f"⌘ share</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div style="height:340px;display:flex;align-items:center;justify-content:center;'
            f'color:{INK_3};font-family:\'EB Garamond\',serif;font-style:italic;font-size:1.2rem;">'
            f'click <strong style="color:{INK};font-style:normal;font-family:Inter,sans-serif;'
            f"font-weight:500;\">⚡ Fold & analyze</strong>&nbsp; in the sidebar"
            f"</div>"
            f'<div class="pl-plddt-row"><span>pLDDT</span><span>—</span></div>'
            f'<div class="pl-plddt-bar"></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fold action — runs on sidebar button
# ══════════════════════════════════════════════════════════════════════════════

if fold_clicked:
    raw = current_seq
    if not raw.strip():
        st.error("Please enter or load a sequence before folding.")
        st.stop()
    ok, result = validate_sequence(raw)
    if not ok:
        st.error(f"Sequence validation failed: {result}")
        st.stop()
    clean = result
    st.session_state["sequence"] = clean
    pname = current_name

    progress = st.progress(0, text="Initializing folding engine...")
    status = st.empty()
    try:
        status.info(f"Folding **{pname}** ({len(clean)} AA) via {backend_label}…")
        progress.progress(20, text="Running diffusion structure prediction...")

        pdb_string, confidence = fold_sequence(clean, model=model_size, num_steps=num_steps)
        st.session_state["pdb_string"] = pdb_string
        st.session_state["confidence"] = confidence
        progress.progress(55, text="Structure ready — computing sequence insights...")

        stats = compute_basic_stats(clean)
        motifs = scan_motifs(clean)
        framing = comparative_framing(stats)
        insights_block = format_for_prompt(stats, motifs, framing)

        progress.progress(65, text="Generating scientific explanation...")
        st.session_state["explanation"] = get_scientific_explanation(
            clean, pname, confidence=confidence, insights_block=insights_block
        )
        progress.progress(85, text="Generating MSL summary...")
        st.session_state["msl_summary"] = get_msl_summary(
            clean, pname, confidence=confidence, insights_block=insights_block
        )
        progress.progress(100, text="Done.")
        time.sleep(0.4)
        progress.empty()
        status.empty()
        st.success(
            f"Structure predicted. Confidence: **{confidence:.1%}** · Length: **{len(clean)} AA**"
        )
        st.rerun()
    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Folding failed: {e}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Lower row — three cards (Scientific / Medical / Analytics)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)

# Always-on insights (cheap, no API)
if current_seq and len(current_seq) >= 10:
    stats = compute_basic_stats(current_seq)
    motifs = scan_motifs(current_seq)
    framing = comparative_framing(stats)
    hydro_profile = kyte_doolittle_profile(current_seq, window=9)
else:
    stats, motifs, framing, hydro_profile = None, [], "", []

card1, card2, card3 = st.columns(3, gap="large")

with card1:
    st.markdown(
        '<div class="pl-card">'
        '<div class="pl-card-num">01 / SCIENTIFIC</div>'
        '<div class="pl-card-title">What this protein does</div>',
        unsafe_allow_html=True,
    )
    if st.session_state["explanation"]:
        excerpt = st.session_state["explanation"]
        st.markdown(f'<div class="pl-card-body pl-explain">{excerpt}</div>',
                    unsafe_allow_html=True)
    elif framing:
        st.markdown(
            f'<div class="pl-card-body pl-explain"><em>Sequence-level read (pre-fold):</em><br>{html.escape(framing)}'
            f'<br><br><span style="color:{INK_3};">Run a fold for the full AI-grade scientific analysis.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="pl-card-body">Load a sequence to begin.</div>',
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with card2:
    st.markdown(
        '<div class="pl-card">'
        '<div class="pl-card-num">02 / MEDICAL</div>'
        '<div class="pl-card-title">For the field</div>',
        unsafe_allow_html=True,
    )
    if st.session_state["msl_summary"]:
        st.markdown(
            f'<div class="pl-card-body pl-explain">{st.session_state["msl_summary"]}</div>',
            unsafe_allow_html=True,
        )
    elif stats:
        motif_summary = (
            f"{len(motifs)} motif hit{'s' if len(motifs) != 1 else ''}"
            if motifs else "no canonical motifs detected"
        )
        st.markdown(
            f'<div class="pl-card-body">{motif_summary} · '
            f"MW {stats['molecular_weight_kda']:.1f} kDa · pI {stats['isoelectric_point']:.2f}<br><br>"
            f'<span style="color:{INK_3};">MSL-ready briefing appears after folding.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="pl-card-body">Load a sequence to begin.</div>',
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with card3:
    st.markdown(
        '<div class="pl-card">'
        '<div class="pl-card-num">03 / ANALYTICS</div>'
        '<div class="pl-card-title">Per-residue confidence</div>',
        unsafe_allow_html=True,
    )
    if has_fold:
        plddt = extract_per_residue_plddt(st.session_state["pdb_string"])
        st.markdown(render_plddt_heatmap(plddt), unsafe_allow_html=True)
        legend = (
            f'<div style="display:flex;gap:1rem;margin-top:0.6rem;font-size:0.72rem;'
            f'color:{INK_3};font-family:\'JetBrains Mono\',monospace;">'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_HI};'
            f'border-radius:2px;margin-right:4px;"></span>&gt;90 very high</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_MID};'
            f'border-radius:2px;margin-right:4px;"></span>70–90 confident</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_LO};'
            f'border-radius:2px;margin-right:4px;"></span>&lt;70 low</span>'
            f"</div>"
        )
        st.markdown(legend, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="pl-card-body" style="color:{INK_3};">Per-residue pLDDT heatmap appears after folding.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sequence insights (always-on)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

ins_left, ins_right = st.columns([1, 1], gap="large")

with ins_left:
    st.markdown(
        '<div class="pl-card">'
        '<div class="pl-card-num">SEQUENCE INSIGHTS</div>'
        '<div class="pl-card-title">Physicochemical character</div>',
        unsafe_allow_html=True,
    )
    if stats:
        stat_rows = [
            ("Length", f"{stats['length']} AA"),
            ("Molecular weight", f"{stats['molecular_weight_kda']:.2f} kDa"),
            ("Isoelectric point (pI)", f"{stats['isoelectric_point']:.2f}"),
            ("Net charge @ pH 7", f"{stats['net_charge_at_ph7']:+.2f}"),
            ("Hydropathy (GRAVY)", f"{stats['gravy']:+.2f}"),
            ("Aromaticity", f"{stats['aromaticity'] * 100:.1f}%"),
            ("Instability index", f"{stats['instability_index']:.1f}"),
            ("ε280 (reduced)", f"{stats['extinction_coefficient_280nm']:,} M⁻¹cm⁻¹"),
        ]
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:0.35rem 0;'
            f'border-bottom:1px dashed {RULE};font-size:0.88rem;">'
            f'<span style="color:{INK_2};">{k}</span>'
            f'<span style="color:{INK};font-family:\'JetBrains Mono\',monospace;">{v}</span></div>'
            for k, v in stat_rows
        )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown(
            f'<p style="margin-top:1rem;font-size:0.88rem;color:{INK_2};line-height:1.6;'
            f'font-style:italic;">{html.escape(framing)}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="pl-card-body" style="color:{INK_3};">No sequence loaded yet.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with ins_right:
    st.markdown(
        '<div class="pl-card">'
        '<div class="pl-card-num">COMPOSITION</div>'
        '<div class="pl-card-title">Amino acid breakdown</div>',
        unsafe_allow_html=True,
    )
    if stats:
        cats = stats["aa_category_pct"]
        cat_chips = " ".join(
            f'<span class="pl-pill">{label}: {cats.get(key, 0):.0f}%</span>'
            for key, label in [
                ("hydrophobic", "hydrophobic"),
                ("polar_uncharged", "polar"),
                ("positive", "+ charge"),
                ("negative", "− charge"),
                ("aromatic", "aromatic"),
                ("special", "C/G/P"),
            ]
        )
        st.markdown(f'<div class="pl-pill-row">{cat_chips}</div>', unsafe_allow_html=True)
        st.markdown(
            render_composition_bars(stats["aa_composition_pct"], top_n=10),
            unsafe_allow_html=True,
        )
        if hydro_profile:
            st.markdown(
                f'<p style="margin-top:1rem;font-size:0.78rem;color:{INK_3};'
                f"font-family:'JetBrains Mono',monospace;letter-spacing:0.08em;\">KYTE-DOOLITTLE HYDROPATHY</p>",
                unsafe_allow_html=True,
            )
            st.line_chart(
                {"hydropathy": hydro_profile},
                height=140,
                use_container_width=True,
            )
        if motifs:
            st.markdown(
                f'<p style="margin-top:0.8rem;font-size:0.78rem;color:{INK_3};'
                f"font-family:'JetBrains Mono',monospace;letter-spacing:0.08em;\">"
                f"MOTIF HITS ({len(motifs)})</p>",
                unsafe_allow_html=True,
            )
            top_motifs = motifs[:8]
            motif_rows = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;'
                f'border-bottom:1px dashed {RULE};font-size:0.83rem;">'
                f'<span style="color:{INK_2};">{html.escape(m["name"])}</span>'
                f'<span style="color:{INK};font-family:\'JetBrains Mono\',monospace;">'
                f'{m["start"]+1}–{m["end"]} · {html.escape(m["match"][:14])}</span></div>'
                for m in top_motifs
            )
            st.markdown(motif_rows, unsafe_allow_html=True)
            if len(motifs) > 8:
                st.caption(f"+{len(motifs) - 8} more")
    else:
        st.markdown(
            f'<div class="pl-card-body" style="color:{INK_3};">No sequence loaded.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Full AI explanations (tabs)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["explanation"] or st.session_state["msl_summary"]:
    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="pl-card">'
        '<div class="pl-card-num">AI ANALYSIS</div>'
        f'<div class="pl-card-title">{html.escape(current_name)}</div>',
        unsafe_allow_html=True,
    )
    tab_sci, tab_msl, tab_seq = st.tabs(["Scientific", "MSL / Medical Affairs", "Sequence"])
    with tab_sci:
        if st.session_state["explanation"]:
            st.markdown(
                f'<div class="pl-explain">{st.session_state["explanation"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Fold the sequence to generate a scientific explanation.")
    with tab_msl:
        if st.session_state["msl_summary"]:
            st.markdown(
                f'<div class="pl-explain">{st.session_state["msl_summary"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Fold the sequence to generate an MSL briefing.")
    with tab_seq:
        st.markdown(
            f'<div class="pl-seq" style="max-height:300px;">{html.escape(current_seq)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{len(current_seq)} amino acids · single-letter IUPAC")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# How it works — model explainer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)
st.markdown(
    f'<div class="pl-card-num">HOW IT WORKS</div>'
    f'<h2 style="font-family:\'EB Garamond\',serif;font-weight:500;font-size:2rem;'
    f'color:{INK};margin:0.4rem 0 1.5rem 0;">From letters to a 3D protein</h2>',
    unsafe_allow_html=True,
)

how_cols = st.columns(4, gap="large")
how_steps = [
    ("01", "Sequence in",
     "Your amino acid string (FASTA or built letter-by-letter) is validated and length-checked. "
     "We compute physicochemical character — MW, pI, charge, hydropathy, motifs — before any model runs."),
    ("02", "Language model embedding",
     "ESM-2 (Meta AI) reads the sequence and emits a per-residue embedding capturing evolutionary "
     "context. This is the same protein language model used to seed many state-of-the-art folders."),
    ("03", "Flow-matching diffusion",
     "SimpleFold (Apple, 2025) starts from random 3D coordinates and iteratively denoises them, "
     "conditioned on the ESM embedding, over N denoising steps. Output: per-atom coordinates."),
    ("04", "Confidence + explanation",
     "A pLDDT confidence head scores each residue's local accuracy (0–100, stored in PDB B-factors). "
     "Venice AI synthesizes a domain-appropriate explanation using your sequence stats as ground truth."),
]
for col, (num, title, body) in zip(how_cols, how_steps):
    with col:
        st.markdown(
            f'<div class="pl-card-num">{num}</div>'
            f'<div style="font-family:\'EB Garamond\',serif;font-weight:500;font-size:1.15rem;'
            f'margin:0.3rem 0 0.6rem 0;color:{INK};">{html.escape(title)}</div>'
            f'<p style="font-size:0.88rem;color:{INK_2};line-height:1.6;">{body}</p>',
            unsafe_allow_html=True,
        )

# pLDDT explainer block — always visible
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="pl-card">
      <div class="pl-card-num">CONFIDENCE · pLDDT</div>
      <div class="pl-card-title">How to read the score</div>
      <p class="pl-card-body">
        <strong>pLDDT</strong> ("predicted Local Distance Difference Test") is a per-residue confidence
        score the model assigns to its own prediction, on a 0–100 scale. We average it for the headline
        number; the per-residue values are stored in the B-factor column of the PDB and shown as the
        heatmap above.
      </p>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:1rem;">
        <div><div style="width:20px;height:20px;background:{PLDDT_HI};border-radius:3px;margin-bottom:0.4rem;"></div>
          <strong style="color:{INK};">&gt; 90 · very high</strong>
          <p style="font-size:0.83rem;color:{INK_2};margin:0.2rem 0 0 0;">Backbone and side chains reliable.</p></div>
        <div><div style="width:20px;height:20px;background:{PLDDT_MID};border-radius:3px;margin-bottom:0.4rem;"></div>
          <strong style="color:{INK};">70–90 · confident</strong>
          <p style="font-size:0.83rem;color:{INK_2};margin:0.2rem 0 0 0;">Backbone reliable, side chains mostly right.</p></div>
        <div><div style="width:20px;height:20px;background:#c08a3a;border-radius:3px;margin-bottom:0.4rem;"></div>
          <strong style="color:{INK};">50–70 · low</strong>
          <p style="font-size:0.83rem;color:{INK_2};margin:0.2rem 0 0 0;">General topology may be right.</p></div>
        <div><div style="width:20px;height:20px;background:{PLDDT_LO};border-radius:3px;margin-bottom:0.4rem;"></div>
          <strong style="color:{INK};">&lt; 50 · very low</strong>
          <p style="font-size:0.83rem;color:{INK_2};margin:0.2rem 0 0 0;">Likely disordered or wrong.</p></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Footer — Sundai Club attribution
# ══════════════════════════════════════════════════════════════════════════════

_SUNDAI_LOGO = os.path.join(os.path.dirname(__file__), "assets", "sundai_club.png")
_sundai_uri = _img_data_uri(_SUNDAI_LOGO)
_logo_html = (
    f'<img src="{_sundai_uri}" alt="Sundai Club" style="height:42px;opacity:0.85;"/>'
    if _sundai_uri else '<div style="font-size:1.5rem;">🍦</div>'
)
st.markdown(
    f"""
    <div class="pl-footer">
      <a href="https://sundai.club" target="_blank" style="text-decoration:none;color:inherit;
         display:inline-flex;flex-direction:column;align-items:center;gap:0.4rem;">
        {_logo_html}
        <div style="font-size:0.8rem;color:{INK_3};">
          Developed at <strong style="color:{ACCENT};letter-spacing:0.04em;">SUNDAI CLUB</strong>
          · A community of builders shipping AI projects
        </div>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)
