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

      /* Hide Streamlit's auto-anchor link icons on headings */
      h1 > a, h2 > a, h3 > a, h4 > a, h5 > a,
      [data-testid="stHeaderActionElements"] {{ display: none !important; }}

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
        text-align:center; padding: 2rem 1rem 1rem 1rem;
        color: {INK_3}; font-size: 0.78rem;
        border-top: 1px solid {RULE}; margin-top: 3rem;
      }}

      /* ── Dashboard tiles ── */
      .pl-tile {{
        background: {PAPER_2}; border: 1px solid {RULE};
        border-radius: 12px; padding: 1.1rem 1.2rem;
        display:flex; flex-direction:column; gap: 0.35rem;
      }}
      .pl-tile-label {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        color: {INK_3}; letter-spacing: 0.08em; text-transform: uppercase;
      }}
      .pl-tile-value {{
        font-family: 'EB Garamond', serif; font-size: 1.9rem;
        color: {INK}; font-weight: 500; line-height: 1.1;
      }}
      .pl-tile-unit {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        color: {INK_3}; margin-left: 0.3rem;
      }}
      .pl-tile-bar {{
        margin-top: 0.4rem; height: 3px; background: {RULE};
        border-radius: 2px; overflow: hidden;
      }}
      .pl-tile-bar-fill {{ height: 100%; }}
      .pl-tile-hint {{
        font-size: 0.75rem; color: {INK_3}; line-height: 1.4;
      }}

      /* ── Donut chart (CSS conic-gradient) ── */
      .pl-donut-wrap {{
        display:flex; align-items:center; gap: 1.5rem;
      }}
      .pl-donut {{
        width: 160px; height: 160px; border-radius: 50%;
        position: relative; flex-shrink: 0;
      }}
      .pl-donut::after {{
        content: ""; position: absolute; inset: 30%;
        background: {PAPER_2}; border-radius: 50%;
      }}
      .pl-donut-center {{
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        font-family: 'EB Garamond', serif; font-size: 1.4rem; color: {INK};
        z-index: 1; text-align: center;
      }}
      .pl-donut-center span {{
        display:block; font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem; color: {INK_3}; letter-spacing: 0.05em;
      }}
      .pl-donut-legend {{ display:flex; flex-direction:column; gap: 0.4rem; flex: 1; }}
      .pl-donut-legend-row {{
        display:flex; align-items:center; gap: 0.6rem; font-size: 0.84rem;
      }}
      .pl-donut-swatch {{
        width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0;
      }}
      .pl-donut-label {{ flex: 1; color: {INK_2}; }}
      .pl-donut-pct {{
        font-family: 'JetBrains Mono', monospace; color: {INK};
        font-size: 0.82rem;
      }}

      /* ── Motif map ── */
      .pl-motif-track {{
        position: relative; height: 28px; background: {RULE}; border-radius: 4px;
        margin: 0.6rem 0;
      }}
      .pl-motif-marker {{
        position: absolute; top: 0; height: 100%;
        background: {ACCENT}; border-radius: 3px; opacity: 0.78;
        min-width: 4px;
      }}
      .pl-motif-axis {{
        display:flex; justify-content:space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: {INK_3};
      }}

      /* ── Info "?" toggle button ── */
      .pl-info-btn {{
        display:inline-flex; align-items:center; justify-content:center;
        width: 16px; height: 16px; border-radius: 50%;
        background: {RULE}; color: {INK_2};
        font-size: 0.7rem; font-weight: 600; cursor: help;
        margin-left: 0.4rem; user-select: none;
        font-family: 'JetBrains Mono', monospace;
      }}
      .pl-info-btn:hover {{ background: {INK}; color: {PAPER_2}; }}

      /* Tabs (main level, larger) */
      .pl-tabs-wrap .stTabs [data-baseweb="tab-list"] {{
        gap: 0; border-bottom: 1px solid {RULE};
      }}
      .pl-tabs-wrap .stTabs [data-baseweb="tab"] {{
        font-family: 'EB Garamond', serif; font-size: 1.15rem;
        padding: 0.7rem 1.4rem; color: {INK_3};
      }}
      .pl-tabs-wrap .stTabs [aria-selected="true"] {{
        color: {INK} !important; border-bottom: 2px solid {ACCENT} !important;
      }}

      /* Section heading inside dashboard */
      .pl-section-eyebrow {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        color: {INK_3}; letter-spacing: 0.1em; text-transform: uppercase;
        margin-top: 2rem; margin-bottom: 0.5rem;
      }}
      .pl-section-title {{
        font-family: 'EB Garamond', serif; font-size: 1.5rem;
        font-weight: 500; color: {INK}; margin: 0 0 1rem 0;
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

    # Larger fixed canvas + zoomTo() then center() so the molecule lands in the middle.
    # The components.html iframe matches the canvas width; we let it fill the card.
    view = py3Dmol.view(width=820, height=480)
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
    # Wrap the iframe in a flex container so it self-centers inside the card.
    raw_html = view._make_html()
    centered_html = (
        f'<div style="display:flex;justify-content:center;align-items:center;width:100%;">'
        f'{raw_html}</div>'
    )
    components.html(centered_html, height=500, scrolling=False)


def render_color_legend(color_scheme: str) -> str:
    """Return an HTML legend explaining the current viewer color scheme."""
    if color_scheme == "Spectrum (Rainbow)":
        gradient = (
            "linear-gradient(90deg, #2a4eff 0%, #00b6c7 25%, #36c93c 50%, "
            "#f3c83f 75%, #e93d3d 100%)"
        )
        return (
            f'<div style="margin-top:0.6rem;">'
            f'<div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:0.7rem;color:{INK_3};margin-bottom:4px;">'
            f'<span>N-terminus (start)</span><span>C-terminus (end)</span></div>'
            f'<div style="height:8px;border-radius:4px;background:{gradient};"></div>'
            f'<p style="font-size:0.78rem;color:{INK_3};margin:0.4rem 0 0 0;line-height:1.4;">'
            f"Color follows residue position — useful for seeing how the chain folds back on itself."
            f"</p></div>"
        )
    if color_scheme == "Confidence (pLDDT)":
        return (
            f'<div style="margin-top:0.6rem;display:flex;gap:0.8rem;flex-wrap:wrap;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:{INK_2};">'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_HI};'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>&gt;90 very high</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_MID};'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>70–90 confident</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:#c08a3a;'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>50–70 low</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_LO};'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>&lt;50 very low</span>'
            f"</div>"
            f'<p style="font-size:0.78rem;color:{INK_3};margin:0.4rem 0 0 0;line-height:1.4;">'
            f"Color = per-residue confidence. Blue/red strands suggest unreliable regions."
            f"</p>"
        )
    if color_scheme == "Secondary Structure":
        return (
            f'<div style="margin-top:0.6rem;display:flex;gap:1rem;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:{INK_2};">'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:#ff00ff;'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>α-helix</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:#ffff00;'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>β-sheet</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:#cccccc;'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>loop / coil</span>'
            f"</div>"
            f'<p style="font-size:0.78rem;color:{INK_3};margin:0.4rem 0 0 0;line-height:1.4;">'
            f"Jmol scheme — α-helices in magenta, β-strands in yellow, loops in gray."
            f"</p>"
        )
    if color_scheme == "Hydrophobicity":
        gradient = "linear-gradient(90deg, #2c5aa0 0%, #f5f5f5 50%, #b03030 100%)"
        return (
            f'<div style="margin-top:0.6rem;">'
            f'<div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:0.7rem;color:{INK_3};margin-bottom:4px;">'
            f'<span>polar (water-loving)</span><span>hydrophobic (water-fearing)</span></div>'
            f'<div style="height:8px;border-radius:4px;background:{gradient};"></div>'
            f'<p style="font-size:0.78rem;color:{INK_3};margin:0.4rem 0 0 0;line-height:1.4;">'
            f"Red residues cluster in the protein core; blue residues face the solvent."
            f"</p></div>"
        )
    # Monochrome — no legend needed
    return ""


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


# ── Dashboard rendering helpers ────────────────────────────────────────────────

# Stable palette for the 6 AA categories — chosen for distinguishability in print
CATEGORY_COLORS: dict[str, str] = {
    "hydrophobic":     "#5b8c5a",
    "polar_uncharged": "#c4a35a",
    "positive":        "#4a7ba6",
    "negative":        "#b85a5a",
    "aromatic":        "#7a5a8a",
    "special":         "#7d7d75",
}
CATEGORY_LABELS: dict[str, str] = {
    "hydrophobic":     "Hydrophobic (AVILM)",
    "polar_uncharged": "Polar uncharged (STNQ)",
    "positive":        "Positively charged (KRH)",
    "negative":        "Negatively charged (DE)",
    "aromatic":        "Aromatic (FYW)",
    "special":         "Special (C/G/P)",
}


def render_vital_tile(label: str, value: str, hint: str = "", bar_pct: Optional[float] = None,
                      bar_color: str = "#1a1a1a", unit: str = "") -> str:
    bar = ""
    if bar_pct is not None:
        pct = max(0, min(100, bar_pct))
        bar = (f'<div class="pl-tile-bar">'
               f'<div class="pl-tile-bar-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>'
               f'</div>')
    unit_html = f'<span class="pl-tile-unit">{html.escape(unit)}</span>' if unit else ""
    hint_html = f'<div class="pl-tile-hint">{html.escape(hint)}</div>' if hint else ""
    return (
        f'<div class="pl-tile">'
        f'<div class="pl-tile-label">{html.escape(label)}</div>'
        f'<div class="pl-tile-value">{value}{unit_html}</div>'
        f'{bar}{hint_html}'
        f'</div>'
    )


def render_donut(category_pct: dict[str, float]) -> str:
    items = [(k, category_pct.get(k, 0.0)) for k in CATEGORY_COLORS]
    items = [(k, v) for k, v in items if v > 0]
    if not items:
        return ""
    cum = 0.0
    stops = []
    for key, pct in items:
        start = cum * 3.6
        cum += pct
        end = cum * 3.6
        stops.append(f"{CATEGORY_COLORS[key]} {start:.1f}deg {end:.1f}deg")
    grad = ", ".join(stops)
    legend_rows = "".join(
        f'<div class="pl-donut-legend-row">'
        f'<span class="pl-donut-swatch" style="background:{CATEGORY_COLORS[k]};"></span>'
        f'<span class="pl-donut-label">{CATEGORY_LABELS[k]}</span>'
        f'<span class="pl-donut-pct">{v:.1f}%</span></div>'
        for k, v in items
    )
    return (
        f'<div class="pl-donut-wrap">'
        f'<div class="pl-donut" style="background:conic-gradient({grad});">'
        f'<div class="pl-donut-center">100%<span>composition</span></div>'
        f'</div>'
        f'<div class="pl-donut-legend">{legend_rows}</div>'
        f'</div>'
    )


def render_motif_map(motifs: list[dict], seq_len: int) -> str:
    if seq_len <= 0:
        return ""
    markers = []
    for m in motifs[:30]:
        left = (m["start"] / seq_len) * 100
        width = max(0.4, ((m["end"] - m["start"]) / seq_len) * 100)
        title = f"{m['name']} @ {m['start']+1}–{m['end']}  ({m['match']})"
        markers.append(
            f'<div class="pl-motif-marker" style="left:{left:.2f}%;width:{width:.2f}%;" '
            f'title="{html.escape(title)}"></div>'
        )
    mid = max(1, seq_len // 2)
    return (
        f'<div class="pl-motif-track">{"".join(markers)}</div>'
        f'<div class="pl-motif-axis">'
        f'<span>1</span><span>{mid}</span><span>{seq_len}</span>'
        f'</div>'
    )


def explainer(text: str) -> str:
    """Render a small '?' bubble whose tooltip shows the explanation."""
    safe = html.escape(text).replace("\n", "&#10;")
    return f'<span class="pl-info-btn" title="{safe}">i</span>'


# Reusable explainer texts — these surface when users hover the (i) badges
TERM_EXPLAIN = {
    "MW":
        "Molecular weight in kilodaltons (kDa). Computed from the sum of average residue masses.",
    "pI":
        "Isoelectric point — the pH at which the protein carries no net electrical charge. "
        "Affects solubility, migration in gels, and binding behavior.",
    "charge":
        "Net charge at pH 7, computed from the count of basic (K, R, H) minus acidic (D, E) residues "
        "weighted by their pKa contributions.",
    "gravy":
        "Grand Average of Hydropathy — mean Kyte-Doolittle score across all residues. "
        "Positive = overall hydrophobic, negative = overall polar.",
    "aromaticity":
        "Fraction of residues that are aromatic (F, W, Y). Drives UV absorbance at 280 nm and "
        "stacking interactions inside the fold.",
    "instability":
        "Predicted in-vitro stability (Guruprasad et al., 1990). < 40 suggests a stable protein, "
        "> 40 suggests one that may degrade quickly.",
    "epsilon":
        "Extinction coefficient at 280 nm, used to calculate protein concentration from UV "
        "absorbance. Predicted from tryptophan + tyrosine + cystine content.",
    "kyte_doolittle":
        "Each amino acid is scored on a hydropathy scale (Ile/Val/Leu most hydrophobic; Lys/Arg "
        "most polar). The sliding window plotted here highlights regions that prefer water (below 0) "
        "vs membrane / hydrophobic cores (above 0). Peaks >1.6 over 9+ residues often mark "
        "transmembrane helices or buried hydrophobic patches.",
    "motifs":
        "Short sequence patterns associated with known biology — phosphorylation sites, "
        "glycosylation sequons, localization signals, etc. A hit suggests possible function but is "
        "not proof — confirm experimentally.",
}


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
    ("folded_sequence", None),
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
        f'font-size:1.4rem;color:{INK};margin-bottom:1.25rem;">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{INK};'
        f'border-radius:50%;margin-right:0.5rem;vertical-align:middle;"></span>'
        f"proteinlens</div>",
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

_SUNDAI_LOGO = os.path.join(os.path.dirname(__file__), "assets", "sundai_club.png")
_sundai_uri = _img_data_uri(_SUNDAI_LOGO)
_sundai_nav_html = (
    f'<a href="https://sundai.club" target="_blank" '
    f'style="display:flex;align-items:center;gap:0.85rem;text-decoration:none;color:{INK_3};">'
    f'<img src="{_sundai_uri}" alt="Sundai Club" style="height:54px;opacity:0.95;"/>'
    f'<span style="font-size:0.82rem;font-family:\'JetBrains Mono\',monospace;line-height:1.3;">'
    f'developed at <strong style="color:{ACCENT};font-size:0.92rem;">SUNDAI CLUB</strong><br>'
    f'<span style="color:{INK_3};opacity:0.8;">5.24.2026</span></span></a>'
    if _sundai_uri else
    f'<a href="https://sundai.club" target="_blank" style="color:{INK_3};text-decoration:none;'
    f'font-size:0.9rem;">developed at <strong style="color:{ACCENT};">SUNDAI CLUB</strong> · 5.24.2026</a>'
)

st.markdown(
    f"""
    <div class="pl-nav">
      <div class="brand"><span class="dot"></span>proteinlens</div>
      <div style="margin-left:auto;">{_sundai_nav_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Hero row
# ══════════════════════════════════════════════════════════════════════════════

current_seq: str = (st.session_state["sequence"] or "").upper()
current_name: str = st.session_state["protein_name"] or "Unknown peptide"

# In Build mode, always reflect the live build_seq (even below the 10-AA fold threshold)
# so the user sees their typing in real time instead of the previously loaded preset.
if input_mode == "Build sequence":
    current_seq = st.session_state["build_seq"].upper()
    current_name = (protein_name_input or "Custom peptide") if current_seq else "—"

# Only show a fold when it corresponds to the sequence currently displayed.
# Prevents the viewer from showing a stale (e.g., default GLP-1) structure after
# the user switches input modes or starts building a new peptide.
has_fold: bool = (
    bool(st.session_state["pdb_string"])
    and st.session_state.get("folded_sequence") == current_seq
    and bool(current_seq)
)

hero_left, hero_right = st.columns([5, 6], gap="large")

with hero_left:
    eyebrow = (
        f"{current_name[:32]} · {len(current_seq)} AA"
        if current_seq else "no sequence loaded"
    )
    st.markdown(f'<div class="pl-eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pl-headline">One sequence.<br>'
        f'<span class="accent">One fold.</span><br>One story.</div>',
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
    st.caption("Use the sidebar to paste a sequence, search UniProt, or build one click-by-click.")

with hero_right:
    st.markdown('<div class="pl-card" style="padding:1.5rem;">', unsafe_allow_html=True)

    backend_label = {
        "simplefold": "SimpleFold 100M · local MLX",
        "modal": "SimpleFold 100M · Modal GPU",
        "esm": "ESMFold · API",
    }.get(FOLDING_BACKEND, FOLDING_BACKEND)

    # Pills row: AA count + model + (when folded) prominent color-coded pLDDT
    conf = st.session_state["confidence"] or 0.0
    conf_pct = conf * 100
    if has_fold:
        if conf >= 0.90:
            conf_color, conf_band = PLDDT_HI, "very high"
        elif conf >= 0.70:
            conf_color, conf_band = PLDDT_MID, "confident"
        elif conf >= 0.50:
            conf_color, conf_band = "#c08a3a", "low"
        else:
            conf_color, conf_band = PLDDT_LO, "very low"
        conf_pill = (
            f'<span class="pl-pill" style="background:{conf_color}15;border-color:{conf_color}50;'
            f'color:{conf_color};font-weight:600;">'
            f'pLDDT {conf_pct:.1f}% · {conf_band}</span>'
        )
    else:
        conf_pill = '<span class="pl-pill">pLDDT —</span>'

    pills_html = (
        f'{conf_pill}'
        f'<span class="pl-pill">{len(current_seq)} AA</span>'
        f'<span class="pl-pill">{html.escape(backend_label)}</span>'
        f'<span class="pl-pill">{html.escape(color_scheme.lower())}</span>'
    )
    st.markdown(f'<div class="pl-pill-row">{pills_html}</div>', unsafe_allow_html=True)

    if has_fold:
        render_protein_3d(st.session_state["pdb_string"], color_scheme=color_scheme)
        legend_html = render_color_legend(color_scheme)
        if legend_html:
            st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown(
            f'<div class="pl-plddt-bar" style="margin-top:0.8rem;">'
            f'<div class="pl-plddt-fill" '
            f'style="width:{min(conf_pct,100):.0f}%;background:{conf_color};"></div></div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "↓  Download PDB",
            data=st.session_state["pdb_string"],
            file_name=f"{current_name.replace(' ', '_')}.pdb",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        if input_mode == "Build sequence" and current_seq and len(current_seq) < 10:
            need = 10 - len(current_seq)
            empty_msg = (
                f"<strong style=\"color:{INK};\">{len(current_seq)} / 10</strong> "
                f"amino acids — add <strong style=\"color:{INK};\">{need}</strong> more, "
                f"then click <strong style=\"color:{INK};font-style:normal;\">⚡ Fold & analyze</strong>"
            )
        elif not current_seq:
            empty_msg = (
                f'load a sequence in the sidebar, then click '
                f'<strong style="color:{INK};font-style:normal;">⚡ Fold & analyze</strong>'
            )
        else:
            empty_msg = (
                f'click <strong style="color:{INK};font-style:normal;font-family:Inter,sans-serif;'
                f'font-weight:500;">⚡ Fold & analyze</strong> in the sidebar'
            )
        st.markdown(
            f'<div style="height:380px;display:flex;align-items:center;justify-content:center;'
            f'text-align:center;color:{INK_3};font-family:\'EB Garamond\',serif;font-style:italic;'
            f'font-size:1.15rem;padding:0 1.5rem;">'
            f'{empty_msg}</div>',
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
    st.session_state["folded_sequence"] = clean
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
# Tabbed dashboard (Overview / Scientific / Medical / How it works)
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

st.markdown('<div class="pl-tabs-wrap">', unsafe_allow_html=True)
tab_overview, tab_sci, tab_msl, tab_how = st.tabs(
    ["Overview", "Scientific", "Medical Affairs", "How it works"]
)
st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 1: OVERVIEW (the dashboard) ───────────────────────────────────────────
with tab_overview:

    # At-a-glance vitals grid
    st.markdown('<div class="pl-section-eyebrow">AT A GLANCE</div>'
                '<div class="pl-section-title">Sequence vitals</div>',
                unsafe_allow_html=True)

    if stats:
        tile_specs = [
            ("Length",
             f"{stats['length']}",
             "AA",
             None, INK,
             "Number of amino acid residues."),
            ("Molecular weight",
             f"{stats['molecular_weight_kda']:.2f}",
             "kDa",
             min(100, stats['molecular_weight_kda'] / 1.5),
             INK,
             "Sum of residue masses + H2O."),
            ("Isoelectric pt (pI)",
             f"{stats['isoelectric_point']:.2f}",
             "",
             (stats['isoelectric_point'] / 14) * 100,
             INK,
             "pH where net charge = 0."),
            ("Net charge @ pH 7",
             f"{stats['net_charge_at_ph7']:+.1f}",
             "",
             None,
             "#4a7ba6" if stats['net_charge_at_ph7'] > 0 else "#b85a5a",
             "Sum of titratable side-chain charges at physiological pH."),
            ("Hydropathy (GRAVY)",
             f"{stats['gravy']:+.2f}",
             "",
             None,
             "#b85a5a" if stats['gravy'] > 0 else "#4a7ba6",
             "Mean Kyte-Doolittle hydropathy. > 0 hydrophobic, < 0 polar."),
            ("Aromaticity",
             f"{stats['aromaticity'] * 100:.1f}",
             "%",
             min(100, stats['aromaticity'] * 100 * 4),
             "#7a5a8a",
             "Fraction F + W + Y residues."),
            ("Stability index",
             f"{stats['instability_index']:.1f}",
             "",
             min(100, stats['instability_index']),
             PLDDT_HI if stats['instability_index'] < 40 else PLDDT_LO,
             "< 40 stable · > 40 unstable in vitro."),
            ("ε280 (reduced)",
             f"{stats['extinction_coefficient_280nm']:,}",
             "M⁻¹·cm⁻¹",
             None, INK,
             "UV absorbance coefficient for concentration."),
        ]
        # 4-column grid
        for row in (tile_specs[:4], tile_specs[4:]):
            cols = st.columns(4, gap="medium")
            for col, (label, val, unit, bar, color, hint) in zip(cols, row):
                with col:
                    st.markdown(
                        render_vital_tile(label, val, hint=hint, bar_pct=bar,
                                          bar_color=color, unit=unit),
                        unsafe_allow_html=True,
                    )

        st.markdown(
            f'<p style="margin-top:1.2rem;font-size:0.92rem;color:{INK_2};line-height:1.6;'
            f'font-style:italic;padding:0 0.25rem;">{html.escape(framing)}</p>',
            unsafe_allow_html=True,
        )

        # ── Composition ──
        st.markdown('<div class="pl-section-eyebrow">COMPOSITION</div>'
                    '<div class="pl-section-title">Amino acid breakdown</div>',
                    unsafe_allow_html=True)

        c_left, c_right = st.columns([1, 1], gap="large")
        with c_left:
            st.markdown(
                f'<div class="pl-card">{render_donut(stats["aa_category_pct"])}'
                f'<p style="margin-top:1rem;font-size:0.82rem;color:{INK_3};line-height:1.5;">'
                f"Categories use the Lehninger groupings: hydrophobic side chains drive folding cores; "
                f"polar residues prefer the surface; charged residues mediate binding and salt bridges; "
                f"aromatic residues stack and absorb UV."
                f"</p></div>",
                unsafe_allow_html=True,
            )
        with c_right:
            st.markdown(
                f'<div class="pl-card">'
                f'<div class="pl-card-num">PER-RESIDUE FREQUENCY (TOP 10)</div>'
                f'<div style="margin-top:0.8rem;">'
                f'{render_composition_bars(stats["aa_composition_pct"], top_n=10)}'
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # ── Hydropathy profile ──
        st.markdown(
            f'<div class="pl-section-eyebrow">HYDROPATHY PROFILE  '
            f'{explainer(TERM_EXPLAIN["kyte_doolittle"])}'
            f"</div>"
            f'<div class="pl-section-title">Kyte-Doolittle, 9-residue window</div>',
            unsafe_allow_html=True,
        )
        if hydro_profile:
            st.line_chart({"hydropathy": hydro_profile},
                          height=180, use_container_width=True)
            st.caption(
                f"Above 0 → hydrophobic (likely buried or membrane-embedded). "
                f"Below 0 → polar (likely solvent-exposed). "
                f"Hover the (i) badge for a deeper explanation."
            )

        # ── Motif map ──
        st.markdown(
            f'<div class="pl-section-eyebrow">MOTIF MAP  '
            f'{explainer(TERM_EXPLAIN["motifs"])}'
            f"</div>"
            f'<div class="pl-section-title">Known sequence patterns ({len(motifs)} hit{"s" if len(motifs) != 1 else ""})</div>',
            unsafe_allow_html=True,
        )
        if motifs:
            st.markdown(render_motif_map(motifs, len(current_seq)), unsafe_allow_html=True)
            motif_rows_html = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0;'
                f'border-bottom:1px dashed {RULE};font-size:0.86rem;">'
                f'<span style="color:{INK_2};">{html.escape(m["name"])}</span>'
                f'<span style="color:{INK};font-family:\'JetBrains Mono\',monospace;">'
                f'{m["start"]+1}–{m["end"]} · {html.escape(m["match"][:20])}</span></div>'
                for m in motifs[:12]
            )
            st.markdown(motif_rows_html, unsafe_allow_html=True)
            if len(motifs) > 12:
                st.caption(f"+{len(motifs) - 12} more — full list available in the AI explanation.")
        else:
            st.caption(
                "No canonical motifs matched. This is common for short or de-novo sequences; "
                "absence of a hit does not imply absence of function."
            )

        # ── Per-residue pLDDT (only if fold is current) ──
        if has_fold:
            st.markdown(
                f'<div class="pl-section-eyebrow">PER-RESIDUE CONFIDENCE</div>'
                f'<div class="pl-section-title">pLDDT heatmap (residue-by-residue)</div>',
                unsafe_allow_html=True,
            )
            plddt = extract_per_residue_plddt(st.session_state["pdb_string"])
            st.markdown(render_plddt_heatmap(plddt), unsafe_allow_html=True)
            legend = (
                f'<div style="display:flex;gap:1.2rem;margin-top:0.4rem;font-size:0.74rem;'
                f'color:{INK_3};font-family:\'JetBrains Mono\',monospace;">'
                f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_HI};'
                f'border-radius:2px;margin-right:4px;"></span>&gt;90 very high</span>'
                f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_MID};'
                f'border-radius:2px;margin-right:4px;"></span>70–90 confident</span>'
                f'<span><span style="display:inline-block;width:10px;height:10px;background:#c08a3a;'
                f'border-radius:2px;margin-right:4px;"></span>50–70 low</span>'
                f'<span><span style="display:inline-block;width:10px;height:10px;background:{PLDDT_LO};'
                f'border-radius:2px;margin-right:4px;"></span>&lt;50 very low</span>'
                f"</div>"
            )
            st.markdown(legend, unsafe_allow_html=True)

        # ── Raw sequence (collapsed) ──
        with st.expander("Show full sequence"):
            st.markdown(
                f'<div class="pl-seq" style="max-height:280px;">{html.escape(current_seq)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{len(current_seq)} amino acids · single-letter IUPAC")
    else:
        st.info(
            "Load a sequence in the sidebar (Quick examples, Paste, UniProt search, or "
            "click-to-build) — the dashboard populates immediately. Click **⚡ Fold & analyze** "
            "for the 3D structure, confidence map, and AI explanations."
        )


# ── Tab 2: SCIENTIFIC (full AI explanation) ───────────────────────────────────
with tab_sci:
    st.markdown(
        f'<div class="pl-section-eyebrow">AI SCIENTIFIC EXPLANATION</div>'
        f'<div class="pl-section-title">{html.escape(current_name)}</div>',
        unsafe_allow_html=True,
    )
    if has_fold and st.session_state["explanation"]:
        st.markdown(
            f'<div class="pl-explain" style="background:{PAPER_2};border:1px solid {RULE};'
            f'border-radius:12px;padding:1.5rem 1.75rem;">'
            f'{st.session_state["explanation"]}</div>',
            unsafe_allow_html=True,
        )
    elif framing:
        st.markdown(
            f'<div class="pl-card">'
            f'<p style="font-style:italic;color:{INK_2};margin:0;">'
            f"{html.escape(framing)}</p><br>"
            f'<p style="color:{INK_3};margin:0;">'
            f"Click <strong>⚡ Fold & analyze</strong> in the sidebar to run the structural "
            f"prediction and generate the full AI scientific analysis."
            f"</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Load a sequence to begin.")


# ── Tab 3: MEDICAL AFFAIRS / MSL ──────────────────────────────────────────────
with tab_msl:
    st.markdown(
        f'<div class="pl-section-eyebrow">MEDICAL AFFAIRS / MSL BRIEFING</div>'
        f'<div class="pl-section-title">{html.escape(current_name)}</div>',
        unsafe_allow_html=True,
    )
    if has_fold and st.session_state["msl_summary"]:
        st.markdown(
            f'<div class="pl-explain" style="background:{PAPER_2};border:1px solid {RULE};'
            f'border-radius:12px;padding:1.5rem 1.75rem;">'
            f'{st.session_state["msl_summary"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "An MSL-ready briefing (executive summary, talking points, HCP Q&A, mechanism "
            "framing) is generated after folding."
        )


# ── Tab 4: HOW IT WORKS ───────────────────────────────────────────────────────
with tab_how:
    st.markdown(
        f'<div class="pl-section-eyebrow">PIPELINE</div>'
        f'<div class="pl-section-title">From letters to a 3D protein</div>',
        unsafe_allow_html=True,
    )
    how_steps = [
        ("01", "Sequence in",
         "Your amino acid string (FASTA, pasted, fetched from UniProt, or built letter-by-letter) "
         "is validated and length-checked (min 10, max 1400 AA). Before any model runs, we compute "
         "physicochemical character — MW, pI, charge, GRAVY hydropathy, Kyte-Doolittle profile, "
         "and motif scan — so even unfolded peptides have an evidence-grounded analysis."),
        ("02", "Language-model embedding",
         "ESM-2 (Meta AI, 3B params) reads the sequence and emits a per-residue embedding that "
         "captures evolutionary context inferred from ~65M natural sequences. This embedding is "
         "the substrate every modern protein folder builds on."),
        ("03", "Flow-matching diffusion",
         "SimpleFold (Apple, 2025) starts from random 3D coordinates and iteratively denoises them, "
         "conditioned on the ESM embedding, over N steps (default 200). Unlike AlphaFold's iterative "
         "MSA refinement, SimpleFold's flow-matching diffusion is faster per step and runs comfortably "
         "on a single GPU. Output: per-atom coordinates in PDB format."),
        ("04", "Confidence + explanation",
         "A pLDDT confidence head scores each residue's local accuracy (0–100, stored in the "
         "B-factor column of the PDB). Venice AI then composes a domain-appropriate explanation — "
         "Scientific for researchers, MSL for medical-affairs — using your computed sequence stats "
         "as authoritative ground truth (the LLM cannot fabricate MW, pI, or motif hits)."),
    ]
    how_cols = st.columns(4, gap="large")
    for col, (num, title, body) in zip(how_cols, how_steps):
        with col:
            st.markdown(
                f'<div class="pl-card-num">{num}</div>'
                f'<div style="font-family:\'EB Garamond\',serif;font-weight:500;font-size:1.15rem;'
                f'margin:0.3rem 0 0.6rem 0;color:{INK};">{html.escape(title)}</div>'
                f'<p style="font-size:0.87rem;color:{INK_2};line-height:1.6;">{body}</p>',
                unsafe_allow_html=True,
            )

    # pLDDT explainer
    st.markdown(
        f"""
        <div class="pl-section-eyebrow" style="margin-top:2.5rem;">CONFIDENCE · pLDDT</div>
        <div class="pl-section-title">How to read the score</div>
        <div class="pl-card">
          <p class="pl-card-body" style="margin-bottom:1rem;">
            <strong>pLDDT</strong> ("predicted Local Distance Difference Test") is a per-residue
            confidence score the model assigns to its own prediction, on a 0–100 scale. We average
            it for the headline number; the per-residue values are stored in the B-factor column
            of the PDB and shown as the heatmap in the Overview tab.
          </p>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
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

    # Backends table
    st.markdown(
        f"""
        <div class="pl-section-eyebrow" style="margin-top:2.5rem;">INFRASTRUCTURE</div>
        <div class="pl-section-title">Folding backends</div>
        <div class="pl-card">
          <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
            <thead>
              <tr style="text-align:left;border-bottom:1px solid {RULE};color:{INK_3};
                         font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                         letter-spacing:0.06em;text-transform:uppercase;">
                <th style="padding:0.6rem 0.4rem;">Backend</th>
                <th style="padding:0.6rem 0.4rem;">When used</th>
                <th style="padding:0.6rem 0.4rem;">Cold start</th>
                <th style="padding:0.6rem 0.4rem;">Warm fold</th>
              </tr>
            </thead>
            <tbody style="color:{INK_2};">
              <tr style="border-bottom:1px dashed {RULE};">
                <td style="padding:0.65rem 0.4rem;color:{INK};">SimpleFold 100M (local MLX)</td>
                <td style="padding:0.65rem 0.4rem;">macOS Apple Silicon with the CLI installed</td>
                <td style="padding:0.65rem 0.4rem;">~10 s</td>
                <td style="padding:0.65rem 0.4rem;">30–90 s</td>
              </tr>
              <tr style="border-bottom:1px dashed {RULE};">
                <td style="padding:0.65rem 0.4rem;color:{INK};">SimpleFold 100M (Modal GPU)</td>
                <td style="padding:0.65rem 0.4rem;">Cloud deployments, deferred to Modal A10G</td>
                <td style="padding:0.65rem 0.4rem;">~70 s (cache hit)</td>
                <td style="padding:0.65rem 0.4rem;">~3 s</td>
              </tr>
              <tr>
                <td style="padding:0.65rem 0.4rem;color:{INK};">ESMFold (public API)</td>
                <td style="padding:0.65rem 0.4rem;">Zero-setup fallback, always available</td>
                <td style="padding:0.65rem 0.4rem;">—</td>
                <td style="padding:0.65rem 0.4rem;">15–30 s</td>
              </tr>
            </tbody>
          </table>
          <p style="margin-top:1rem;font-size:0.85rem;color:{INK_3};line-height:1.5;">
            The Modal backend caches the 6 GB SimpleFold checkpoint plus the boltz auxiliary
            (CCD + pLDDT side-car) on a persistent volume, so only the first-ever container pays
            the download cost. Subsequent containers cold-start in ~70 s; warm containers within
            the 5-minute scaledown window serve folds in seconds.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tech stack
    st.markdown(
        f"""
        <div class="pl-section-eyebrow" style="margin-top:2.5rem;">TECH STACK</div>
        <div class="pl-section-title">Built with</div>
        <div class="pl-card">
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem 2rem;">
            <div><strong style="color:{INK};">SimpleFold</strong>
              <p style="font-size:0.85rem;color:{INK_2};margin:0.2rem 0;">
                Apple, 2025 — flow-matching protein folder, 100 M params.</p></div>
            <div><strong style="color:{INK};">ESM-2</strong>
              <p style="font-size:0.85rem;color:{INK_2};margin:0.2rem 0;">
                Meta AI — 3 B-param protein language model for embeddings.</p></div>
            <div><strong style="color:{INK};">Venice AI</strong>
              <p style="font-size:0.85rem;color:{INK_2};margin:0.2rem 0;">
                Uncensored chat model that composes scientific + MSL explanations.</p></div>
            <div><strong style="color:{INK};">Modal</strong>
              <p style="font-size:0.85rem;color:{INK_2};margin:0.2rem 0;">
                Serverless GPU runtime; A10G containers with persistent volume cache.</p></div>
            <div><strong style="color:{INK};">py3Dmol</strong>
              <p style="font-size:0.85rem;color:{INK_2};margin:0.2rem 0;">
                WebGL molecular viewer rendered via Streamlit components.</p></div>
            <div><strong style="color:{INK};">Biopython · ProtParam</strong>
              <p style="font-size:0.85rem;color:{INK_2};margin:0.2rem 0;">
                MW, pI, ε280, instability, aromaticity. Industry reference.</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="pl-footer">
      proteinlens · built with SimpleFold (Apple) · ESM-2 (Meta) · Venice AI · Modal · py3Dmol<br>
      <span style="opacity:0.75;">attribution in the top-right nav · sundai.club</span>
    </div>
    """,
    unsafe_allow_html=True,
)
