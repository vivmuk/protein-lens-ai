"""Amino-acid sequence analytics for ProteinLens AI.

Pure-Python utilities that compute physicochemical stats, hydropathy profiles,
and functional motif hits for any peptide, with a renderer that feeds Venice AI.
"""

from __future__ import annotations

import re
from typing import Any

from Bio.SeqUtils.ProtParam import ProteinAnalysis

STANDARD_AAS: str = "ACDEFGHIKLMNPQRSTVWY"

KD_SCALE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

AA_CATEGORIES: dict[str, str] = {
    "hydrophobic": "AVILM",
    "polar_uncharged": "STNQ",
    "positive": "KRH",
    "negative": "DE",
    "aromatic": "FYW",
    "special": "CGP",
}

MOTIF_PATTERNS: list[tuple[str, str, str]] = [
    ("N-glycosylation site", r"N[^P][ST][^P]",
     "N-linked glycosylation consensus sequence"),
    ("Casein kinase II phosphorylation", r"[ST].{2}[DE]",
     "CK2 phosphorylation site"),
    ("Protein kinase C phosphorylation", r"[ST].[RK]",
     "PKC phosphorylation site"),
    ("cAMP/cGMP-dependent kinase", r"[RK][RK].[ST]",
     "PKA/PKG phosphorylation site"),
    ("Tyrosine kinase phosphorylation", r"[RK].{2,3}[DE].{2,3}Y",
     "Tyrosine kinase phosphorylation site"),
    ("N-myristoylation", r"^MG[^EDRKHPFYW].[STAGCN][^P]",
     "N-terminal myristoylation signal"),
    ("Leucine zipper", r"L.{6}L.{6}L.{6}L",
     "Heptad leucine repeat (coiled-coil dimerization)"),
    ("Zinc finger C2H2", r"C.{2,4}C.{12}H.{3,5}H",
     "C2H2-type zinc finger (DNA/RNA binding)"),
    ("RGD cell attachment", r"RGD",
     "Integrin-binding cell attachment motif"),
    ("Nuclear localization (poly-basic)", r"[KR]{4,}",
     "Classic monopartite NLS-like basic cluster"),
    ("Nuclear localization (PKKKR-like)", r"[PKR][PKR][KR][KR]",
     "Monopartite NLS pattern"),
    ("ER retention signal", r"KDEL$",
     "C-terminal ER lumen retention signal"),
]

MAX_MOTIF_HITS: int = 50


def _sanitize(sequence: str) -> str:
    return "".join(c for c in sequence.upper() if c in STANDARD_AAS)


def compute_basic_stats(sequence: str) -> dict[str, Any]:
    clean = _sanitize(sequence)
    analysis = ProteinAnalysis(clean)

    try:
        instability = float(analysis.instability_index())
    except (KeyError, ValueError):
        instability = float("nan")

    try:
        aromaticity = float(analysis.aromaticity())
    except (KeyError, ValueError):
        aromaticity = 0.0

    try:
        gravy = float(analysis.gravy())
    except (KeyError, ValueError):
        gravy = 0.0

    try:
        pi = float(analysis.isoelectric_point())
    except (KeyError, ValueError):
        pi = 7.0

    try:
        mw = float(analysis.molecular_weight())
    except (KeyError, ValueError):
        mw = 0.0

    try:
        ext_reduced, _ = analysis.molar_extinction_coefficient()
        ext_280 = int(ext_reduced)
    except (KeyError, ValueError, TypeError):
        ext_280 = 0

    try:
        net_charge = float(analysis.charge_at_pH(7.0))
    except (KeyError, ValueError, AttributeError):
        net_charge = 0.0

    length = len(clean)
    counts = {aa: clean.count(aa) for aa in STANDARD_AAS}
    aa_composition_pct = {
        aa: (counts[aa] / length * 100.0) if length else 0.0
        for aa in STANDARD_AAS
    }

    aa_category_pct: dict[str, float] = {}
    for cat, members in AA_CATEGORIES.items():
        cat_count = sum(counts[aa] for aa in members)
        aa_category_pct[cat] = (cat_count / length * 100.0) if length else 0.0

    return {
        "length": length,
        "molecular_weight": mw,
        "molecular_weight_kda": mw / 1000.0,
        "isoelectric_point": pi,
        "instability_index": instability,
        "aromaticity": aromaticity,
        "gravy": gravy,
        "extinction_coefficient_280nm": ext_280,
        "net_charge_at_ph7": net_charge,
        "aa_composition_pct": aa_composition_pct,
        "aa_category_pct": aa_category_pct,
    }


def kyte_doolittle_profile(sequence: str, window: int = 9) -> list[float]:
    """Sliding-window KD hydropathy. Edges are mirror-padded so the output
    length equals len(sequence); unknown residues contribute 0.0."""
    clean = _sanitize(sequence)
    n = len(clean)
    if n == 0:
        return []
    if window < 1:
        window = 1
    if window > n:
        window = n

    values = [KD_SCALE.get(aa, 0.0) for aa in clean]
    half = window // 2

    padded = (
        list(reversed(values[1:half + 1]))
        + values
        + list(reversed(values[max(0, n - half - 1):n - 1]))
    )
    while len(padded) < n + 2 * half:
        padded.append(0.0)

    profile: list[float] = []
    for i in range(n):
        chunk = padded[i:i + window]
        profile.append(sum(chunk) / window)
    return profile


def _signal_peptide_hint(sequence: str) -> dict[str, Any] | None:
    clean = _sanitize(sequence)
    head = clean[:30]
    if len(head) < 7:
        return None
    win = 7
    for i in range(0, len(head) - win + 1):
        chunk = head[i:i + win]
        gravy = sum(KD_SCALE.get(aa, 0.0) for aa in chunk) / win
        if gravy > 1.5:
            return {
                "name": "Signal peptide hint",
                "start": i + 1,
                "end": i + win,
                "match": chunk,
                "description": (
                    "Hydrophobic N-terminal stretch (gravy>1.5 over 7 AA) "
                    "consistent with a signal peptide or TM helix"
                ),
            }
    return None


def scan_motifs(sequence: str) -> list[dict[str, Any]]:
    clean = _sanitize(sequence)
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for name, pattern, description in MOTIF_PATTERNS:
        for m in re.finditer(pattern, clean):
            key = (name, m.start())
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "name": name,
                "start": m.start() + 1,
                "end": m.end(),
                "match": m.group(0),
                "description": description,
            })
            if len(hits) >= MAX_MOTIF_HITS:
                return hits

    sp = _signal_peptide_hint(clean)
    if sp is not None:
        key = (sp["name"], sp["start"] - 1)
        if key not in seen and len(hits) < MAX_MOTIF_HITS:
            hits.append(sp)

    return hits


def comparative_framing(stats: dict[str, Any]) -> str:
    length = int(stats.get("length", 0))
    gravy = float(stats.get("gravy", 0.0))
    net_charge = float(stats.get("net_charge_at_ph7", 0.0))
    aromaticity = float(stats.get("aromaticity", 0.0))

    if length < 30:
        size_word = "Short peptide"
    elif length < 100:
        size_word = "Mini-protein"
    elif length < 250:
        size_word = "Mid-size globular candidate"
    elif length < 500:
        size_word = "Large multi-domain protein"
    else:
        size_word = "Very large protein"

    if gravy > 0.8:
        character = "highly hydrophobic — resembles transmembrane or signal-peptide-like fragments"
    elif gravy > 0.2:
        character = "leaning hydrophobic — likely membrane-associated or buried-core rich"
    elif gravy < -0.8:
        character = "very polar/charged — typical of intrinsically disordered or solvent-exposed regions"
    elif abs(net_charge) > max(5.0, 0.05 * length):
        polarity = "basic" if net_charge > 0 else "acidic"
        character = f"strongly {polarity} — suggests nucleic-acid binding or charged surface interactions"
    elif aromaticity > 0.12:
        character = "aromatic-rich — possible ligand-binding pocket or stacking interface"
    else:
        character = "balanced charge and polarity — typical of small enzymes or single-domain regulators"

    return f"{size_word} (~{length} AA), {character}."


def format_for_prompt(
    stats: dict[str, Any],
    motifs: list[dict[str, Any]],
    framing: str,
) -> str:
    comp = stats.get("aa_composition_pct", {})
    top5 = sorted(comp.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top5_str = ", ".join(f"{aa} {pct:.1f}%" for aa, pct in top5 if pct > 0)

    cat = stats.get("aa_category_pct", {})
    cat_str = ", ".join(
        f"{k} {cat.get(k, 0.0):.1f}%"
        for k in ("hydrophobic", "polar_uncharged", "positive",
                  "negative", "aromatic", "special")
    )

    instab = stats.get("instability_index", float("nan"))
    instab_label = "stable" if isinstance(instab, (int, float)) and instab < 40 else "unstable"
    instab_str = (
        f"{instab:.1f} ({instab_label})"
        if isinstance(instab, (int, float)) and instab == instab
        else "n/a"
    )

    if motifs:
        shown = motifs[:15]
        motif_lines = "\n".join(
            f"- {m['name']} @ {m['start']}-{m['end']} `{m['match']}`"
            for m in shown
        )
        extra = f"\n- ...and {len(motifs) - len(shown)} more" if len(motifs) > len(shown) else ""
        motif_block = motif_lines + extra
    else:
        motif_block = "- none detected"

    return (
        "**Sequence analysis**\n"
        f"- Length & MW: {stats.get('length', 0)} AA, "
        f"{stats.get('molecular_weight_kda', 0.0):.2f} kDa "
        f"({stats.get('molecular_weight', 0.0):.1f} Da)\n"
        f"- Charge & pI: net charge {stats.get('net_charge_at_ph7', 0.0):+.2f} at pH 7, "
        f"pI {stats.get('isoelectric_point', 0.0):.2f}, "
        f"ε280 {stats.get('extinction_coefficient_280nm', 0)} M-1 cm-1\n"
        f"- Hydropathy: GRAVY {stats.get('gravy', 0.0):+.3f}, "
        f"hydrophobic residues {cat.get('hydrophobic', 0.0):.1f}%, "
        f"aromaticity {stats.get('aromaticity', 0.0):.3f}\n"
        f"- Top composition: {top5_str or 'n/a'}\n"
        f"- Category breakdown: {cat_str}\n"
        f"- Stability index: {instab_str}\n"
        f"- Motif hits ({len(motifs)}):\n{motif_block}\n"
        f"- Framing: {framing}"
    )
