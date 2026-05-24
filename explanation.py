"""
explanation.py — Venice AI-powered scientific and MSL explanation engine.

Uses the Venice AI API (OpenAI-compatible) to generate:
  - Scientific explanations for structural biologists / researchers
  - Medical Affairs / MSL summaries for HCP communication

Public API:
  - get_scientific_explanation(sequence, protein_name, confidence, insights_block)
  - get_msl_summary(sequence, protein_name, confidence, insights_block)
  - get_sequence_annotation(sequence)
"""

from __future__ import annotations

import os
from typing import Optional

import requests

# ── Venice API config ──────────────────────────────────────────────────────────
VENICE_API_KEY = os.environ.get(
    "VENICE_API_KEY",
    "VENICE_INFERENCE_KEY_KntgMduZd8nN5cxhGd99gD3a5HpqD142saxKSn0uOs",
)
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL = "venice-uncensored"


_UNKNOWN_NAMES = {
    "",
    "unknown",
    "unknown protein",
    "unknown peptide",
    "n/a",
    "untitled",
}


def _venice_chat(prompt: str, system: str = "", temperature: float = 0.6) -> str:
    """
    Send a chat completion request to Venice AI.

    Args:
        prompt:      User message / scientific query.
        system:      Optional system prompt to set expert persona.
        temperature: Creativity level (lower = more precise, higher = more varied).

    Returns:
        Response text string.

    Raises:
        RuntimeError: If the API call fails.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{VENICE_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {VENICE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": VENICE_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1600,
        },
        timeout=90,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Venice API error {resp.status_code}: {resp.text[:300]}"
        )

    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _is_unknown(protein_name: str) -> bool:
    return (protein_name or "").strip().lower() in _UNKNOWN_NAMES


def _confidence_band(conf: float) -> str:
    if conf > 0.90:
        return "very high"
    if conf >= 0.70:
        return "confident"
    if conf >= 0.50:
        return "low"
    return "very low"


def _confidence_context_block(confidence: Optional[float]) -> str:
    """Render a prompt-side context block describing the pLDDT value, or empty."""
    if confidence is None:
        return ""
    pct = confidence * 100.0
    band = _confidence_band(confidence)
    return (
        "\n**Model confidence (pLDDT):**\n"
        f"- Average pLDDT (0..1): {confidence:.3f}  (i.e. {pct:.1f}%)\n"
        f"- Band: {band}\n"
        "- Band definitions: >0.90 very high, 0.70–0.90 confident, "
        "0.50–0.70 low, <0.50 very low.\n"
    )


def _insights_context_block(insights_block: Optional[str]) -> str:
    """Wrap the precomputed insights as authoritative ground-truth, or fallback."""
    if insights_block and insights_block.strip():
        return (
            "\n**Sequence Insights (AUTHORITATIVE — use these exact numbers; "
            "do not invent or recompute):**\n"
            f"{insights_block.strip()}\n"
        )
    return (
        "\n**Sequence Insights:** (none precomputed — base any statistics on the "
        "raw length only; do NOT invent MW/pI/charge values. Prefer to note that "
        "richer stats from aa_insights.format_for_prompt() were not supplied.)\n"
    )


_PLDDT_SECTION_INSTRUCTION_SCI = (
    "## 📏 Model Confidence (pLDDT)\n"
    "Explain pLDDT plainly: it is the model's per-residue confidence (0..1) averaged "
    "across the structure. State the band thresholds (>0.90 very high, 0.70–0.90 "
    "confident, 0.50–0.70 low, <0.50 very low). Then interpret the SPECIFIC value "
    "given above: what it implies for how much to trust the backbone, side-chain "
    "rotamers, and any structural claims made earlier in this response."
)

_PLDDT_SECTION_INSTRUCTION_MSL = (
    "## 📏 How To Discuss Model Confidence\n"
    "Explain pLDDT in HCP-friendly language: it is the model's own confidence in "
    "each residue's position (0..1), averaged here. Give the bands (>0.90 very high, "
    "0.70–0.90 confident, 0.50–0.70 low, <0.50 very low). Then say what THIS value "
    "means for how strongly an MSL can speak about the structural picture."
)

_FABRICATION_GUARD = (
    "\n**Rules you MUST follow:**\n"
    "- Do not fabricate numerical values — use ONLY the stats provided in the "
    "Sequence Insights block above.\n"
    "- If a section has no relevant data, write "
    "'Insufficient evidence to answer from sequence alone' "
    "rather than speculating.\n"
    "- Prefer 'consistent with' over 'is' when discussing predicted families "
    "or functions for uncharacterized sequences.\n"
)


# ══════════════════════════════════════════════════════════════════════════════
# Scientific explanation
# ══════════════════════════════════════════════════════════════════════════════

SCIENTIFIC_SYSTEM = """You are a world-class structural biologist and biochemist with expertise in:
- Protein structure-function relationships
- Disease biology and drug targets
- Mechanistic biochemistry
- Structural genomics
- Interpretation of AlphaFold-style pLDDT confidence scores

Provide clear, accurate, and insightful scientific explanations. Use markdown formatting
(headers, bold, bullet points) to structure your response. Be precise but accessible
to a PhD-level researcher. Never speculate beyond scientific consensus.

When a pLDDT confidence score is provided, you MUST explain what it means and how it
gates the strength of any structural claims you make. When a Sequence Insights block
is provided, treat its numbers as authoritative ground-truth and quote/cite them
rather than inventing your own statistics."""


SCIENTIFIC_UNKNOWN_SYSTEM = """You are a world-class structural biologist analyzing an
UNCHARACTERIZED peptide or protein sequence. You have no canonical name to anchor on,
so you must reason from first principles using the provided sequence statistics, motif
hits, and fold-model confidence.

Be rigorous, comparative, and explicitly hedged. Use phrases like 'consistent with',
'resembles', 'compatible with' rather than 'is'. Never invent statistics — use only
the numbers given in the Sequence Insights block. When a pLDDT value is provided,
explain it and use it to gate how strongly you commit to any structural claim."""


def get_scientific_explanation(
    sequence: str,
    protein_name: str = "",
    confidence: Optional[float] = None,
    insights_block: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive scientific explanation for the folded protein.

    Args:
        sequence:       Amino acid sequence.
        protein_name:   Optional human-readable protein name. Unknown values
                        ('', 'Unknown', 'Unknown protein', ...) trigger the
                        de-novo / uncharacterized-peptide prompt shape.
        confidence:     Optional average pLDDT in [0, 1]. When given, the output
                        includes an explicit pLDDT explainer section.
        insights_block: Optional markdown block from aa_insights.format_for_prompt()
                        with MW, pI, charge, hydropathy, composition, motifs.
                        When given, it is treated as authoritative.

    Returns:
        Markdown-formatted scientific explanation.
    """
    seq_preview = sequence[:60] + ("..." if len(sequence) > 60 else "")
    conf_ctx = _confidence_context_block(confidence)
    insights_ctx = _insights_context_block(insights_block)
    unknown = _is_unknown(protein_name)

    if unknown:
        prompt = f"""A user has predicted the 3D structure of an UNCHARACTERIZED peptide / protein. There is no recognized name to anchor on, so reason from the sequence and the stats below.

**Sequence:**
- Length: {len(sequence)} amino acids
- First 60 residues: `{seq_preview}`
{conf_ctx}{insights_ctx}
Produce a structured analysis with EXACTLY these sections, in this order:

## 🧬 Sequence Character
Interpret the amino-acid statistics in the Sequence Insights block (size, hydrophobicity, charge, pI). What kind of molecule is this likely to be (small bioactive peptide, soluble globular domain, membrane-associated, intrinsically disordered, etc.)?

## 🔍 Motif & Pattern Analysis
Discuss any motif hits listed in the Sequence Insights block and what they suggest functionally. If no motifs are listed, say so explicitly.

## 🏗️ Predicted Structural Interpretation
Given the fold confidence and sequence length, what is the most likely topology — helical bundle, beta-rich, mixed α/β, intrinsically disordered, single amphipathic helix, β-hairpin, disulfide-stabilized scaffold? Justify from the stats.

## 🧭 Comparative Framing
What known peptide / protein families does this resemble (e.g. defensins, neuropeptides, signal peptides, hormone-like peptides, cytokine folds, small protease inhibitors) based on size and composition? Be careful not to overclaim — say "consistent with" not "is".

## 🎯 Possible Function Hypotheses
Give 2–3 plausible functional hypotheses, each with reasoning rooted in the provided stats (charge → membrane interaction, cysteine pairs → disulfide scaffold, etc.).

## 🧪 Suggested Wet-Lab Validation
Concrete next experiments: CD spectroscopy for secondary structure, mass spec for PTMs / disulfides, alanine scan for key residues, antimicrobial / receptor binding assays as appropriate, NMR for small peptides.

{_PLDDT_SECTION_INSTRUCTION_SCI}
Tie the pLDDT band back to how strongly the structural interpretation above can be trusted.
{_FABRICATION_GUARD}"""
        system = SCIENTIFIC_UNKNOWN_SYSTEM
    else:
        prompt = f"""A user has predicted the 3D structure of **{protein_name}**.

**Protein details:**
- Name: {protein_name}
- Sequence length: {len(sequence)} amino acids
- First 60 residues: `{seq_preview}`
{conf_ctx}{insights_ctx}
Provide a comprehensive scientific analysis with EXACTLY these sections, in this order:

## 🔬 Biological Function
Known or predicted function. What cellular processes does it participate in?

## 🧬 Structural Features
Expected secondary / tertiary elements (alpha helices, beta sheets, domains, motifs), tied to the known fold of {protein_name}.

## 🏥 Disease Relevance
Diseases, conditions, or phenotypes associated with this protein or its dysfunction. Include notable mutations.

## ⚙️ Mechanism of Action
Molecular-level mechanism: key interactions, catalytic steps, signaling role.

## 💊 Therapeutic Relevance
Is this a drug target? Approved therapies, clinical candidates, or active programs?

## 🔭 Scientific Context
Notable research directions, recent discoveries, or open questions.

{_PLDDT_SECTION_INSTRUCTION_SCI}

## 🧬 Sequence Snapshot
2–3 bullets summarizing the Sequence Insights block above (MW, pI/charge, hydropathy, any motif hits). If no insights block was provided, write 'Insufficient evidence to answer from sequence alone'.
{_FABRICATION_GUARD}"""
        system = SCIENTIFIC_SYSTEM

    try:
        return _venice_chat(prompt, system=system, temperature=0.5)
    except Exception as e:
        return (
            f"⚠️ Could not generate explanation: {str(e)}\n\n"
            "Check your Venice API key and internet connection."
        )


# ══════════════════════════════════════════════════════════════════════════════
# MSL / Medical Affairs summary
# ══════════════════════════════════════════════════════════════════════════════

MSL_SYSTEM = """You are a senior Medical Science Liaison (MSL) with deep expertise in:
- Translational medicine and mechanism of action communication
- HCP (healthcare professional) scientific engagement
- Medical Affairs communication strategy
- Clinical data interpretation and scientific storytelling
- Communicating structural-model confidence (pLDDT) without overclaiming

Your role is to translate complex structural biology into clear, clinically relevant,
and HCP-appropriate scientific narratives. Be precise, evidence-based, and
strategically framed for medical-scientific discussions. When given a Sequence
Insights block, treat it as ground-truth and cite its numbers rather than inventing
your own."""


MSL_UNKNOWN_SYSTEM = """You are a senior Medical Science Liaison preparing internal
notes about an UNCHARACTERIZED peptide / protein sequence. There is no approved
indication, no clinical data, and no canonical biology to lean on.

Your job is to give an MSL an honest, well-hedged framing they could use if asked
about this sequence by an HCP — without making clinical or therapeutic claims. Be
explicit about what is unknown. Use only the stats provided; do not invent numbers.
When a pLDDT score is given, explain it and use it to bound the strength of any
structural language."""


def get_msl_summary(
    sequence: str,
    protein_name: str = "",
    confidence: Optional[float] = None,
    insights_block: Optional[str] = None,
) -> str:
    """
    Generate a Medical Affairs / MSL-focused summary for the protein.

    Args:
        sequence:       Amino acid sequence.
        protein_name:   Optional human-readable protein name. Unknown values
                        trigger the uncharacterized-peptide MSL framing.
        confidence:     Optional average pLDDT in [0, 1]. When given, output
                        includes an HCP-friendly pLDDT explainer.
        insights_block: Optional markdown block from aa_insights.format_for_prompt().
                        Treated as authoritative when present.

    Returns:
        Markdown-formatted MSL talking points and summary.
    """
    conf_ctx = _confidence_context_block(confidence)
    insights_ctx = _insights_context_block(insights_block)
    unknown = _is_unknown(protein_name)

    if unknown:
        prompt = f"""You are preparing internal MSL notes about an UNCHARACTERIZED peptide / protein sequence. There is no approved name, no indication, and no clinical data — you must frame the conversation honestly around what the sequence and the fold model can and cannot tell us.

**Sequence details:**
- Length: {len(sequence)} amino acids
{conf_ctx}{insights_ctx}
Produce an MSL-ready briefing with EXACTLY these sections, in this order:

## 🎯 What We Know From The Sequence
Frame what the Sequence Insights block tells us (size class, charge character, hydropathy, any motif hits) in HCP-appropriate language. Avoid jargon where a plainer word works.

## 🧭 Likely Biological Class
Comparative framing rooted in the motifs and composition: which broad peptide / protein families does this most resemble (e.g. antimicrobial peptide-like, neuropeptide-like, signal-peptide-like, small disulfide-rich scaffold)? Use 'consistent with' language.

## ⚠️ Important Caveats For MSL Communication
Explicitly call out that this is an uncharacterized sequence: no approved indication, no clinical data, no validated mechanism. Clinical or therapeutic claims CANNOT be made. List the specific things an MSL must NOT say.

{_PLDDT_SECTION_INSTRUCTION_MSL}
Explicitly link the confidence band to how strongly any structural statement above can be made to an HCP.

## ❓ Likely HCP Questions
4–5 honest Q/A pairs an MSL might face about an unknown peptide (e.g. "Is this in trials?", "What does the AI confidence actually mean?", "How was the sequence chosen?", "What would you need to claim function?"). Answers must be evidence-honest.
{_FABRICATION_GUARD}"""
        system = MSL_UNKNOWN_SYSTEM
    else:
        prompt = f"""You are preparing a scientific briefing for a Medical Science Liaison (MSL) about **{protein_name}**.

**Protein details:**
- Name: {protein_name}
- Sequence length: {len(sequence)} amino acids
{conf_ctx}{insights_ctx}
Generate a Medical Affairs-ready scientific briefing with EXACTLY these sections, in this order:

## 🎯 Executive Summary (2-3 sentences)
Concise overview for an MSL preparing for an HCP conversation. What is this protein and why does it matter clinically?

## 📋 Key Scientific Talking Points
3–5 bullet points an MSL can use when engaging a specialist physician. Clear, defensible, scientifically accurate.

## ⚙️ Mechanism Explanation (HCP-friendly)
A 2–3 paragraph mechanism of action explanation at the level of a specialist physician. Avoid oversimplification but stay clinically framed.

## ❓ Anticipated HCP Questions
4–5 questions a KOL or specialist might ask, each with a brief MSL-appropriate response.

## 🔬 Scientific Differentiation Points
What is unique or scientifically notable about this protein vs related targets?

## 📚 Key Data to Know
Landmark trials, pivotal papers, or scientific findings an MSL should know.

## 🗓️ Suggested Discussion Framing
How should an MSL open and frame a scientific conversation about this protein with a KOL?

{_PLDDT_SECTION_INSTRUCTION_MSL}

## 🧪 Sequence-level Evidence
One short bullet list drawing from the Sequence Insights block above (MW, pI/charge, hydropathy, motif hits). If no insights block was provided, write 'Insufficient evidence to answer from sequence alone'.
{_FABRICATION_GUARD}
Keep all content scientifically accurate, balanced, and appropriate for Medical Affairs use. Do not make unsubstantiated claims."""
        system = MSL_SYSTEM

    try:
        return _venice_chat(prompt, system=system, temperature=0.55)
    except Exception as e:
        return (
            f"⚠️ Could not generate MSL summary: {str(e)}\n\n"
            "Check your Venice API key and internet connection."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Bonus: Quick sequence annotation
# ══════════════════════════════════════════════════════════════════════════════

def get_sequence_annotation(sequence: str) -> dict:
    """
    Return basic annotations about a sequence without API call.
    Used for quick stats display.
    """
    aa_properties = {
        "hydrophobic": set("AVILMFYW"),
        "polar": set("STNQ"),
        "charged_pos": set("KRH"),
        "charged_neg": set("DE"),
        "special": set("CGP"),
    }

    counts = {k: sum(1 for aa in sequence if aa in v) for k, v in aa_properties.items()}
    total = len(sequence)

    return {
        "length": total,
        "hydrophobic_pct": counts["hydrophobic"] / total * 100,
        "charged_pct": (counts["charged_pos"] + counts["charged_neg"]) / total * 100,
        "molecular_weight_kda": total * 0.110,
        "estimated_tm": "Globular" if total < 600 else "Likely multi-domain",
    }
