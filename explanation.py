"""
explanation.py — Venice AI-powered scientific and MSL explanation engine.

Uses the Venice AI API (OpenAI-compatible) to generate:
  - Scientific explanations for structural biologists / researchers
  - Medical Affairs / MSL summaries for HCP communication
"""

import os
import requests
from typing import Optional

# ── Venice API config ──────────────────────────────────────────────────────────
VENICE_API_KEY = os.environ.get(
    "VENICE_API_KEY",
    "VENICE_INFERENCE_KEY_KntgMduZd8nN5cxhGd99gD3a5HpqD142saxKSn0uOs",
)
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL = "venice-uncensored"


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
            "max_tokens": 1200,
        },
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Venice API error {resp.status_code}: {resp.text[:300]}"
        )

    return resp.json()["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════════════════════
# Scientific explanation
# ══════════════════════════════════════════════════════════════════════════════

SCIENTIFIC_SYSTEM = """You are a world-class structural biologist and biochemist with expertise in:
- Protein structure-function relationships
- Disease biology and drug targets
- Mechanistic biochemistry
- Structural genomics

Provide clear, accurate, and insightful scientific explanations. Use markdown formatting
(headers, bold, bullet points) to structure your response. Be precise but accessible
to a PhD-level researcher. Never speculate beyond scientific consensus."""


def get_scientific_explanation(sequence: str, protein_name: str = "") -> str:
    """
    Generate a comprehensive scientific explanation for the folded protein.

    Args:
        sequence:     Amino acid sequence.
        protein_name: Optional human-readable protein name.

    Returns:
        Markdown-formatted scientific explanation.
    """
    name_ctx = f"**{protein_name}**" if protein_name and protein_name != "Unknown protein" else "an unknown protein"
    seq_preview = sequence[:60] + ("..." if len(sequence) > 60 else "")

    prompt = f"""A user has predicted the 3D structure of {name_ctx}.

**Protein details:**
- Name: {protein_name or "Unknown"}
- Sequence length: {len(sequence)} amino acids
- First 60 residues: `{seq_preview}`

Please provide a comprehensive scientific analysis covering:

## 🔬 Biological Function
What is the known or predicted function of this protein? What cellular processes does it participate in?

## 🧬 Structural Features
Based on the sequence length and identity, describe expected secondary/tertiary structural elements (alpha helices, beta sheets, domains, motifs).

## 🏥 Disease Relevance
What diseases, conditions, or phenotypes are associated with this protein or its dysfunction? Include any known mutations of significance.

## ⚙️ Mechanism of Action
How does this protein perform its function at the molecular level? Describe key interactions, catalytic mechanisms, or signaling roles.

## 💊 Therapeutic Relevance
Is this protein a drug target? Are there approved therapies, clinical candidates, or active research programs targeting it?

## 🔭 Scientific Context
Any notable research directions, recent discoveries, or open questions in the field?

Be precise, cite known biology, and use scientific language appropriate for a researcher or advanced student."""

    try:
        return _venice_chat(prompt, system=SCIENTIFIC_SYSTEM, temperature=0.5)
    except Exception as e:
        return f"⚠️ Could not generate explanation: {str(e)}\n\nCheck your Venice API key and internet connection."


# ══════════════════════════════════════════════════════════════════════════════
# MSL / Medical Affairs summary
# ══════════════════════════════════════════════════════════════════════════════

MSL_SYSTEM = """You are a senior Medical Science Liaison (MSL) with deep expertise in:
- Translational medicine and mechanism of action communication
- HCP (healthcare professional) scientific engagement
- Medical Affairs communication strategy
- Clinical data interpretation and scientific storytelling

Your role is to translate complex structural biology into clear, clinically relevant,
and HCP-appropriate scientific narratives. Be precise, evidence-based, and
strategically framed for medical-scientific discussions."""


def get_msl_summary(sequence: str, protein_name: str = "") -> str:
    """
    Generate a Medical Affairs / MSL-focused summary for the protein.

    Args:
        sequence:     Amino acid sequence.
        protein_name: Optional human-readable protein name.

    Returns:
        Markdown-formatted MSL talking points and summary.
    """
    name_ctx = protein_name if protein_name and protein_name != "Unknown protein" else "this protein"

    prompt = f"""You are preparing a scientific briefing for a Medical Science Liaison (MSL) about {name_ctx}.

**Protein details:**
- Name: {protein_name or "Unknown"}
- Sequence length: {len(sequence)} amino acids

Generate a Medical Affairs-ready scientific briefing with the following sections:

## 🎯 Executive Summary (2-3 sentences)
Concise overview for an MSL preparing for an HCP conversation. What is this protein and why does it matter clinically?

## 📋 Key Scientific Talking Points
3-5 bullet points an MSL can use when engaging a specialist physician. Clear, defensible, scientifically accurate.

## ⚙️ Mechanism Explanation (HCP-friendly)
A 2-3 paragraph mechanism of action explanation written at the level of a specialist physician (cardiologist, oncologist, neurologist, etc.). Avoid oversimplification but remain clinically framed.

## ❓ Anticipated HCP Questions
4-5 questions a KOL or specialist might ask about this protein, with brief MSL-appropriate responses.

## 🔬 Scientific Differentiation Points
What is unique or scientifically notable about this protein compared to related targets? What makes it interesting from a therapeutic perspective?

## 📚 Key Data to Know
What clinical trial data, landmark papers, or scientific findings should an MSL know when discussing this target?

## 🗓️ Suggested Discussion Framing
How should an MSL open and frame a scientific conversation about this protein with a KOL?

Keep all content scientifically accurate, balanced, and appropriate for Medical Affairs use. Do not make unsubstantiated claims."""

    try:
        return _venice_chat(prompt, system=MSL_SYSTEM, temperature=0.55)
    except Exception as e:
        return f"⚠️ Could not generate MSL summary: {str(e)}\n\nCheck your Venice API key and internet connection."


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
        "molecular_weight_kda": total * 0.110,  # ~110 Da per residue
        "estimated_tm": "Globular" if total < 600 else "Likely multi-domain",
    }
