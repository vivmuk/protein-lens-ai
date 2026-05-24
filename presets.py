"""
Curated example proteins for ProteinLens AI demos.

Sequences are human proteins from UniProt (single-letter codes, no gaps).
Lengths are chosen to fold well on Modal / ESMFold within the app MVP limit.
"""

from __future__ import annotations

from typing import TypedDict


class ProteinPreset(TypedDict):
    id: str
    name: str
    description: str
    sequence: str


# Default selection when the app opens
DEFAULT_PRESET_ID = "insulin"


EXAMPLE_PROTEINS: list[ProteinPreset] = [
    {
        "id": "insulin",
        "name": "Human insulin (A+B chains)",
        "description": "Metabolic hormone · diabetes therapeutics · 51 AA",
        "sequence": "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    },
    {
        "id": "glp1",
        "name": "GLP-1 (7–36 amide)",
        "description": "Incretin · Ozempic/Wegovy mechanism · 30 AA",
        "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
    },
    {
        "id": "hemoglobin_alpha",
        "name": "Hemoglobin α (human)",
        "description": "Oxygen transport · classic structural biology · 141 AA",
        "sequence": (
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKY"
        ),
    },
    {
        "id": "lysozyme",
        "name": "Lysozyme (hen egg white)",
        "description": "Enzyme · teaching structure · 129 AA",
        "sequence": (
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
        ),
    },
    {
        "id": "egf",
        "name": "EGF (epidermal growth factor)",
        "description": "Growth factor · EGFR / oncology signaling · 53 AA",
        "sequence": "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR",
    },
    {
        "id": "p53_dbd",
        "name": "p53 DNA-binding domain",
        "description": "Tumor suppressor core · cancer biology · 194 AA",
        "sequence": (
            "SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEEN"
        ),
    },
    {
        "id": "trp_cage",
        "name": "Trp-cage (mini protein)",
        "description": "Fast sanity check · ~20 AA · good for smoke tests",
        "sequence": "NLYIQWLKDGGPSSGRPPPS",
    },
]


def get_preset(preset_id: str) -> ProteinPreset | None:
    for preset in EXAMPLE_PROTEINS:
        if preset["id"] == preset_id:
            return preset
    return None


def get_default_preset() -> ProteinPreset:
    return get_preset(DEFAULT_PRESET_ID) or EXAMPLE_PROTEINS[0]


def preset_select_label(preset: ProteinPreset) -> str:
    n = len(preset["sequence"])
    return f"{preset['name']} ({n} AA)"


def find_preset_by_label(label: str) -> ProteinPreset | None:
    for preset in EXAMPLE_PROTEINS:
        if preset_select_label(preset) == label:
            return preset
    return None
