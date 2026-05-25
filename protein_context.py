"""Public protein context lookups for ProteinLens.

All network calls in this module are best-effort. A failed public service should
remove one insight panel, not fail a local fold.
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

import requests

try:
    from Bio.Data.PDBData import protein_letters_3to1
    from Bio.PDB import PDBParser, Superimposer

    HAS_BIO_PDB = True
except Exception:  # pragma: no cover - only used for graceful degradation
    protein_letters_3to1 = {}
    PDBParser = None
    Superimposer = None
    HAS_BIO_PDB = False


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
INTERPRO_BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

REQUEST_TIMEOUT = 20
INTERPRO_APPS = "PfamA,SMART,PrositeProfiles,PrositePatterns"

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(sequence: str) -> str:
    return "".join(ch for ch in sequence.upper() if ch in STANDARD_AAS)


def safe_fetch_public_context(
    sequence: str,
    protein_name: str = "",
    pdb_string: str | None = None,
    uniprot_id: str | None = None,
    gene_symbol: str | None = None,
    interpro_wait_seconds: int = 35,
) -> dict[str, Any]:
    """Fetch all public context with per-service error isolation."""
    clean = sanitize_sequence(sequence)
    context: dict[str, Any] = {
        "uniprot": {"status": "skipped"},
        "rcsb": {"status": "skipped", "hits": []},
        "interpro": {"status": "skipped", "domains": []},
        "open_targets": {"status": "skipped"},
        "alphafold": {"status": "skipped"},
    }

    metadata = {
        "uniprot_id": uniprot_id,
        "gene_symbol": gene_symbol,
        "protein_name": protein_name,
    }
    if not uniprot_id and protein_name and protein_name.lower() not in {
        "unknown",
        "unknown peptide",
        "custom peptide",
    }:
        context["uniprot"] = _guarded("uniprot", lambda: lookup_uniprot_metadata(protein_name, clean))
        if context["uniprot"].get("status") == "ok":
            metadata["uniprot_id"] = context["uniprot"].get("accession") or uniprot_id
            metadata["gene_symbol"] = context["uniprot"].get("gene_symbol") or gene_symbol
    else:
        if uniprot_id or gene_symbol:
            context["uniprot"] = {
                "status": "ok",
                "accession": uniprot_id,
                "gene_symbol": gene_symbol,
                "protein_name": protein_name,
            }

    context["rcsb"] = _guarded("rcsb", lambda: search_rcsb_by_sequence(clean))
    context["interpro"] = _guarded(
        "interpro",
        lambda: run_interproscan(clean, wait_seconds=interpro_wait_seconds),
    )

    ot_query = metadata.get("gene_symbol") or metadata.get("uniprot_id") or protein_name
    if ot_query:
        context["open_targets"] = _guarded(
            "open_targets",
            lambda: fetch_open_targets_context(str(ot_query)),
        )

    af_uniprot_id = metadata.get("uniprot_id")
    if af_uniprot_id:
        context["alphafold"] = _guarded(
            "alphafold",
            lambda: fetch_alphafold_reference(
                str(af_uniprot_id),
                clean,
                pdb_string=pdb_string,
            ),
        )

    return context


def _guarded(service: str, fn) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {"status": "error", "service": service, "message": str(exc)}


def lookup_uniprot_metadata(query: str, sequence: str = "") -> dict[str, Any]:
    """Return accession and gene metadata for a protein-name query."""
    resp = requests.get(
        UNIPROT_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "size": 5,
            "fields": "accession,id,gene_primary,protein_name,organism_name,sequence",
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return {"status": "not_found", "query": query}

    clean = sanitize_sequence(sequence)
    exact_match = None
    human_match = None
    for hit in results:
        hit_seq = sanitize_sequence(hit.get("sequence", {}).get("value", ""))
        organism = hit.get("organism", {}).get("scientificName", "")
        if clean and hit_seq == clean:
            exact_match = hit
            break
        if organism == "Homo sapiens" and human_match is None:
            human_match = hit
    chosen = exact_match or human_match or results[0]

    gene_symbol = None
    genes = chosen.get("genes") or []
    if genes:
        gene_symbol = genes[0].get("geneName", {}).get("value")

    return {
        "status": "ok",
        "query": query,
        "accession": chosen.get("primaryAccession"),
        "uniprot_id": chosen.get("uniProtkbId"),
        "gene_symbol": gene_symbol,
        "protein_name": _uniprot_protein_name(chosen) or query,
        "organism": chosen.get("organism", {}).get("scientificName"),
        "sequence_length": chosen.get("sequence", {}).get("length"),
        "exact_sequence_match": bool(exact_match),
    }


def _uniprot_protein_name(hit: dict[str, Any]) -> str | None:
    desc = hit.get("proteinDescription") or {}
    try:
        return desc["recommendedName"]["fullName"]["value"]
    except (KeyError, TypeError):
        pass
    submitted = desc.get("submittedNames") or []
    if submitted:
        try:
            return submitted[0]["fullName"]["value"]
        except (KeyError, TypeError):
            return None
    return None


def search_rcsb_by_sequence(sequence: str, limit: int = 3) -> dict[str, Any]:
    """Search RCSB experimental polymer entities by protein sequence."""
    if len(sequence) < 10:
        return {"status": "skipped", "hits": [], "message": "Sequence too short for RCSB search."}

    payload = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": 0.3,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "results_content_type": ["experimental"],
            "scoring_strategy": "sequence",
            "results_verbosity": "verbose",
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    resp = requests.post(RCSB_SEARCH_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    hits = []
    for item in data.get("result_set", [])[:limit]:
        identifier = item.get("identifier", "")
        if "_" not in identifier:
            continue
        pdb_id, entity_id = identifier.split("_", 1)
        match = _extract_rcsb_match_context(item)
        details = _fetch_rcsb_metadata(pdb_id, entity_id)
        hits.append(
            {
                "pdb_id": pdb_id.upper(),
                "entity_id": entity_id,
                "title": details.get("title") or details.get("description") or "Untitled structure",
                "description": details.get("description"),
                "organism": details.get("organism"),
                "method": details.get("method"),
                "identity_pct": (
                    match.get("sequence_identity", 0.0) * 100
                    if match.get("sequence_identity") is not None
                    else None
                ),
                "alignment_length": match.get("alignment_length"),
                "query_range": _range_text(match.get("query_beg"), match.get("query_end")),
                "subject_range": _range_text(match.get("subject_beg"), match.get("subject_end")),
                "evalue": match.get("evalue"),
                "score": item.get("score"),
                "url": f"https://www.rcsb.org/structure/{pdb_id.upper()}",
            }
        )

    return {"status": "ok", "total_count": data.get("total_count", 0), "hits": hits}


def _extract_rcsb_match_context(item: dict[str, Any]) -> dict[str, Any]:
    for service in item.get("services", []):
        for node in service.get("nodes", []):
            contexts = node.get("match_context") or []
            if contexts:
                return contexts[0]
    return {}


def _fetch_rcsb_metadata(pdb_id: str, entity_id: str) -> dict[str, Any]:
    entry_resp = requests.get(RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=REQUEST_TIMEOUT)
    entity_resp = requests.get(
        RCSB_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id),
        timeout=REQUEST_TIMEOUT,
    )
    entry_resp.raise_for_status()
    entity_resp.raise_for_status()
    entry = entry_resp.json()
    entity = entity_resp.json()

    organisms = entity.get("rcsb_entity_source_organism") or entity.get("entity_src_nat") or []
    organism = None
    if organisms:
        organism = (
            organisms[0].get("ncbi_scientific_name")
            or organisms[0].get("pdbx_organism_scientific")
            or organisms[0].get("scientific_name")
        )

    methods = entry.get("exptl") or []
    return {
        "title": (entry.get("struct") or {}).get("title"),
        "method": methods[0].get("method") if methods else None,
        "description": (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
        "organism": organism,
    }


def _range_text(start: Any, end: Any) -> str | None:
    if start is None or end is None:
        return None
    return f"{start}-{end}"


def run_interproscan(sequence: str, wait_seconds: int = 35) -> dict[str, Any]:
    """Submit a sequence to InterProScan and wait briefly for domain results."""
    if len(sequence) < 10:
        return {"status": "skipped", "domains": [], "message": "Sequence too short."}

    email = os.environ.get("INTERPRO_EMAIL", "proteinlens@example.com")
    submit = requests.post(
        f"{INTERPRO_BASE_URL}/run",
        data={
            "email": email,
            "title": "ProteinLens",
            "sequence": sequence,
            "stype": "p",
            "appl": INTERPRO_APPS,
            "goterms": "false",
            "pathways": "false",
        },
        timeout=REQUEST_TIMEOUT,
    )
    submit.raise_for_status()
    job_id = submit.text.strip()
    return poll_interproscan(job_id, wait_seconds=wait_seconds)


def poll_interproscan(job_id: str, wait_seconds: int = 10) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, wait_seconds)
    status = "RUNNING"
    while time.monotonic() <= deadline:
        resp = requests.get(f"{INTERPRO_BASE_URL}/status/{job_id}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        status = resp.text.strip()
        if status == "FINISHED":
            result = requests.get(f"{INTERPRO_BASE_URL}/result/{job_id}/json", timeout=REQUEST_TIMEOUT)
            result.raise_for_status()
            domains = parse_interpro_json(result.json())
            return {"status": "ok", "job_id": job_id, "domains": domains}
        if status in {"ERROR", "FAILURE", "NOT_FOUND"}:
            return {"status": "error", "job_id": job_id, "message": status, "domains": []}
        time.sleep(3)
    return {
        "status": "pending",
        "job_id": job_id,
        "message": f"InterProScan still {status}; refresh to poll later.",
        "domains": [],
    }


def parse_interpro_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    domains = []
    for result in data.get("results", []):
        for match in result.get("matches", []):
            signature = match.get("signature") or {}
            entry = signature.get("entry") or {}
            locations = match.get("locations") or []
            for loc in locations:
                fragments = loc.get("fragments") or [loc]
                for frag in fragments:
                    start = frag.get("start") or loc.get("start")
                    end = frag.get("end") or loc.get("end")
                    if start is None or end is None:
                        continue
                    library = (
                        signature.get("signatureLibraryRelease", {}).get("library")
                        or signature.get("library")
                        or match.get("signatureLibrary")
                    )
                    domains.append(
                        {
                            "name": (
                                entry.get("name")
                                or signature.get("name")
                                or signature.get("description")
                                or "InterPro match"
                            ),
                            "accession": entry.get("accession") or signature.get("accession"),
                            "signature": signature.get("accession"),
                            "library": library,
                            "start": int(start),
                            "end": int(end),
                            "description": entry.get("description") or signature.get("description"),
                            "evalue": loc.get("evalue"),
                            "score": loc.get("score"),
                        }
                    )
    domains.sort(key=lambda d: (d["start"], d["end"], d.get("name") or ""))
    return _dedupe_domains(domains)


def _dedupe_domains(domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for domain in domains:
        key = (
            domain.get("accession"),
            domain.get("signature"),
            domain.get("start"),
            domain.get("end"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(domain)
    return out


def fetch_open_targets_context(query: str) -> dict[str, Any]:
    """Resolve a target query, then fetch diseases, tractability, and clinical candidates."""
    search_query = """
    query SearchTarget($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 3}) {
        hits {
          id
          name
          entity
          object {
            ... on Target {
              id
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
    """
    target_id = _graphql(
        search_query,
        {"q": query},
    ).get("search", {}).get("hits", [])
    if not target_id:
        return {"status": "not_found", "query": query}

    chosen = target_id[0].get("object") or {"id": target_id[0].get("id"), "approvedSymbol": target_id[0].get("name")}
    ensembl_id = chosen.get("id")
    if not ensembl_id:
        return {"status": "not_found", "query": query}

    target_query = """
    query TargetContext($id: String!) {
      target(ensemblId: $id) {
        id
        approvedSymbol
        approvedName
        biotype
        tractability {
          label
          modality
          value
        }
        associatedDiseases(page: {index: 0, size: 5}) {
          rows {
            disease { id name }
            score
            datatypeScores { id score }
          }
        }
        drugAndClinicalCandidates {
          count
          rows {
            drug { id name }
            maxClinicalStage
            diseases {
              diseaseFromSource
              disease { id name }
            }
          }
        }
      }
    }
    """
    target = _graphql(target_query, {"id": ensembl_id}).get("target")
    if not target:
        return {"status": "not_found", "query": query}

    return {
        "status": "ok",
        "query": query,
        "target": {
            "id": target.get("id"),
            "symbol": target.get("approvedSymbol"),
            "name": target.get("approvedName"),
            "biotype": target.get("biotype"),
        },
        "tractability": [
            row
            for row in target.get("tractability", [])
            if row.get("value") is True
        ][:8],
        "diseases": [
            {
                "id": (row.get("disease") or {}).get("id"),
                "name": (row.get("disease") or {}).get("name"),
                "score": row.get("score"),
                "evidence": sorted(
                    row.get("datatypeScores", []),
                    key=lambda x: x.get("score", 0),
                    reverse=True,
                )[:3],
            }
            for row in target.get("associatedDiseases", {}).get("rows", [])
        ],
        "drugs": [
            {
                "id": (row.get("drug") or {}).get("id"),
                "name": (row.get("drug") or {}).get("name"),
                "max_clinical_stage": row.get("maxClinicalStage"),
                "diseases": [
                    ((d.get("disease") or {}).get("name") or d.get("diseaseFromSource"))
                    for d in row.get("diseases", [])[:3]
                ],
            }
            for row in target.get("drugAndClinicalCandidates", {}).get("rows", [])[:5]
            if row.get("drug")
        ],
        "drug_count": target.get("drugAndClinicalCandidates", {}).get("count", 0),
    }


def _graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        OPENTARGETS_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        message = payload["errors"][0].get("message", "Open Targets GraphQL error")
        raise RuntimeError(message)
    return payload.get("data", {})


def fetch_alphafold_reference(
    uniprot_id: str,
    sequence: str,
    pdb_string: str | None = None,
) -> dict[str, Any]:
    resp = requests.get(ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    records = resp.json()
    if not records:
        return {"status": "not_found", "uniprot_id": uniprot_id}

    record = records[0]
    pdb_url = record.get("pdbUrl")
    if not pdb_url:
        return {"status": "not_found", "uniprot_id": uniprot_id, "message": "No PDB URL in AlphaFold DB."}

    pdb_resp = requests.get(pdb_url, timeout=REQUEST_TIMEOUT)
    pdb_resp.raise_for_status()
    reference_pdb = pdb_resp.text

    comparison = None
    if pdb_string:
        comparison = calculate_ca_rmsd(
            predicted_pdb=pdb_string,
            reference_pdb=reference_pdb,
            query_sequence=sequence,
            reference_sequence=record.get("sequence") or record.get("uniprotSequence") or "",
        )

    return {
        "status": "ok",
        "uniprot_id": uniprot_id,
        "entry_id": record.get("entryId") or record.get("modelEntityId"),
        "gene": record.get("gene"),
        "organism": record.get("organismScientificName"),
        "description": record.get("uniprotDescription"),
        "latest_version": record.get("latestVersion"),
        "global_plddt": record.get("globalMetricValue"),
        "pdb_url": pdb_url,
        "pdb_string": reference_pdb,
        "sequence_length": len(record.get("sequence") or record.get("uniprotSequence") or ""),
        "comparison": comparison,
    }


def calculate_ca_rmsd(
    predicted_pdb: str,
    reference_pdb: str,
    query_sequence: str,
    reference_sequence: str,
) -> dict[str, Any]:
    if not HAS_BIO_PDB:
        return {"status": "skipped", "message": "Bio.PDB is unavailable."}

    pred_seq, pred_atoms = _ca_atoms_and_sequence(predicted_pdb, "predicted")
    ref_seq, ref_atoms = _ca_atoms_and_sequence(reference_pdb, "reference")
    clean_query = sanitize_sequence(query_sequence)
    clean_ref = sanitize_sequence(reference_sequence) or ref_seq

    if not pred_atoms or not ref_atoms:
        return {"status": "skipped", "message": "Missing CA atoms for RMSD."}

    ref_start = clean_ref.find(clean_query) if clean_query else -1
    if ref_start >= 0 and len(pred_atoms) >= len(clean_query) and len(ref_atoms) >= ref_start + len(clean_query):
        pred_sel = pred_atoms[: len(clean_query)]
        ref_sel = ref_atoms[ref_start : ref_start + len(clean_query)]
        method = f"query sequence aligned to AlphaFold residues {ref_start + 1}-{ref_start + len(clean_query)}"
    elif pred_seq == ref_seq and len(pred_atoms) == len(ref_atoms):
        pred_sel = pred_atoms
        ref_sel = ref_atoms
        method = "full-length CA alignment"
    elif len(pred_atoms) == len(ref_atoms):
        pred_sel = pred_atoms
        ref_sel = ref_atoms
        method = "same-length CA alignment"
    else:
        return {
            "status": "skipped",
            "message": (
                "Prediction and AlphaFold canonical sequence have different residue sets; "
                "RMSD needs a reliable sequence alignment."
            ),
            "predicted_residues": len(pred_atoms),
            "reference_residues": len(ref_atoms),
        }

    if len(pred_sel) < 3:
        return {"status": "skipped", "message": "Need at least 3 aligned CA atoms."}

    sup = Superimposer()
    sup.set_atoms(ref_sel, pred_sel)
    return {
        "status": "ok",
        "rmsd": float(sup.rms),
        "aligned_residues": len(pred_sel),
        "method": method,
    }


def _ca_atoms_and_sequence(pdb_string: str, structure_id: str) -> tuple[str, list[Any]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(structure_id, io.StringIO(pdb_string))
    residues = []
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue
                resname = residue.get_resname().upper()
                one = protein_letters_3to1.get(resname, "X")
                residues.append(one)
                atoms.append(residue["CA"])
        break
    return "".join(residues), atoms


def public_context_for_prompt(context: dict[str, Any] | None) -> str:
    if not context:
        return ""

    lines = ["\n**Public Database Context (best-effort; do not invent beyond these hits):**"]
    rcsb = context.get("rcsb", {})
    if rcsb.get("status") == "ok" and rcsb.get("hits"):
        lines.append("- RCSB similar experimental structures:")
        for hit in rcsb["hits"][:3]:
            identity = hit.get("identity_pct")
            match = (
                f"{identity:.1f}% over {hit.get('alignment_length')} residues"
                if identity is not None and hit.get("alignment_length")
                else "sequence-similar hit"
            )
            lines.append(
                f"  - {hit.get('pdb_id')}: {hit.get('title')} ({hit.get('organism') or 'organism unknown'}), {match}"
            )

    interpro = context.get("interpro", {})
    domains = interpro.get("domains") or []
    if domains:
        lines.append("- InterProScan domains:")
        for domain in domains[:8]:
            lines.append(
                f"  - {domain.get('library') or 'InterPro'} {domain.get('name')} "
                f"residues {domain.get('start')}-{domain.get('end')}"
            )

    ot = context.get("open_targets", {})
    if ot.get("status") == "ok":
        target = ot.get("target", {})
        lines.append(
            f"- Open Targets target: {target.get('symbol')} ({target.get('name')}); "
            f"{len(ot.get('diseases', []))} disease links shown, {ot.get('drug_count', 0)} clinical candidate links."
        )

    af = context.get("alphafold", {})
    if af.get("status") == "ok":
        comp = af.get("comparison") or {}
        if comp.get("status") == "ok":
            lines.append(
                f"- AlphaFold DB reference {af.get('entry_id')} v{af.get('latest_version')}: "
                f"CA RMSD {comp.get('rmsd'):.2f} A over {comp.get('aligned_residues')} residues."
            )
        else:
            lines.append(
                f"- AlphaFold DB reference {af.get('entry_id')} v{af.get('latest_version')}: "
                f"global pLDDT {af.get('global_plddt')}."
            )

    return "\n".join(lines) + "\n"


def serializable_public_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return context with bulky PDB payloads stripped for JSON downloads."""
    if not context:
        return None
    cleaned = {}
    for key, value in context.items():
        if isinstance(value, dict):
            cleaned[key] = {k: v for k, v in value.items() if k != "pdb_string"}
        else:
            cleaned[key] = value
    return cleaned
