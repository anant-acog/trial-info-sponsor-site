import os
import asyncio
import sys
import json
import re
import aiohttp
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Dict, List, Tuple
import hashlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baml_client import b

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

VALID_STATUS_CODES = set(range(200, 300)) | set(range(300, 400))


import dataclasses
from typing import Any

def to_primitive(obj: Any) -> Any:
    """
    Recursively convert objects (Pydantic models, dataclasses, objects with __dict__, lists)
    into plain Python primitives (dict/list/str/numbers) safe for json.dumps().
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple, set)):
        return [to_primitive(v) for v in obj]

    if isinstance(obj, dict):
        return {k: to_primitive(v) for k, v in obj.items()}

    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            dumped = obj.model_dump()
            return to_primitive(dumped)
        except Exception:
            pass

    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            dumped = obj.dict()
            return to_primitive(dumped)
        except Exception:
            pass

    try:
        if dataclasses.is_dataclass(obj):
            return to_primitive(dataclasses.asdict(obj))
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        try:
            return to_primitive(vars(obj))
        except Exception:
            pass

    try:
        return str(obj)
    except Exception:
        return None

async def is_url_working(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = 8
) -> bool:
    try:
        async with session.head(url, allow_redirects=True, timeout=timeout) as resp:
            if resp.status in VALID_STATUS_CODES:
                return True

        async with session.get(url, allow_redirects=True, timeout=timeout) as resp:
            return resp.status in VALID_STATUS_CODES

    except Exception:
        return False

async def filter_working_urls(urls: List[str]) -> List[str]:
    if not urls:
        return []

    timeout = aiohttp.ClientTimeout(total=12)
    connector = aiohttp.TCPConnector(limit=20)

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={"User-Agent": "ClinicalPipelineValidator/1.0"}
    ) as session:
        checks = [is_url_working(session, u) for u in urls]
        results = await asyncio.gather(*checks, return_exceptions=True)

    return [
        url for url, ok in zip(urls, results)
        if ok is True
    ]

def md5_signature(*parts: str) -> str:
    joined = "||".join(p.lower().strip() for p in parts if p)
    return hashlib.md5(joined.encode()).hexdigest()

def extract_json_object(text: str) -> dict:
    """
    Robustly extracts the first valid JSON object from a Gemini response.
    Handles markdown fences and surrounding prose.
    """
    if not text:
        return {}
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def build_target_to_urls_map(search_report: Dict) -> Dict[str, List[str]]:
    """
    Builds a mapping:
    normalized_target_name -> evidence_urls

    URLs come ONLY from the Gemini-generated search_report.
    """
    mapping: Dict[str, set] = {}

    for entry in search_report.get("molecular_targets", []):
        target = entry.get("target_name", "").strip().lower()
        urls = entry.get("evidence_urls", [])

        if not target or not urls:
            continue

        mapping.setdefault(target, set()).update(urls)

    return {k: list(v) for k, v in mapping.items()}

def extract_grounding_urls_from_response(genai_response) -> list:
    """
    Extract grounding chunk URIs from the model response.

    The response format can vary depending on SDK/proto. This function attempts
    to handle common shapes similar to the example you provided:
    response.candidates -> each candidate has groundingMetadata -> groundingChunks -> [{'web': {'uri': '...', ...}}, ...]
    """
    urls = []
    try:
        candidates = getattr(genai_response, "candidates", None) or genai_response.get("candidates", None) or []
    except Exception:
        candidates = []

    if isinstance(candidates, dict):
        candidates = [candidates]

    for cand in candidates:
        gm = None
        try:
            gm = getattr(cand, "groundingMetadata", None)
        except Exception:
            pass
        if gm is None:
           
            try:
                gm = cand.get("groundingMetadata", None)
            except Exception:
                gm = None

        if not gm:
            continue

        grounding_chunks = None
        try:
            grounding_chunks = getattr(gm, "groundingChunks", None)
        except Exception:
            pass
        if grounding_chunks is None:
            try:
                grounding_chunks = gm.get("groundingChunks", None)
            except Exception:
                grounding_chunks = None

        if not grounding_chunks:
            continue

        for chunk in grounding_chunks:
            web_field = None
            try:
                web_field = getattr(chunk, "web", None)
            except Exception:
                pass
            if web_field is None:
                try:
                    web_field = chunk.get("web", None)
                except Exception:
                    web_field = None

            if not web_field:
                continue

            uri = None
            try:
                uri = getattr(web_field, "uri", None)
            except Exception:
                pass
            if not uri:
                try:
                    uri = web_field.get("uri", None)
                except Exception:
                    uri = None

            if uri and isinstance(uri, str):
                urls.append(uri)

    seen = set()
    uniq_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq_urls.append(u)

    return uniq_urls


async def get_pipeline_data(company: str):
    print(f"\nAgent is researching {company}...")

    agent_prompt = f"""
### PERSONA
You are an expert Clinical Data Analyst. Your goal is to extract the official clinical trial pipeline for {company}.


### OBJECTIVE
Perform a comprehensive web search to find the official clinical trial pipeline for {company}. Extract ALL drug candidates currently in development, capturing every single asset from Preclinical up to Filed/Approved status.


### SEARCH STRATEGY
1.  **Primary Search:** Search for "{company} official clinical pipeline" and "{company} R&D portfolio". Prioritize the official company website.
2.  **Secondary Validation (News & PR):** Search for recent "{company} press releases" and specific queries on major biotech news outlets to find recent updates, discontinuations, or phase changes that may not be on the main website yet.
    * *Keywords to use:* "{company} clinical results", "{company} program update", "{company} pipeline STAT news", "{company} FierceBiotech", "{company} Endpoints News", "{company} BioPharma Dive".
3.  **Cross-Referencing:** If the official website is vague (e.g., "multiple targets"), use the news sources above to identify specific candidate names.


### EXTRACTION RULES (STRICT)
* **Completeness:** List EVERYTHING found. Do not summarize. Do not filter for "top" assets only.
* **URL Accuracy:** **Do not construct or guess URLs** (e.g., do not invent `company.com/pipeline/drug-name`). Use the *groundingChunks.web.uri* URL returned by your search tool. 
* **Formatting:** Return the data ONLY as a valid JSON list. Do not include markdown formatting (like ```json) or conversational filler text.
* **Context Management:** Prioritize accuracy. If the list is long, finish the current object cleanly and stop rather than outputting broken JSON.


### DATA SCHEMA (JSON)
For each drug candidate, use this exact structure:


[
  {{
    "sponsor_name": "string (Official company name)",
    "candidate_name": "string (Drug code like 'VX-548' or name. Do not include dosage)",
    "disease_indication": [
      "string (Specific disease, e.g., 'Cystic Fibrosis'. Avoid generic terms like 'pain' if 'Acute Pain' is specified)"
    ],
    "development_phase": "string (Normalize to: 'Preclinical', 'Phase I', 'Phase II', 'Phase III', 'Filed', 'Approved')",
    "moa": "string or null (Mechanism of Action, e.g., 'NaV1.8 Inhibitor')", 
    "route_of_admin": "string or null (e.g., 'Oral', 'IV')",
    "status": "string (Default to 'Active' unless 'Discontinued' is explicitly stated)",
    "source_url": "string (The url which you got from the search tool starting for a particular candidate starts with https://vertexaisearch.cloud.google.com/grounding-api-redirect/)",
  }}
]


### CRITICAL INSTRUCTION
Do not provide an intro or outro. Output **only** the raw JSON list. Begin the search now for {company}.
    """


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=agent_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
    )

    search_report = getattr(response, "text", None)
    if not search_report:
        print(f"Agent found nothing for {company}")
        return

    grounding_urls = extract_grounding_urls_from_response(response)
    if grounding_urls:
        print(f"Found grounding URLs ({len(grounding_urls)}):")
        for u in grounding_urls:
            print("  -", u)
    
    
    print(f"Structuring data for {company}...")

    pipeline_data = None
    try:
        pipeline_data = b.ExtractPipeline(
            search_report=search_report,
            company=company,
            grounding_urls=grounding_urls  
        )
        print(f"SUCCESS: b.ExtractPipeline returned {len(pipeline_data)} records (with grounding_urls param).")
    except TypeError:
        try:
            pipeline_data = b.ExtractPipeline(
                search_report=search_report,
                company=company,
                grounding_urls=grounding_urls
            )
            print(f"SUCCESS: b.ExtractPipeline returned {len(pipeline_data)} records (without grounding_urls param).")
        except Exception as e:
            print(f"BAML Error when calling ExtractPipeline: {e}")
            return
    except Exception as e:
        print(f"BAML Error: {e}")
        return

    if not pipeline_data:
        print("No pipeline data returned by ExtractPipeline.")
        return

    enriched = []
    for item in pipeline_data:
        item_dict = to_primitive(item)

        existing = item_dict.get("source_url", None)
        merged_urls = []

        if existing:
            if isinstance(existing, list):
                merged_urls.extend(existing)
            elif isinstance(existing, str):
                merged_urls.append(existing)

        for u in grounding_urls:
            if u not in merged_urls:
                merged_urls.append(u)

        final_source = await filter_working_urls(merged_urls)

        item_dict["source_url"] = final_source

        if "status" not in item_dict or item_dict.get("status") is None:
            item_dict["status"] = "Active"

        enriched.append(item_dict)

    return enriched

async def get_search_report_for_drug(drug_name: str, disease_name: str, run_id: int) -> Dict:
    print(f"\nRun {run_id + 1}: Generating search report for {drug_name} in the context of {disease_name}")

    prompt = f"""
Role & Objective
You are assisting a Senior Drug Discovery Intelligence Analyst. Generate a STRICTLY STRUCTURED JSON report identifying the molecular target(s) of the drug "{drug_name}" in the context of {disease_name}.
The report must include:
Target name(s)
Corresponding human gene symbol(s)
Mechanism of action (MoA)
Regulatory approval status of the drug for the specified disease

Data Source Hierarchy (in order of priority)
Peer-reviewed scientific literature
ChEMBL
DrugBank/UniProt
If no explicit molecular target is reported in any of the above sources, the target must be explicitly reported as "Unknown".

Strict Rules
Include ONLY explicitly named molecular targets
DO NOT infer, predict, or speculate
Restrict targets to Homo sapiens
Each target MUST have at least one publicly accessible evidence URL
Report only molecular-level targets (no pathways, phenotypes, or biomarkers)
Return JSON ONLY (no explanations, no markdown, no prose)

Field Definitions
target_name: Official protein/complex/miRNA name
gene_symbol: Standard HGNC gene symbol
target_type: One of SINGLE PROTEIN or PROTEIN COMPLEX or PROTEIN FAMILY or miRNA or Other 
moa: Mechanism of action (e.g., inhibitor, antagonist, agonist, etc)
approval_status: Regulatory approval status (FDA/EMA/etc.) of the drug only for the specified disease

JSON schema:
{{
  "drug_name": "{drug_name}",
  "disease_name": "{disease_name}"
  "molecular_targets": [
    {{
      "target_name": "string",
      "gene_symbol" : "string",
      "target_type": "SINGLE PROTEIN | PROTEIN COMPLEX | PROTEIN FAMILY | miRNA | Other",
      "moa": "string",
      "approval_status": "Approved" | "Not approved", 
      "evidence_urls": ["string"]
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )

    raw_text = getattr(response, "text", "")
    parsed = extract_json_object(raw_text)

    if not parsed:
        print("Failed to extract JSON from Gemini response")
        return {}

    return parsed

def extract_target_cores(search_report: Dict, drug_name: str, disease_name: str):
    """
    BAML Extracts:
    - target_name
    - target_type
    - moa
    - gene_symbol
    - approval_status
    """
    return b.ExtractTargetCoresByDrugAndDisease(
        search_report=json.dumps(search_report),
        drug_name=drug_name,
        disease_name=disease_name
    )



async def resolve_targets_for_drug(
    drug_name: str,
    disease_name: str,
    run_id: int
) -> List[Dict]:

    search_report = await get_search_report_for_drug(drug_name, disease_name, run_id)
    if not search_report:
        return []

    target_url_map = build_target_to_urls_map(search_report)
    target_cores = extract_target_cores(search_report, drug_name, disease_name)

    final_results = []

    for t in target_cores:
        try:
            core = t.model_dump()
        except Exception:
            core = dict(t)

        key = core["target_name"].lower().strip()
        urls = await filter_working_urls(target_url_map.get(key, []))

        if not urls:
            continue

        final_results.append({
            "drug_name": drug_name,
            "disease_name": disease_name,
            "target_name": core["target_name"],
            "target_type": core["target_type"],
            "approval_status": core["approval_status"],
            "gene_symbol": core["gene_symbol"],
            "moa": core["moa"],
            "evidence_urls": urls
        })

    return final_results

async def aggregate_drugs(
    drugs: List[str],
    diseases: List[str],
    num_runs: int = 3
) -> List[Dict]:

    all_entries: List[Dict] = []
    seen = set()

    for drug, disease in zip(drugs, diseases):
        for run_id in range(num_runs):
            entries = await resolve_targets_for_drug(drug, disease, run_id)

            for e in entries:
                sig = md5_signature(
                    e["drug_name"],
                    e["disease_name"],
                    e["target_name"],
                    e["moa"],
                    e["approval_status"],
                    e["gene_symbol"]
                )

                if sig in seen:
                    continue

                seen.add(sig)
                all_entries.append(e)
    return all_entries

async def main():
    
    companies_to_analyze = ["Mission Therapeutics"]  #Merck Sharp & Dohme LLC Mission Therapeutics
    for company in companies_to_analyze:
        data = await get_pipeline_data(company)
        print("====COMPANIES RESULTS====")
        json_text = json.dumps(data, indent=2)
        print(json_text)

    drugs = ["CRD-4730", "RIOCIGUAT"]
    diseases = ["Cardiovascular diseases","pulmonary arterial hypertension"]
    print("========DRUG & DISEASE TO TARGET RESULTS========")
    data = await aggregate_drugs(drugs, diseases)
    json_text = json.dumps(data, indent=2)
    print(json_text)

if __name__ == "__main__":
    asyncio.run(main())