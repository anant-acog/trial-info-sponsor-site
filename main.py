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

    # Strip markdown fences
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} block
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
    else:
        print(f"No grounding URLs found for {company}")
    
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
        try:
            item_dict = item.model_dump()
        except Exception:
            try:
                item_dict = dict(item)
            except Exception:
                item_dict = {}

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

        try:
            if hasattr(item, "source_url"):
                try:
                    setattr(item, "source_url", final_source)
                    enriched.append(item)
                    continue
                except Exception:
                    pass

            item_dict["source_url"] = final_source
            enriched.append(item_dict)
        except Exception:
            minimal = {
                "sponsor_name": item_dict.get("sponsor_name", company),
                "candidate_name": item_dict.get("candidate_name", None),
                "disease_indication": item_dict.get("disease_indication", []),
                "development_phase": item_dict.get("development_phase", None),
                "moa": item_dict.get("moa", None),
                "route_of_admin": item_dict.get("route_of_admin", None),
                "status": item_dict.get("status", "Active"),
                "source_url": final_source
            }
            enriched.append(minimal)

    return enriched

async def get_search_report_for_drug(drug_name: str, run_id: int) -> Dict:
    print(f"\nRun {run_id + 1}: Generating search report for {drug_name}")

    prompt = f"""
You are a Senior Drug Discovery Intelligence Analyst.

Objective:
Your goal is to generate a STRICTLY STRUCTURED JSON report and provide the target name and the corresponding gene symbol for the drug "{drug_name}".Prioritize data from peer-reviewed literature followed by databases search like ChEMBL, DrugBank, and UniProt. If no information is available about the drug's target from literature or any of the sources listed above, explicitly state it to be "Unknown"

Rules:
- Include ONLY explicitly named molecular targets
- Restrict results to human (Homo sapiens) targets.
- Each target MUST include evidence_urls (public sources)
- Do NOT infer or speculate
- Return JSON ONLY (no prose)

JSON schema:
{{
  "drug_name": "{drug_name}",
  "molecular_targets": [
    {{
      "target_name": "string",
      "Gene symbol" : "string",
      "target_type": "SINGLE PROTEIN | PROTEIN COMPLEX | PROTEIN FAMILY | Other",
      "action": "Agonist | Antagonist | Inhibitor | Modulator | Binder | Other",
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

def extract_target_cores(search_report: Dict, drug_name: str):
    """
    BAML extracts ONLY:
    - target_name
    - target_type
    - action
    """
    return b.ExtractTargetCoresByDrug(
        search_report=json.dumps(search_report),
        drug_name=drug_name
    )



async def resolve_targets_for_drug(
    drug_name: str,
    run_id: int
) -> List[Dict]:

    search_report = await get_search_report_for_drug(drug_name, run_id)
    if not search_report:
        return []

    target_url_map = build_target_to_urls_map(search_report)
    target_cores = extract_target_cores(search_report, drug_name)

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
            "target_name": core["target_name"],
            "target_type": core["target_type"],
            "action": core["action"],
            "evidence_urls": urls
        })

    return final_results

async def aggregate_drugs(
    drugs: List[str],
    num_runs: int = 3
) -> List[Dict]:

    all_entries: List[Dict] = []
    seen = set()

    for drug in drugs:
        for run_id in range(num_runs):
            entries = await resolve_targets_for_drug(drug, run_id)

            for e in entries:
                sig = md5_signature(
                    e["drug_name"],
                    e["target_name"],
                    e["action"]
                )

                if sig in seen:
                    continue

                seen.add(sig)
                all_entries.append(e)

    return all_entries

async def main():
    
    companies_to_analyze = [""] 
    for company in companies_to_analyze:
        data = await get_pipeline_data(company)
        print("====COMPANIES RESULTS====")
        print(data)

    drugs = ["ACETAMINOPHEN"]
    for drug in drugs:
        results = await resolve_targets_for_drug(drug, 0)
        print("====DRUGS RESULTS====")
        print(results)

if __name__ == "__main__":
    asyncio.run(main())