import os
import sys
import json
import re
import asyncio
import hashlib
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baml_client import b

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])




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


# -----------------------------
# Gemini: Drug → Search Report
# -----------------------------

async def get_search_report_for_drug(drug_name: str, run_id: int) -> Dict:
    print(f"\nRun {run_id + 1}: Generating search report for {drug_name}")

    prompt = f"""
You are a Senior Drug Discovery Intelligence Analyst.

Objective:
Generate a STRICTLY STRUCTURED JSON report identifying ALL explicit molecular
targets (primary and off-target) of the drug "{drug_name}".

Rules:
- Include ONLY explicitly named molecular targets
- Each target MUST include evidence_urls (public sources)
- Do NOT infer or speculate
- Return JSON ONLY (no prose)

JSON schema:
{{
  "drug_name": "{drug_name}",
  "molecular_targets": [
    {{
      "target_name": "string",
      "target_type": "Primary Target | Off-target",
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
        print("⚠️ Failed to extract JSON from Gemini response")
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
        urls = target_url_map.get(key, [])

        # STRICT: no URLs → drop
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

    drugs = [
        "ACETAMINOPHEN",
        "Bexarotene"
    ]

    results = await aggregate_drugs(
        drugs=drugs,
        num_runs=3
    )

    output = {
        "drugs": drugs,
        "total_unique_targets": len(results),
        "results": results
    }
    return output

if __name__ == "__main__":
    asyncio.run(main())
