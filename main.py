import os
import asyncio
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Fix imports path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baml_client import b

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def save_to_json(data: list, filename: str):
    """
    Saves the list of Pydantic models (or dicts) to a JSON file.
    Ensures each item has a source_url field (list or string).
    """
    # Normalize pydantic models -> dicts
    json_ready_data = []
    for item in data:
        try:
            # if item is a pydantic model
            obj = item.model_dump()
        except Exception:
            # if item is already a dict
            obj = dict(item)
        json_ready_data.append(obj)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_ready_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved data to {filename}")


def extract_grounding_urls_from_response(genai_response) -> list:
    """
    Extract grounding chunk URIs from the model response.

    The response format can vary depending on SDK/proto. This function attempts
    to handle common shapes similar to the example you provided:
    response.candidates -> each candidate has groundingMetadata -> groundingChunks -> [{'web': {'uri': '...', ...}}, ...]
    """
    urls = []
    try:
        # Many SDKs expose candidates as an iterable attribute
        candidates = getattr(genai_response, "candidates", None) or genai_response.get("candidates", None) or []
    except Exception:
        candidates = []

    # If candidates might be a list of dict-like objects
    if isinstance(candidates, dict):
        # unlikely but handle gracefully
        candidates = [candidates]

    for cand in candidates:
        # support both object attribute and dict access
        gm = None
        try:
            gm = getattr(cand, "groundingMetadata", None)
        except Exception:
            pass
        if gm is None:
            # try dict style
            try:
                gm = cand.get("groundingMetadata", None)
            except Exception:
                gm = None

        if not gm:
            continue

        # groundingChunks may be present
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

        # iterate chunks and extract web.uri if present
        for chunk in grounding_chunks:
            # chunk might be an object with .web or dict with 'web'
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

            # web_field may be object or dict containing 'uri'
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

    # deduplicate while preserving order
    seen = set()
    uniq_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq_urls.append(u)

    return uniq_urls


async def get_pipeline_data(company: str):
    print(f"\n🔎 Agent is researching {company}...")


    # 1. Search Step
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
* **URL Accuracy:** **Do not construct or guess URLs** (e.g., do not invent `company.com/pipeline/drug-name`). Use the *actual* URL returned by your search tool. If a specific deep-link is not available, use the main Pipeline page URL where the data was verified.
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
    "source_url": "string (The EXACT URL found in your search results. Do not hallucinate links.)"
  }}
]


### CRITICAL INSTRUCTION
Do not provide an intro or outro. Output **only** the raw JSON list. Begin the search now for {company}.
    """


    # Generate content and preserve the full response for grounding extraction
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=agent_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearchRetrieval)],
        )
    )

    # The textual search report (what you were using before)
    search_report = getattr(response, "text", None) or response.get("text", None) or ""
    if not search_report:
        print(f"⚠️ Agent found nothing for {company}")
        return

    # Try to extract grounding URLs from the full response object
    grounding_urls = extract_grounding_urls_from_response(response)
    if grounding_urls:
        print(f"🔗 Found grounding URLs ({len(grounding_urls)}):")
        for u in grounding_urls:
            print("  -", u)
    else:
        print("⚠️ No grounding URLs found in model response metadata. Proceeding without grounding URLs.")

    # 2. Structuring Step (BAML)
    print(f"🧠 Structuring data for {company}...")

    pipeline_data = None
    try:
        # Try passing grounding URLs into ExtractPipeline (if supported)
        pipeline_data = b.ExtractPipeline(
            search_report=search_report,
            company=company,
            grounding_urls=grounding_urls  # many baml clients may accept extra kwargs
        )
        print(f"✅ SUCCESS: b.ExtractPipeline returned {len(pipeline_data)} records (with grounding_urls param).")
    except TypeError:
        # b.ExtractPipeline doesn't accept grounding_urls — fallback to legacy call
        try:
            pipeline_data = b.ExtractPipeline(
                search_report=search_report,
                company=company
            )
            print(f"✅ SUCCESS: b.ExtractPipeline returned {len(pipeline_data)} records (without grounding_urls param).")
        except Exception as e:
            print(f"❌ BAML Error when calling ExtractPipeline: {e}")
            return
    except Exception as e:
        print(f"❌ BAML Error: {e}")
        return

    # Ensure pipeline_data is a list-like collection
    if not pipeline_data:
        print("⚠️ No pipeline data returned by ExtractPipeline.")
        return

    # 3. Attach grounding URLs to each record's source_url if missing or merge if present
    enriched = []
    for item in pipeline_data:
        # Convert pydantic model to dict if necessary but keep original for save
        try:
            item_dict = item.model_dump()
        except Exception:
            try:
                item_dict = dict(item)
            except Exception:
                item_dict = {}

        # Normalize existing source_url into a list
        existing = item_dict.get("source_url", None)
        merged_urls = []

        if existing:
            if isinstance(existing, list):
                merged_urls.extend(existing)
            elif isinstance(existing, str):
                merged_urls.append(existing)

        # Add grounding URLs, avoiding duplicates
        for u in grounding_urls:
            if u not in merged_urls:
                merged_urls.append(u)

        # If merged_urls is empty, keep source_url as empty list (could be useful)
        final_source = merged_urls if merged_urls else []

        # Attempt to set the source_url on the original item if it's a mutable object, otherwise keep dict
        try:
            # If pydantic model, set attribute if possible
            if hasattr(item, "source_url"):
                try:
                    setattr(item, "source_url", final_source)
                    enriched.append(item)
                    continue
                except Exception:
                    pass
            # If item supports model_dump and reconstruction, create a copy dict
            item_dict["source_url"] = final_source
            enriched.append(item_dict)
        except Exception:
            # fallback — produce a minimal record
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

    # 4. Save the enriched results to JSON
    try:
        safe_name = company.replace(" ", "_").lower()
        filename = f"{safe_name}_pipeline.json"
        save_to_json(enriched, filename)
    except Exception as e:
        print(f"❌ Error saving results: {e}")


async def main():
    # You can add as many companies as you want here
    companies = ["biomarin"]

    for company in companies:
        await get_pipeline_data(company)


if __name__ == "__main__":
    asyncio.run(main())
