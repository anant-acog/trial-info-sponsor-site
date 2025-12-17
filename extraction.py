import os
import asyncio
import sys
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
import hashlib
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baml_client import b

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved → {filename}")

def save_to_excel(drugs_list, input_file_path, output_filename):
    """
    Reads existing Excel file, appends new drug records, and saves to a new file.
    Maps fields to existing columns:
    - Column B: disease_indication
    - Column C: drug_name
    - Column E: development_status
    - Column I: sponsor
    - Column J: target
    """
    try:
        # Read existing Excel file
        df_existing = pd.read_excel(input_file_path, sheet_name='Pipeline_Data')
        print(f"Existing file loaded: {len(df_existing)} rows")
    except Exception as e:
        print(f"Error reading existing Excel: {e}")
        return

    rows_to_add = []
    
    for drug in drugs_list:
        target = drug.get("target", "")
        drug_name = drug.get("drug_name", "")
        sponsor = drug.get("sponsor", "")
        dev_status = drug.get("development_status", "")
        indications = drug.get("disease_indication", [])
        
        # Create a row for each indication
        if not indications:
            indications = [""]
        
        for indication in indications:
            row = {
                df_existing.columns[1]: indication,           # Column B (index 1)
                df_existing.columns[2]: drug_name,            # Column C (index 2)
                df_existing.columns[4]: dev_status,           # Column E (index 4)
                df_existing.columns[8]: sponsor,              # Column I (index 8)
                df_existing.columns[9]: target                # Column J (index 9)
            }
            rows_to_add.append(row)
    
    if not rows_to_add:
        print("No data to add.")
        return
    
    # Create DataFrame from new rows
    df_new = pd.DataFrame(rows_to_add)
    
    # Append to existing DataFrame
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Save to new Excel file
    try:
        df_final.to_excel(output_filename, sheet_name='Pipeline_Data', index=False)
        print(f"✅ Saved Excel → {output_filename} ({len(df_new)} new rows added)")
    except Exception as e:
        print(f"Error saving Excel: {e}")

def get_unique_column_excel(file_path):
    """
    Reads column C from the 'Pipeline_Data' sheet of the specified Excel file
    and returns a list of unique values.
    
    Args:
        file_path (str): Path to the Excel file
    
    Returns:
        list: List of unique values from column C
    """
    df = pd.read_excel(file_path, sheet_name='Pipeline_Data', usecols='C')
    
    unique_values = df.iloc[:, 0].dropna().unique().tolist()
    
    return unique_values

def get_drug_signature(drug):
    """Create unique signature for deduplication."""
    key_parts = [
        drug.get('drug_name', '').lower().strip(),
        drug.get('target', '').lower().strip(),
        drug.get('development_status', '').lower().strip()
    ]
    return hashlib.md5('||'.join(key_parts).encode()).hexdigest()

def extract_grounding_urls(response):
    """Safe extraction of grounding URLs from response."""
    grounding_urls = []
    
    try:
        candidates = getattr(response, "candidates", [])
        if candidates:
            candidate = candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            if grounding_metadata is None and hasattr(candidate, "get"):
                grounding_metadata = candidate.get("groundingMetadata")
            
            if grounding_metadata:
                # 1. Extract from search_entry_point (Vertex AI style)
                search_entry_point = getattr(grounding_metadata, 'search_entry_point', None)
                if search_entry_point and hasattr(search_entry_point, 'rendered_content'):
                    try:
                        import re
                        from bs4 import BeautifulSoup
                        html_content = search_entry_point.rendered_content
                        soup = BeautifulSoup(html_content, 'html.parser')
                        chips = soup.find_all('a', class_='chip')
                        for chip in chips:
                            href = chip.get('href', '')
                            if href and 'vertexaisearch.cloud.google.com/grounding-api-redirect/' in href:
                                grounding_urls.append(href)
                    except ImportError:
                        pass  # Skip if BeautifulSoup not available
                
                # 2. Traditional grounding_chunks
                grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None)
                if grounding_chunks is None and hasattr(grounding_metadata, "get"):
                    grounding_chunks = grounding_metadata.get("groundingChunks")
                
                if grounding_chunks:
                    for chunk in grounding_chunks:
                        web = getattr(chunk, "web", None)
                        if web:
                            uri = getattr(web, "uri", None)
                            if uri:
                                grounding_urls.append(uri)
    except Exception:
        pass
    
    return list(dict.fromkeys(grounding_urls))  # Remove duplicates preserving order

async def get_missing_drugs_for_target(
    target_name: str,
    known_drugs: list[str],
    run_id: int = 0
):
    print(f"\n🔍 Run {run_id + 1}: Searching missing drugs for target: {target_name}")

    agent_prompt = f"""
### PERSONA

You are a **Senior Drug Discovery Intelligence Analyst** specializing in **target-centric therapeutic landscape mapping**, including **active and discontinued clinical programs**.

---

### OBJECTIVE

Identify **ALL therapeutic drugs and clinical trial programs** that **directly target `{target_name}`**, including:

* **Active programs**
* **Completed programs**
* **Discontinued / Terminated / Suspended programs**

across **Preclinical, Clinical, Filed, and Approved** stages, **excluding** any drugs listed in the **Known Drugs List**.

Your mandate is **maximum completeness with zero hallucination**. Every entry must be explicitly supported by a credible public source.

Ensure to include **ALL companies**, regardless of size (including large, small, micro, and all types of biotechs/biopharmas), that have programs directly targeting `{target_name}`. NO exclusions based on company type or size. Expand search to comprehensively capture all entities involved, from large pharmas to emerging startups, academic spinouts, and all relevant industry participants.

---

### INPUTS

* **Target:** `{target_name}`
* **Known Drugs (EXCLUDE ALL):**

```json
{json.dumps(known_drugs, indent=2)}
```

---

### SCOPE OF INCLUSION

Include drugs that:

* Appear in **any lifecycle state**, including:

  * Preclinical
  * Phase I
  * Phase II
  * Phase III
  * Filed
  * Approved
  * **Discontinued / Terminated / Suspended / Stopped**

Include **discontinued programs ONLY if**:

* `{target_name}` is clearly stated as the mechanistic target
* The discontinuation is documented in a reliable public source

---

### SEARCH & DISCOVERY STRATEGY (MANDATORY)

Perform **iterative, exhaustive searches** across all categories below.

#### 1. Clinical Trial Registries

* ClinicalTrials.gov
* EU Clinical Trials Register
* WHO ICTRP

Search using:

* `{target_name}`
* `{target_name} drugs`
* Target synonyms / gene symbols / protein aliases
* `{target_name} inhibitor`
* `{target_name} antibody`
* `{target_name} agonist / antagonist`
* `{target_name} degrader / PROTAC`
* Filters for **Completed**, **Terminated**, **Suspended**, **Withdrawn**

#### 2. Regulatory & Approval Databases

* FDA (Drugs@FDA, labels, approval letters)
* EMA (EPARs)
* PMDA

#### 3. Company & Pipeline Intelligence. 

* Company pipeline pages
* R&D day presentations
* Press releases announcing **termination, reprioritization, or pipeline removals**
* SEC filings (10-K, 20-F)
## Strict Rule (Mandatory)
Ensure to include all clinical stage as well as preclinical/research stage biopharmas/biotechs that have drugs targeting `{target_name}` that are mentioned in their websites/latest investor presentations/R&D presentations etc.

**Additionally, expand your intelligence gathering to all companies of any size, including small, micro, and emerging biotechs/biopharmas—no listing is too minor—if public confirmation of a `{target_name}`-targeting drug exists.**

#### 4. Scientific & Review Literature

* Target-focused review papers
* Tables of discontinued programs
* Mechanism-of-action summaries
* Clinical trial summaries

#### 5. News & Industry Coverage

* Biopharma news outlets like BioSpace, FierceBiotech, FiercePharma, Endpoints.news etc.
* Conference abstracts (AACR, ASCO, ESMO etc.) of a company that has programs targeting `{target_name}`.
* Analyst reports referencing program discontinuation

---


General Rules:

* Use **Discontinued** for programs that are terminated, suspended, withdrawn, or stopped
* Do **not** infer discontinuation from absence in pipelines
* The highest **confirmed phase reached before discontinuation** must be used

---

### VALIDATION & EVIDENCE RULES (NON-NEGOTIABLE)

* Every drug MUST have:

  * Explicit confirmation of `{target_name}` as the target
  * A **credible, public, grounding source URL**
* If **any required field cannot be confirmed**, exclude the drug
* Do **not infer sponsors, phases, indications, or status**
* Do **not include preclinical concepts without named compounds**

---

### DEDUPLICATION & EXCLUSION RULES

* Exclude **all drugs present in the Known Drugs list**, including:

  * Synonyms
  * Alternate spellings
  * Development codes
* If identity overlap is uncertain, **include the entry**
* One JSON object per unique drug

---

### OUTPUT REQUIREMENTS (STRICT)

* Output **ONLY valid JSON**
* No commentary, no markdown, no explanations
* If **no qualifying drugs are found**, return:

```json
[]
```

---

### OUTPUT SCHEMA

```json
[
  {{
    "target": "{target_name}",
    "drug_name": "string",
    "sponsor": "string or null",
    "disease_indication": ["string"],
    "development_status": "Preclinical | Phase I | Phase II | Phase III | Filed | Approved | Discontinued",
    "source_url": "string"
  }}
]
```
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=agent_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
    )

    search_report = getattr(response, "text", "") or ""
    if not search_report:
        print(f"Run {run_id + 1}: No data found.")
        return [], []

    # Extract grounding URLs
    grounding_urls = extract_grounding_urls(response)
    print(f"Run {run_id + 1}: Found {len(grounding_urls)} grounding URLs")

    # BAML extraction
    missing_drugs = b.ExtractMissingDrugsByTarget(
        search_report=search_report,
        target_name=target_name,
        known_drugs=known_drugs
    )
    
    result = []
    for item in missing_drugs:
        try:
            item_dict = item.model_dump()
        except Exception:
            item_dict = dict(item)
        
        # Attach grounding URLs to each drug
        item_dict["source_urls"] = grounding_urls
        item_dict["run_id"] = run_id + 1
        result.append(item_dict)
    
    return result, grounding_urls

async def aggregate_runs(target_name: str, known_drugs: list[str], num_runs: int = 4):
    """Run multiple executions and aggregate unique results."""
    all_results = []
    all_urls = set()
    
    print(f"\n🚀 Starting {num_runs} runs for {target_name}")
    
    for run_id in range(num_runs):
        run_results, run_urls = await get_missing_drugs_for_target(
            target_name, known_drugs, run_id
        )
        all_results.extend(run_results)
        all_urls.update(run_urls)
        print(f"Run {run_id + 1} complete: {len(run_results)} drugs found")
    
    # Deduplicate using drug signature
    seen_signatures = set()
    unique_results = []
    
    for drug in all_results:
        signature = get_drug_signature(drug)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            # Clean up run_id for final output
            drug_clean = drug.copy()
            drug_clean.pop("run_id", None)
            unique_results.append(drug_clean)
    
    print(f"\n✅ Aggregation complete: {len(unique_results)} unique drugs from {len(all_results)} total records")
    print(f"📊 Total unique URLs: {len(all_urls)}")
    
    return unique_results, list(all_urls)



async def main():
    target_name = "GPR75"
    input_file_path = "GPR75-Target-Pipeline.xlsx"
    output_file_path = f"GPR75-Target-Pipeline-Updated.xlsx"
    
    # Get known drugs
    known_drugs = get_unique_column_excel(input_file_path)

    # Get new data
    unique_drugs, all_urls = await aggregate_runs(target_name, known_drugs, num_runs=4)
    
    # 1. Save JSON
    json_filename = f"missing_drugs_{target_name.lower().replace(' ', '_')}_aggregated.json"
    final_result = {
        "target": target_name,
        "total_runs": 4,
        "unique_drugs": unique_drugs,
        "all_grounding_urls": all_urls
    }
    save_to_json(final_result, json_filename)
    
    # 2. Append to existing Excel and save as new file
    save_to_excel(unique_drugs, input_file_path, output_file_path)

if __name__ == "__main__":
    asyncio.run(main())
