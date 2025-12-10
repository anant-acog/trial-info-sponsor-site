// Data Models based on FR-02
class PipelineEntry {
  sponsor_name string @description("The official name of the company (e.g. 'Vertex Pharmaceuticals').")
  
  candidate_name string @description("The specific code name (e.g. 'VX-548') or brand name of the drug. Do not include dosage info here.")
  
  disease_indication string[] @description("List of specific diseases or conditions the drug is targeting (e.g. 'Cystic Fibrosis', 'Acute Pain'). Avoid broad terms like 'Rare Diseases' if specific ones are listed.")
  
  development_phase string @description("The current clinical phase. Normalize to: 'Preclinical', 'Phase I', 'Phase II', 'Phase III', 'Filed', or 'Approved'.")
  
  moa string? @description("Mechanism of Action. How the drug works (e.g. 'NaV1.8 Inhibitor', 'mRNA Vaccine'). Null if not found.")
  
  route_of_admin string? @description("Route of administration (e.g. 'Oral', 'Intravenous', 'Subcutaneous'). Null if not found.")
  
  status string? @description("Current status of the program (e.g. 'Active', 'Discontinued', 'Hold'). Default to 'Active' if implied.")
  
  source_url string[]? @description("The specific URL where this data point was found.")
}


class TargetAsset {
  intervention_name string @description("The drug name, code, or molecule (e.g. 'AMG 510', 'Sotorasib'). If preclinical and unnamed, use a descriptive term like 'KRAS G12C Inhibitor'.")

  trial_ids string[] @description("List of registry IDs if available (e.g. 'NCT04380753', 'EudraCT-2021...'). Leave empty if Preclinical.")

  disease_indication string[] @description("Specific indications mentioned in context of this target (e.g. 'NSCLC', 'Solid Tumors').")

  development_phase string @description("Current phase. Normalize to: 'Preclinical', 'Phase I', 'Phase II', 'Phase III', 'Filed', 'Approved'. Use 'Observational' for non-interventional studies.")

  status string @description("Current status. Use 'Active', 'Recruiting', 'Completed', 'Discontinued', or 'Provisional' if the link to the target is ambiguous.")

  source_url string[]? @description("URLs specific to this asset/trial finding.")
}

class OrganizationEntry {
  organization_name string @description("The official name of the sponsor/company (e.g. 'Amgen', 'Mirati Therapeutics').")
  
  assets TargetAsset[] @description("List of all assets/interventions this organization is developing for the specified target.")
}


// The extraction function
function ExtractPipeline(search_report: string, company: string, grounding_urls: string[]) -> PipelineEntry[] {
  client CustomGemini
  prompt #"
    Task: Extract structured data from the report below.
    
    Context:
    Company: {{ company }}
    
    Report:
    {{ search_report }}
    
    Instructions:
    1. Extract every drug candidate mentioned.
    2. Normalize phases to: Preclinical, Phase I, Phase II, Phase III, Approved.
    3. Use the EXACT JSON keys defined below.
    
    JSON Format Example:
    [
      {
        "sponsor_name": "Vertex",
        "candidate_name": "VX-548",
        "disease_indication": ["Acute Pain"],
        "development_phase": "Phase III",
        "moa": "NaV1.8 Inhibitor",
        "source_url": [{{ grounding_urls }}]
      }
    ]

    Output JSON:
  "#
}

function ExtractTargetLandscape(search_report: string, target_name: string, grounding_urls: string[]) -> OrganizationEntry[] {
  client CustomGemini
  prompt #"
    ### ROLE
    You are a Competitive Intelligence Analyst.

    ### OBJECTIVE
    Analyze the search report below to identify the competitive landscape for the biological target: {{ target_name }}.
    Group all findings by the Organization (Sponsor) doing the work.

    ### INPUT DATA
    Target: {{ target_name }}
    Search Report:
    {{ search_report }}

    ### EXTRACTION RULES
    1. **Scope:** Include Interventional trials (NCT/EudraCT), Observational studies, and Preclinical programs.
    2. **Grouping:** Create ONE entry per Organization. If a company has multiple drugs/trials for {{ target_name }}, list them all under that company's 'assets' list.
    3. **Ambiguity:** If a mention is vague (e.g. "Company X is exploring {{ target_name }} targets" without a specific drug name), capture it but set status to 'Provisional'.
    4. **Trial IDs:** Aggressively extract NCT numbers, EudraCT numbers, or ISRCTN numbers.
    5. **Normalization:**
       - Map 'Discovery' / 'Lead Op' -> 'Preclinical'
       - Map 'Marketed' -> 'Approved'

    ### JSON OUTPUT FORMAT
    Return a list of OrganizationEntry objects.
    [
      {
        "organization_name": "Amgen",
        "assets": [
          {
            "intervention_name": "Sotorasib",
            "trial_ids": ["NCT03600883"],
            "disease_indication": ["NSCLC"],
            "development_phase": "Approved",
            "status": "Active",
            "source_url": [{{ grounding_urls }}]
          }
        ]
      }
    ]
  "#
}