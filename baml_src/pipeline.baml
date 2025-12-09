// Data Models based on FR-02
class PipelineEntry {
  sponsor_name string @description("The official name of the company (e.g. 'Vertex Pharmaceuticals').")
  
  candidate_name string @description("The specific code name (e.g. 'VX-548') or brand name of the drug. Do not include dosage info here.")
  
  disease_indication string[] @description("List of specific diseases or conditions the drug is targeting (e.g. 'Cystic Fibrosis', 'Acute Pain'). Avoid broad terms like 'Rare Diseases' if specific ones are listed.")
  
  development_phase string @description("The current clinical phase. Normalize to: 'Preclinical', 'Phase I', 'Phase II', 'Phase III', 'Filed', or 'Approved'.")
  
  moa string? @description("Mechanism of Action. How the drug works (e.g. 'NaV1.8 Inhibitor', 'mRNA Vaccine'). Null if not found.")
  
  route_of_admin string? @description("Route of administration (e.g. 'Oral', 'Intravenous', 'Subcutaneous'). Null if not found.")
  
  status string? @description("Current status of the program (e.g. 'Active', 'Discontinued', 'Hold'). Default to 'Active' if implied.")
  
  source_url string? @description("The specific URL where this data point was found.")
}


// The extraction function
function ExtractPipeline(search_report: string, company: string) -> PipelineEntry[] {
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
        "source_url": "https://..."
      }
    ]

    Output JSON:
  "#
}