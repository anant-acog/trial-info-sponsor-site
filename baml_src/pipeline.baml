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

class DrugTargetCore {
  target_name string @description("Explicit molecular target name (e.g. 'Epidermal Growth Factor Receptor')")
  
  gene_symbol string? @description("HGNC symbol. Use comma-separated string for complexes (e.g. 'CD3D, CD3E'). Null if non-protein.")
  
  target_type "Single Protein" | "Protein Complex" | "Protein Family" | "Nucleic Acid" | "Other" @description("Classification of the target")
  
  moa string @description("Mechanism of Action (e.g. 'Inhibitor', 'Antagonist', 'Agonist')")
  
  approval_status "Approved" | "Not Approved" @description("Regulatory status specifically for the input disease")
}

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

function ExtractTargetCoresByDrugAndDisease(
  search_report: string,
  drug_name: string,
  disease_name: string
) -> DrugTargetCore[] {

  client CustomGemini

  prompt #"
    Task: Extract structured data from the report below.

    Context:
    Drug Name: {{ drug_name }}
    Disease/Indication: {{ disease_name }}
    
    Report:
    {{ search_report }}

    Instructions:
    1. Extract every target core mentioned.
    2. Use the EXACT JSON keys defined below.
    
    JSON Format Example:
    [
      {
        "target_name": "Epidermal Growth Factor Receptor",
        "gene_symbol": "EGFR",
        "target_type": "Single Protein",
        "moa": "Inhibitor",
        "approval_status": "Approved",
      }
    ]

    Output JSON:
"#
}
