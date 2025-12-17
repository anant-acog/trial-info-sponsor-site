class MissingDrugEntry {
  target string @description("The target protein or molecule")
  drug_name string @description("The drug name")
  sponsor string? @description("The sponsor of the drug")
  disease_indication string[] @description("The disease indications")
  development_status string @description("The development status")
}

function ExtractMissingDrugsByTarget(
  search_report: string,
  target_name: string,
  known_drugs: string[],
) -> MissingDrugEntry[] {
  client CustomGemini
  prompt #"
    ### ROLE
    You are a Drug Discovery Competitive Intelligence Analyst.

    ### OBJECTIVE
    Analyze the search report below to identify ALL drugs that target the biological target: {{ target_name }},
    which are NOT present in the provided list of known drugs.
    Capture assets across Preclinical, Clinical, Filed, and Approved stages.

    ### INPUT DATA
    Target: {{ target_name }}

    Known Drugs (Exclude these):
    {{ known_drugs }}

    Search Report:
    {{ search_report }}

    ### EXTRACTION RULES
    1. **Exclusion Logic:** Do NOT include any drug whose name matches or is a clear synonym of a drug in the known drugs list.
    2. **Scope:** Include Preclinical programs, Interventional clinical trials, and Approved/Filed drugs.
    3. **Drug Identity:** Extract explicit drug names or codes only (e.g., "PF-07081532"). If no drug name is given, do NOT include the entry.
    4. **Disease Mapping:** Capture all disease indications explicitly associated with each drug.
    5. **Sponsor Attribution:** Assign the primary developing organization when available; otherwise set to null.
    6. **Normalization:**
       - Map 'Discovery', 'Lead Optimization', 'Research' → 'Preclinical'
       - Map 'Phase 1', 'Phase I/II' → 'Phase I'
       - Map 'Phase 2', 'Phase IIa/b' → 'Phase II'
       - Map 'Phase 3' → 'Phase III'
       - Map 'Marketed' → 'Approved'
    7. **Status Handling:**
       - Default status to 'Active'
       - If explicitly stated as stopped or terminated, set status to 'Discontinued'

    ### JSON OUTPUT FORMAT
    Return a list of MissingDrugEntry objects.

    [
      {
        "target": "{{ target_name }}",
        "drug_name": "string",
        "sponsor": "string or null",
        "disease_indication": ["string"],
        "development_status": "Preclinical | Phase I | Phase II | Phase III | Filed | Approved",
        "status": "Active | Discontinued | Provisional",
      }
    ]
  "#
}

