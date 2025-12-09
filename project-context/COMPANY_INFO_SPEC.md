# 📑 Specification Document: Biopharma Pipeline Scraper


## 1. Introduction and Goals

### 1.1 Problem Statement
Manually tracking the drug development pipelines of multiple biopharma and drug companies is an **inefficient and time-consuming process**. Information is fragmented across various website structures (tables, interactive charts, text lists). A lack of a standardized, consolidated dataset hinders timely competitive intelligence and market analysis.

### 1.2 Project Goal
To develop a robust and automated **system** that, given a list of biopharma/drug company names, can navigate to the company's official "Pipeline" or "Product Candidates" page and extract key data points into a standardized, structured format.

### 1.3 Target Users
* Competitive Intelligence Analysts
* Market Research Team
* R&D Portfolio Management

---

## 2. Functional Requirements (What the System Must Do)

### 2.1 Input and Company Identification (FR-01)
The system must accept a single input: the **Company Name** (e.g., "Novartis," "Pfizer," "Vertex Pharmaceuticals").
* **Sub-Requirement:** The system must be able to reliably determine the **base URL** and the specific **Pipeline URL** for the given company name (e.g., using a lookup table or a search-and-verify mechanism).

### 2.2 Data Extraction and Handling (FR-02)
The scraper must successfully navigate to the identified Pipeline URL and extract the following data fields for **EACH unique drug/candidate** listed. 

| Field Name | Description | Data Type | Mandatory? | Notes/Examples |
| :--- | :--- | :--- | :--- | :--- |
| **Sponsor Name** | The name of the scraped sponsor website. | String | Yes | Novartis |
| **Drug/Candidate Name** | The name or code of the drug/compound. | String | Yes | Imatinib, VTX-1234 |
| **Disease/Indication** | The condition(s) the drug is targeting. | String (List) | Yes | Oncology, Type 2 Diabetes |
| **Target/Mechanism of Action (MOA)** | How the drug works (if available). | String | No | EGFR Inhibitor, Gene Therapy |
| **Development Phase** | The current clinical stage. | String | Yes | Preclinical, Phase I, Phase II, Phase III, NDA/BLA Submitted, Approved |
| **Route of Administration** | How the drug is administered (if available). | String | No | Oral, IV, Subcutaneous |
| **Trial/Start/End Date** | Any available date information. | Date/String | No | Q4 2025, 2026 H1 |
| **Source URL** | The direct URL where the data was found. | String | Yes | `https://company.com/pipeline` |
| **Status** | A general status, if provided (e.g., "Active," "On Hold," "Discontinued"). | String | No | Active |

### 2.3 Output Format (FR-03)
The final extracted data for all fields must be consolidated into a structured, machine-readable format.
* **Format:** **JSON**
* **Schema:** The output must adhere strictly to the schema defined in FR-02.

## 3\. Solution Architecture

### 3.1 High-Level Approach

The solution leverages a **"Search-then-Synthesize"** architecture, mimicking the behavior of research assistants like Perplexity or Gemini. Instead of hard-coding XPaths or CSS selectors for specific websites, the system uses a Large Language Model (LLM) equipped with search tools to dynamically locate, read, and structure information.

### 3.2 Core Components

  * **Orchestrator (Python):** The central control logic that manages the workflow, handles errors, and ensures data integrity.
  * **Discovery Module (Google GenAI Search Tool):** Utilizes the `google-generativeai` library with **Grounding** (Google Search) enabled. This allows the model to find the most current pipeline URL for a given company.
  * **Extraction Engine (Gemini 1.5 Pro/Flash):** The reasoning engine that processes the raw textual/HTML content from the discovered URL and maps it to the JSON schema defined in FR-02.
  * **Validation Layer:** A Pydantic or BAML-based layer to ensure the LLM output matches strict type definitions (e.g., ensuring "Phase 1" is normalized to "Phase I").

## 4\. Technical Workflow (Logic Flow)

The system will execute the following sequential steps for each company input.

### 4.1 Step 1: Dynamic Discovery (Search)

  * **Action:** The system queries the Google Search Grounding tool.
  * **Query Generation:** The LLM generates a query such as: `"{Company Name} official clinical trials pipeline development page."`
  * **Objective:** Retrieve the specific "Source URL" and verify it is the official domain (e.g., rejecting news articles or third-party aggregators in favor of `novartis.com/research/pipeline`).

### 4.2 Step 2: Content Acquisition

  * **Action:** Once the URL is identified, the system must "read" the page.
  * **Mechanism:**
      * **Scenario A (Static/Simple Sites):** The GenAI search tool automatically ingests the snippet content.
      * **Scenario B (Complex/Interactive Sites):** If the search snippet is insufficient, the system uses a lightweight scraping utility (e.g., `requests` or `Playwright`) to fetch the full page text and pass it as **context** to the LLM.

### 4.3 Step 3: Semantic Extraction & Structuring

  * **Action:** The LLM receives the raw text and the target Schema (FR-02).
  * **Prompting Strategy:** The prompt instructs the model to act as a Data Analyst, ignoring marketing fluff and focusing strictly on the tabular data regarding drug candidates.
  * **Normalization:** The model is instructed to normalize fuzzy data (e.g., converting "Ph2", "Phase 2", and "P-II" all to "Phase II").

### 4.4 Step 4: JSON Serialization

  * **Action:** The output is forced into a valid JSON object list.
  * **Library Recommendation:** Use `google.generativeai` with `response_mime_type="application/json"` or integrate with **Instructor/BAML** to guarantee schema adherence.

## 5\. Technical Stack & Implementation Details

### 5.1 Primary Libraries

  * **Language:** Python 3.10+
  * **AI SDK:** `google-generativeai` (specifically querying models like `gemini-1.5-flash` or `gemini-1.5-pro`).
  * **Data Validation:** `Pydantic` (for defining the schema in FR-02).

### 5.2 Implementation Snippet (Conceptual)

This snippet illustrates how to initialize the Gemini model with Google Search grounding to fulfill FR-01 and FR-02 without writing a custom crawler.

```python
import google.generativeai as genai
import os

# Configuration
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define the Tool (Grounding with Google Search)
tools = [
    {"google_search": {}} # Enables the model to browse/search live
]

# Initialize Model
model = genai.GenerativeModel(
    'models/gemini-1.5-pro-002', # Pro model recommended for complex extraction
    tools=tools
)

def extract_pipeline(company_name: str):
    prompt = f"""
    Find the official clinical trial pipeline page for {company_name}.
    Once found, extract the top 5 distinct drug candidates currently in development.
    
    Return the data strictly as a JSON list matching this schema:
    {{
        "sponsor_name": "{company_name}",
        "candidate_name": "String",
        "disease_indication": ["String", "String"],
        "development_phase": "String (e.g., Phase I, Phase II)",
        "source_url": "The URL you found"
    }}
    """
    
    # The model performs search -> reads content -> formats JSON
    response = model.generate_content(prompt)
    
    # Post-processing to handle the response object
    return response.text
```

