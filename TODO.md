# Project Roadmap & Technical Debt

This document tracks known limitations and upcoming architectural upgrades for the Election Prediction System.

## Phase 1: Gemini News Agent (Dynamic Data)
Status: Completed (v2 - Batched & Cached)

### Known Limitations / Technical Debt
*   **[ ] Local Language Blindspot:** Search queries are currently hardcoded in English. 
    *   *Fix:* Implement localized query generation (e.g., Malayalam for Kerala) to catch regional ground-reality before national media.
*   **[ ] Cache Rigidity:** The 6-hour file cache prevents multiple LLM calls but doesn't respect "breaking news".
    *   *Fix:* Implement a fast "breaking news check" bypass, or reduce cache TTL during active election weeks.
*   **[ ] Hardcoded Event Tags:** The model is forced into ['alliance', 'protest', 'scandal', 'campaign activity', 'general'].
    *   *Fix:* Make tags dynamic or expand the list mapping to the TFT Forecasting input matrix.
*   **[ ] Tavily Recency Bias:** Search indexing might miss major breaking news happening <48 hours before polling.
    *   *Fix:* Implement a Serper API (Google Search) fallback specifically triggered during the final 3 days of a campaign for real-time indexing.
*   **[ ] Model Attention Span (Lost in the Middle):** Batching 10 constituencies sends ~120 search snippets to the LLM. 
    *   *Audit:* Ensure Gemini 2.5 isn't dropping the middle constituencies. Consider reducing `BATCH_SIZE` to 5 if accuracy drops.

## Phase 2: Wiki API Agent (Candidate Identity Meta)
Status: Completed (v3.2.0)
*   **[x] Task:** Build agent to fetch live candidate metadata via Wikimedia REST API.
*   **[x] Task:** Integrate Wiki Agent into MCP Orchestrator.

## Phase 2.5: Indian Context Data Ingestion (NEW)
Status: Completed (v3.3.0)

*   **[x] Task:** Build `tcpd_agent.py` to ingest historical booth/assembly data (outcomes, turnout, margins) from ECI/TCPD CSV files.
*   **[x] Task:** Build `demographic_node.py` to ingest static constituency demographics (rural/urban, caste proxy, religious splits).
*   **[x] Task:** Build `polling_node.py` to aggregate CSDS-Lokniti and Axis My India survey tracker data.

### Known Limitations / Technical Debt
*   **[ ] Demographics Aging (Census 2011):** Current static demographics rely on 15-year-old census data, blinding the model to fast-urbanizing constituencies.
    *   *Fix:* Integrate DevDataLab SHRUG dataset (nighttime satellite lights) as a proxy for modern urbanization ratios.
*   **[ ] TCPD Hard Cutoff Risk:** A local CSV means the historical node is blind to delimitation boundary changes or by-elections.
    *   *Fix:* Implement a documented maintenance script or automated ECI scraper that updates the master CSV at the start of every election cycle.

## Phase 3: MCP Orchestrator
Status: Completed (v3.1.0)

*   **[x] Task:** Build `server.py` and wrap News Agent tools (Completed).
*   **[x] Task:** Build `political_model.py` as a heuristic risk classifier (Completed).
*   **[x] Task:** Test integrated pipeline in an MCP-compliant client (`test_client.py`) (Completed).
*   **[ ] Upgrade:** Replace heuristic rules in `political_model.py` with a trained ML model (logistic regression or gradient boosting) once Wiki historical data is available from co-developer.
*   **[ ] Task:** Add co-developer's Wiki Agent tools to the server once ready.

## Phase 4: Feature Fusion & TFT Forecasting Engine
Status: Completed (v4.0.0)

*   **[x] Task:** Design feature fusion layer — align static (Wiki/Demographics), dynamic (Regional Gemini/Polling), and historical (TCPD) data.
*   **[x] Task:** Build `dataset_builder.py` to recursively compile historical PyTorch inputs dynamically per constituency using the wiki registry.
*   **[x] Task:** Train Temporal Fusion Transformer on historical election + sentiment features via `model_training.py`.
*   **[x] Task:** Output multi-horizon predictions via quantiles.

## Phase 5: UI & Final Reporting (Up Next)
*   **[ ] Task:** Build Web Dashboard (Streamlit/FastAPI) to surface TFT quantiles, Risk Indicators, and Gemini Sentiment strings.
