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
*   **[ ] Model Attention Span (Lost in the Middle):** Batching 10 constituencies sends ~120 search snippets to the LLM. 
    *   *Audit:* Ensure Gemini 2.5 isn't dropping the middle constituencies. Consider reducing `BATCH_SIZE` to 5 if accuracy drops.

## Phase 2: Wiki API Agent (Candidate Identity Meta)
Status: Completed (v3.2.0)
*   **[x] Task:** Build agent to fetch live candidate metadata via Wikimedia REST API.
*   **[x] Task:** Integrate Wiki Agent into MCP Orchestrator.

## Phase 2.5: Indian Context Data Ingestion (NEW)
Status: Pending

*   **[ ] Task:** Build `tcpd_agent.py` to ingest historical booth/assembly data (outcomes, turnout, margins) from ECI/TCPD CSV files.
*   **[ ] Task:** Build `demographic_node.py` to ingest static constituency demographics (rural/urban, caste proxy, religious splits).
*   **[ ] Task:** Build `polling_node.py` to aggregate CSDS-Lokniti and Axis My India survey tracker data.

## Phase 3: MCP Orchestrator
Status: Completed (v3.1.0)

*   **[x] Task:** Build `server.py` and wrap News Agent tools (Completed).
*   **[x] Task:** Build `political_model.py` as a heuristic risk classifier (Completed).
*   **[x] Task:** Test integrated pipeline in an MCP-compliant client (`test_client.py`) (Completed).
*   **[ ] Upgrade:** Replace heuristic rules in `political_model.py` with a trained ML model (logistic regression or gradient boosting) once Wiki historical data is available from co-developer.
*   **[ ] Task:** Add co-developer's Wiki Agent tools to the server once ready.

## Phase 4: Feature Fusion & TFT Forecasting Engine
Status: Pending (blocked on Phase 2.5 Indian Context data)

*   **[ ] Task:** Design feature fusion layer — align static (Wiki/Demographics), dynamic (Regional Gemini/Polling), and historical (TCPD) data.
*   **[ ] Task:** Train Temporal Fusion Transformer on historical election + sentiment features.
*   **[ ] Task:** Output multi-horizon predictions (win probability, vote share trends).
