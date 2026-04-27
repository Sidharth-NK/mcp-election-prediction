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

## Phase 2: Wiki API Agent (Static Data)
Status: Assigned to Co-Developer

*   **[ ] Task:** Build an agent to fetch candidate history, past margins, and demographic baselines using Wiki APIs.

## Phase 3: MCP Orchestrator
Status: Completed (v3.1.0)

*   **[x] Task:** Build `server.py` and wrap News Agent tools (Completed).
*   **[x] Task:** Build `political_model.py` as a heuristic risk classifier (Completed).
*   **[x] Task:** Test integrated pipeline in an MCP-compliant client (`test_client.py`) (Completed).
*   **[ ] Upgrade:** Replace heuristic rules in `political_model.py` with a trained ML model (logistic regression or gradient boosting) once Wiki historical data is available from co-developer.
*   **[ ] Task:** Add co-developer's Wiki Agent tools to the server once ready.

## Phase 4: TFT Forecasting Engine
Status: Pending (requires Wiki data + Political Model upgrade)

*   **[ ] Task:** Design feature fusion layer — align static (Wiki) + dynamic (Gemini + Political Model) data into a time-series format.
*   **[ ] Task:** Train Temporal Fusion Transformer on historical election + sentiment features.
*   **[ ] Task:** Output multi-horizon predictions (win probability, vote share trends).
