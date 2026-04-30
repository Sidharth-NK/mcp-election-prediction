# MCP based Agentic Election Forecaster

A multi-agentic machine learning pipeline designed to predict Indian State Assembly elections at the constituency level.

This project goes beyond simple polling or baseline historical statistics by utilizing a **Data Fusion Architecture**. It automatically merges decades of historical election data with statically mapped demographics and live, AI-driven sentiment analysis to classify the political risk and predict the outcome of high-volatility constituencies.

##  Architecture Stack

1. **The Data Ingestion Layer (`agents/`)**: Specialized tools that gather distinct domains of knowledge.
   - `tcpd_node.py`: Computes historical rolling average margins, voter turnout, and vote share indices from real-world Trivedi Centre for Political Data (TCPD) CSVs.
   - `demographic_node.py`: Integrates Census 2011 JSON outputs (rural, urban, SC/ST, and literacy rates) mapped perfectly to current districts.
   - `gemini_news_agent.py`: Uses Tavily to run 4 parallel internet news searches per constituency (tracking localized media) and feeds the text batches to Gemini 2.5 Flash to generate a real-time Sentiment Score. Includes API-limit batching constraints and caching.

2. **The ML Risk Expert System**: 
   - Uses a fine-tuned Gradient Boosting Classifier to read historical wave theories (like turnout spikes) combined with the AI-live sentiment scores. Outputs political strategy labels such as `SAFE HOLD` or `TOSS-UP / POSSIBLE FLIP` along with a 0-10 volatility score.

3. **Temporal Fusion Transformer (`tft_engine/`)**: 
   - A deep learning sequence-to-sequence model capable of probabilistic quantile generation. Uses PyTorch Lightning for advanced mathematical forecasting based purely on time-series historical dynamics.

##  How to Run the System

You do not need to piece things together manually. Everything is orchestrated to run smoothly.

**1. Validate the Pipeline**
Ensures no mathematical impossibilities exist (e.g., margins > 100%) and all CSVs parse perfectly.
```bash
python scripts/validate_pipeline.py
```

**2. Train the Expert Model**
Re-train the Gradient Boosting model on historical TCPD datasets using time-series shifting. 
```bash
python scripts/train_political_model.py
```
*(This saves the model weights to `agents/political_risk_model.pkl`)*

**3. Run the Final Prediction Generation**
Pulls all 800+ constituencies, triggers the live Gemini internet sweep, queries the ML model, maps the party name, and generates your final spreadsheet.
```bash
python scripts/run_predictions.py
```

##  Final Folder Structure

* `agents/` — API fetchers, data formatters, and the trained ML brain.
* `scripts/` — The main execution scripts to validate, train, and run.
* `tft_engine/` — Deep learning components for time-series evaluation.
* `predictions/` — Where your output `election_predictions_2026.csv` and JSON outputs are stored!
