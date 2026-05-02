"""
Political Risk Model
Loads the trained Gradient Boosting model and classifies constituency risk
using historical TCPD features + live Gemini sentiment/events.

Falls back to domain-expert heuristics if model is unavailable.
"""

import os
from typing import Dict, List
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "political_risk_model.pkl")

# ─── Load Trained Model ──────────────────────────────────────────────────────
_model = None
_features = None

try:
    artifact = joblib.load(MODEL_PATH)
    _model    = artifact["model"]
    _features = artifact["features"]
    print(f"[political_model] Loaded ML model (strategy: {artifact.get('label_strategy')})")
except Exception as e:
    print(f"[political_model] Warning: ML model not loaded ({e}). Using heuristic fallback.")

# Event Severity Map
EVENT_SEVERITY_MAP = {
    "scandal":           1.0,
    "protest":           0.6,
    "alliance":          0.3,
    "campaign activity": 0.1,
    "general":           0.0,
}

def calculate_event_severity(event_tags: List[str]) -> float:
    """Aggregates event tags into a single severity score."""
    if not event_tags:
        return 0.0
    total = sum(EVENT_SEVERITY_MAP.get(tag, 0.0) for tag in event_tags)
    return round(total, 3)


def classify_political_risk(
    rolling_avg_margin: float,
    turnout_std: float,
    last_vote_share: float,
    n_elections: int,
    sentiment_score: float,
    severity: float,
) -> str:
    """
    Uses the trained GBM to predict risk level. Falls back to heuristics.
    """
    if _model and _features:
        try:
            # Feature names must match exactly what the model was trained with
            df = pd.DataFrame([{
                "rolling_avg_margin": rolling_avg_margin,
                "turnout_std":        turnout_std,
                "last_vote_share":    last_vote_share,
                "n_elections":        n_elections,
                "live_sentiment":     sentiment_score,
                "live_severity":      severity,
            }])
            return _model.predict(df)[0]
        except Exception as e:
            print(f"[political_model] Prediction error: {e}. Falling back.")

    # Heuristic Fallback (domain-expert thresholds) 
    if rolling_avg_margin < 3.0:
        return "CRITICAL"
    elif rolling_avg_margin < 7.0:
        return "HIGH"
    elif rolling_avg_margin < 15.0:
        return "MODERATE"
    else:
        return "LOW"


def analyze_political_signal(
    gemini_output: Dict,
    historical: Dict,
) -> Dict:
    """
    Main entry point: combines Gemini sentiment output with TCPD historical 
    baseline to produce the enriched political risk feature set.
    """
    sentiment_score = gemini_output.get("sentiment_score", 0.0)
    event_tags      = gemini_output.get("event_tags", [])
    severity        = calculate_event_severity(event_tags)

    # Historical features from tcpd_node
    rolling_avg_margin = historical.get("rolling_avg_margin", 5.0)
    turnout_std        = historical.get("turnout_std", 0.0)
    last_vote_share    = historical.get("last_vote_share", 50.0)
    n_elections        = historical.get("n_elections", 1)

    risk_level = classify_political_risk(
        rolling_avg_margin, turnout_std, last_vote_share,
        n_elections, sentiment_score, severity
    )

    return {
        "constituency":      gemini_output.get("constituency"),
        "state":             gemini_output.get("state"),
        "date":              gemini_output.get("date"),
        "sentiment_score":   sentiment_score,
        "event_severity":    severity,
        "rolling_avg_margin": rolling_avg_margin,
        "turnout_std":       turnout_std,
        "last_vote_share":   last_vote_share,
        "ml_risk_level":     risk_level,
        "event_tags":        event_tags,
    }


if __name__ == "__main__":
    sample_gemini = {
        "constituency":    "Thiruvananthapuram",
        "state":           "Kerala",
        "date":            "2026-04-30",
        "sentiment_score": -0.4,
        "event_tags":      ["scandal", "protest"],
    }
    sample_history = {
        "rolling_avg_margin": 4.5,
        "turnout_std":        3.2,
        "last_vote_share":    48.0,
        "n_elections":        5,
    }
    result = analyze_political_signal(sample_gemini, sample_history)
    print("====== Political Model Output ======")
    for k, v in result.items():
        print(f"  {k}: {v}")
