"""
End-to-End Election Prediction Pipeline
========================================
Runs predictions for ALL constituencies across ALL 5 states using:
  1. TCPD Historical Data (real CSV) → rolling margins, turnout volatility
  2. Political Risk ML Model (trained GBM) → risk classification
  3. Wiki Candidate Metadata → party, alliance context
  4. Demographics (Census 2011) → static sociodemographic features
  
Note: Gemini News Agent and Polling Node are SKIPPED in this batch run
because they require live API calls (Tavily + Gemini) which have rate limits.
Instead, we set sentiment/severity to neutral (0.0) baseline — the model 
handles this gracefully as it was trained with neutral live signals.

Usage:
    python run_predictions.py
    
Output:
    predictions/election_predictions_2026.csv
    predictions/election_predictions_2026.json
"""

import os
import sys
import json
import datetime
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.tcpd_node import get_historical_baseline, get_all_constituencies, _load_all_tcpd
from agents.political_model import analyze_political_signal, classify_political_risk, calculate_event_severity
from agents.demographic_node import get_demographic_vector


# ─── Load Wiki Candidate Registry ────────────────────────────────────────────
def load_wiki_registry() -> Dict:
    """Loads candidate registry from wiki_static_meta.json."""
    wiki_path = os.path.join(os.path.dirname(__file__), "..", "wiki_static_meta.json")
    if not os.path.exists(wiki_path):
        print("[WARN] wiki_static_meta.json not found. Candidate context will be empty.")
        return {}

    with open(wiki_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Index by (state, constituency) → list of candidates
    registry = defaultdict(list)
    for c in data.get("candidates", []):
        key = (c.get("state", "").strip(), c.get("constituency", "").strip())
        if key[0] and key[1]:
            registry[key].append({
                "candidate": c.get("candidate_name", ""),
                "party": c.get("party", ""),
                "alliance": c.get("alliance", ""),
                "has_wiki_page": c.get("has_wiki_page", False),
            })
    return dict(registry)


# ─── Risk Level to Numeric ───────────────────────────────────────────────────
RISK_MAP = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}


def predict_constituency(
    state: str,
    constituency: str,
    district: str,
    wiki_registry: Dict,
) -> Optional[Dict]:
    """
    Runs the full prediction pipeline for a single constituency.
    Returns a flat prediction record.
    """
    # ─── 1. TCPD Historical Data ─────────────────────────────────────────
    history = get_historical_baseline(state, constituency)
    if history.get("status") != "success":
        return None

    # ─── 2. Political Risk Model (ML) ────────────────────────────────────
    # Use neutral live signals (no Gemini in batch mode)
    risk_level = classify_political_risk(
        rolling_avg_margin=history["rolling_avg_margin"],
        turnout_std=history["turnout_std"],
        last_vote_share=history["last_vote_share"],
        n_elections=history["n_elections"],
        sentiment_score=0.0,  # Neutral — no live news
        severity=0.0,         # Neutral — no live events
    )

    # ─── 3. Demographics ─────────────────────────────────────────────────
    demo = get_demographic_vector(state, district)
    rural_pct     = demo.get("rural_percentage", 0.0) or 0.0
    urban_pct     = demo.get("urban_percentage", 0.0) or 0.0
    literacy      = demo.get("literacy_rate", 0.0) or 0.0
    sc_st_pct     = demo.get("sc_st_percentage", 0.0) or 0.0

    # ─── 4. Wiki Candidate Info ──────────────────────────────────────────
    candidates_2026 = wiki_registry.get((state, constituency), [])
    n_candidates_2026 = len(candidates_2026)
    parties_2026 = [c["party"] for c in candidates_2026]
    alliances_2026 = list(set(c["alliance"] for c in candidates_2026 if c["alliance"]))

    # ─── 5. Compute Composite Indicators ─────────────────────────────────
    # Swing Index: how likely the seat is to change hands
    margin = history["rolling_avg_margin"]
    swing_index = round(min(10.0, 10.0 / (margin + 0.5)), 2)

    # Competitiveness Score (0-10): combines margin tightness + turnout volatility
    competitiveness = round(min(10.0, swing_index + (history["turnout_std"] / 3.0)), 2)

    # ─── 6. Predicted Outcome Inference ──────────────────────────────────
    # Based on historical patterns: if margin is large → likely retention
    # If margin is tight → toss-up
    winner_2021 = history.get("winner", "UNKNOWN")
    if margin > 15.0:
        predicted_outcome = "SAFE HOLD"
        confidence = "HIGH"
    elif margin > 7.0:
        predicted_outcome = "LIKELY HOLD"
        confidence = "MODERATE"
    elif margin > 3.0:
        predicted_outcome = "LEAN / TOSS-UP"
        confidence = "LOW"
    else:
        predicted_outcome = "TOSS-UP / POSSIBLE FLIP"
        confidence = "VERY LOW"

    return {
        # Identity
        "state": state,
        "district": district,
        "constituency": constituency,
        
        # Historical (TCPD)
        "last_election_year": history["past_election_year"],
        "winner_2021": winner_2021,
        "winning_margin_pct": history["winning_margin_percentage"],
        "rolling_avg_margin": history["rolling_avg_margin"],
        "voter_turnout_pct": history["voter_turnout_percentage"],
        "turnout_std": history["turnout_std"],
        "last_vote_share": history["last_vote_share"],
        "n_elections_tracked": history["n_elections"],
        
        # ML Risk Model
        "ml_risk_level": risk_level,
        "risk_numeric": RISK_MAP.get(risk_level, 0),
        
        # Demographics
        "rural_pct": rural_pct,
        "urban_pct": urban_pct,
        "literacy_rate": literacy,
        "sc_st_pct": sc_st_pct,
        
        # Wiki 2026
        "n_candidates_2026": n_candidates_2026,
        "parties_2026": ", ".join(parties_2026) if parties_2026 else "",
        "alliances_2026": ", ".join(alliances_2026) if alliances_2026 else "",
        
        # Computed Indicators
        "swing_index": swing_index,
        "competitiveness": competitiveness,
        
        # Prediction
        "predicted_outcome": predicted_outcome,
        "prediction_confidence": confidence,
        "prediction_basis": "Historical TCPD + ML Risk Model (neutral sentiment baseline)",
    }


def run_full_pipeline():
    """Runs predictions for ALL constituencies across all 5 states."""
    print("=" * 70)
    print("  ELECTION PREDICTION PIPELINE — 2026 State Assembly Elections")
    print(f"  Run Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load candidate registry
    print("\n[1/3] Loading Wiki Candidate Registry...")
    wiki_registry = load_wiki_registry()
    print(f"       Found {sum(len(v) for v in wiki_registry.values())} candidates across {len(wiki_registry)} constituencies")

    # Get all constituencies from TCPD
    print("[2/3] Loading TCPD Historical Data...")
    all_constituencies = get_all_constituencies()
    print(f"       Found {len(all_constituencies)} unique constituencies across all states")

    # Group by state for progress tracking
    by_state = defaultdict(list)
    for c in all_constituencies:
        by_state[c["state"]].append(c)

    print(f"\n       State Breakdown:")
    for s in sorted(by_state.keys()):
        print(f"         {s}: {len(by_state[s])} constituencies")

    # Run predictions
    print(f"\n[3/3] Running Predictions...")
    all_predictions = []
    failed = 0
    
    for state_name in sorted(by_state.keys()):
        constituencies = by_state[state_name]
        print(f"\n  ── {state_name} ({len(constituencies)} seats) ──")
        
        success_count = 0
        for c in constituencies:
            try:
                result = predict_constituency(
                    state=c["state"],
                    constituency=c["constituency"],
                    district=c.get("district", ""),
                    wiki_registry=wiki_registry,
                )
                if result:
                    all_predictions.append(result)
                    success_count += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

        print(f"      {success_count} predicted |  {len(constituencies) - success_count} skipped")

    # ─── Output Results ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total Predictions: {len(all_predictions)}")
    print(f"  Failed/Skipped:    {failed}")

    if not all_predictions:
        print("\n  No predictions generated. Check data files.")
        return

    df = pd.DataFrame(all_predictions)

    # Summary Statistics
    print(f"\n  ── Risk Distribution ──")
    risk_dist = df["ml_risk_level"].value_counts()
    for level in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        cnt = risk_dist.get(level, 0)
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {level:<10} {cnt:>5} seats ({pct:>5.1f}%)  {bar}")

    print(f"\n  ── Outcome Predictions ──")
    outcome_dist = df["predicted_outcome"].value_counts()
    for outcome, cnt in outcome_dist.items():
        pct = cnt / len(df) * 100
        print(f"    {outcome:<25} {cnt:>5} seats ({pct:>5.1f}%)")

    print(f"\n  ── State-wise Summary ──")
    for state_name in sorted(df["state"].unique()):
        state_df = df[df["state"] == state_name]
        avg_margin = state_df["rolling_avg_margin"].mean()
        critical = (state_df["ml_risk_level"] == "CRITICAL").sum()
        tossup = (state_df["predicted_outcome"].str.contains("TOSS-UP")).sum()
        print(f"    {state_name:<15} {len(state_df):>4} seats | Avg Margin: {avg_margin:>5.1f}% | Critical: {critical:>3} | Toss-ups: {tossup:>3}")

    # ─── Save CSV ────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "election_predictions_2026.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n   CSV saved: {csv_path}")

    # ─── Save JSON (top 20 most competitive) ─────────────────────────────────
    top_competitive = df.nlargest(20, "competitiveness")
    json_path = os.path.join(out_dir, "top_competitive_seats.json")
    top_competitive.to_json(json_path, orient="records", indent=2)
    print(f"   Top 20 competitive seats: {json_path}")

    # ─── Print Top 10 Most Competitive ────────────────────────────────────
    print(f"\n  ── TOP 10 MOST COMPETITIVE SEATS ──")
    print(f"  {'Rank':<5} {'State':<15} {'Constituency':<25} {'Margin%':<10} {'Risk':<10} {'Outcome'}")
    print(f"  {'-'*85}")
    for i, (_, row) in enumerate(top_competitive.head(10).iterrows(), 1):
        print(f"  {i:<5} {row['state']:<15} {row['constituency']:<25} {row['rolling_avg_margin']:<10.1f} {row['ml_risk_level']:<10} {row['predicted_outcome']}")

    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete. {len(all_predictions)} constituencies predicted.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_full_pipeline()
