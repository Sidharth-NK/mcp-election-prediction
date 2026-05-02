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
import asyncio
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.tcpd_node import get_historical_baseline, get_all_constituencies, _load_all_tcpd
from agents.political_model import analyze_political_signal, classify_political_risk, calculate_event_severity
from agents.demographic_node import get_demographic_vector
from agents.gemini_news_agent import batch_analyze
from agents.swing_model import apply_uniform_swing, classify_alliance, calibrate_from_polling
from agents.polling_node import analyze_polling_data

# ─── Constituency → District Map (extracted from TCPD) ────────────────────────
_DISTRICT_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "agents", "data", "constituency_district_map.json")
_DISTRICT_MAP = {}
if os.path.exists(_DISTRICT_MAP_PATH):
    with open(_DISTRICT_MAP_PATH, "r") as f:
        _DISTRICT_MAP = json.load(f)
    print(f"[district_map] Loaded {len(_DISTRICT_MAP)} constituency→district mappings.")

def resolve_district(state: str, constituency: str) -> str:
    """Looks up the district name for a constituency from the TCPD-derived map."""
    # Try exact match first (TCPD uses UPPERCASE constituency names)
    key = f"{state}|{constituency.upper()}"
    if key in _DISTRICT_MAP:
        return _DISTRICT_MAP[key]
    # Try with underscored state name (Tamil_Nadu vs Tamil Nadu)
    key2 = f"{state.replace(' ', '_')}|{constituency.upper()}"
    if key2 in _DISTRICT_MAP:
        return _DISTRICT_MAP[key2]
    return ""


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


# ─── Party Name Normalization ────────────────────────────────────────────────
# TCPD uses 'CPM', wiki scraper returns 'CPI(M)'; normalize to a common key.
PARTY_ALIASES: Dict[str, str] = {
    # LDF parties
    "CPI(M)":   "CPM",
    "CPIM":      "CPM",
    "CPI-M":     "CPM",
    "CONG(S)":   "C(S)",
    "CONG-S":    "C(S)",
    "KC-M":      "KC(M)",
    "KCM":       "KC(M)",
    "RJD":       "RJD",
    # UDF parties
    "INC(I)":   "INC",
    "MUL":       "IUML",
    # NDA parties
    "TTP":       "BJP",
    # Tamil Nadu
    "AIADMK":    "ADMK",
    "AI-ADMK":   "ADMK",
    # West Bengal / Full names from wiki
    "AITC":      "TMC",
    "TRNC":      "TMC",
    "Trinamool Congress": "TMC",
    "All India Trinamool Congress": "TMC",
    "Bharatiya Janata Party": "BJP",
    "Indian National Congress": "INC",
    "Dravida Munnetra Kazhagam": "DMK",
    "All India N.R. Congress": "AINRC",
    "Indian Secular Front": "ISF",
    "Bharatiya Gorkha Prajatantrik Morcha": "BGPM",
    "Namadhu Makkal Kazhagam": "NMK",
}


def normalize_party(party: str) -> str:
    """Canonicalize party name to match TCPD abbreviations."""
    if not party:
        return ""
    p = party.strip()
    return PARTY_ALIASES.get(p, p)


async def predict_constituency(
    state: str,
    constituency: str,
    district: str,
    wiki_registry: Dict,
    live_sentiment_score: float = 0.0,
    live_severity_score: float = 0.0,
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
    # Use LIVE signals fetched via Gemini mapping
    risk_level = classify_political_risk(
        rolling_avg_margin=history["rolling_avg_margin"],
        turnout_std=history["turnout_std"],
        last_vote_share=history["last_vote_share"],
        n_elections=history["n_elections"],
        sentiment_score=live_sentiment_score,
        severity=live_severity_score,
    )

    # ─── 3. Demographics ─────────────────────────────────────────────────
    demo = await get_demographic_vector(state, district)
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
    winner_2021 = history.get("winner", "UNKNOWN")
    top_cands = history.get("top_candidates", [])
    runner_up_2021 = top_cands[1]["party"] if len(top_cands) > 1 else "UNKNOWN_CHALLENGER"

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

    # ─── 7. Explicit Party Winner Prediction (2026) ───────────────────────
    #
    # Strategy: Use the actual 2026 candidate list as the authoritative source.
    # Find the most-credible winner based on three signals:
    #   a) Historical dominance (winner_2021 / runner_up_2021 from TCPD)
    #   b) Whether a strong alliance party is fielding a candidate in 2026
    #   c) Live Gemini sentiment — negative flips the prediction to the
    #      strongest challenger from the 2026 candidate list.
    #
    # Define alliance structures for major states
    # Note: both raw wiki names AND TCPD abbreviations included
    LDF  = {"CPM", "CPI(M)", "CPI", "NCP", "JD(S)", "KC(M)", "KCM", "LJD",
            "KEC(B)", "INL", "CPS", "C(S)", "Cong(S)", "CONG(S)", "RJD",
            "RSP", "KTP", "SP", "SJP", "NLP"}
    UDF  = {"INC", "INC(I)", "IUML", "KEC", "KEC(M)", "RMPI", "KC", "CMPC",
            "JD(U)", "ML", "KSP"}
    NDA  = {"BJP", "BDJS", "NSS", "TTP", "BDML"}
    DMK_FRONT  = {"DMK", "INC", "CPM", "CPI(M)", "CPI", "VCK", "MDMK", "MMK",
                  "KMDK", "AIFB", "MNM"}
    ADMK_FRONT = {"ADMK", "AIADMK", "PMK", "BJP", "TTP", "DMDK"}
    TMC_FRONT  = {"TMC", "AITC"}
    BJP_FRONT  = {"BJP", "AGP", "UPPL", "BOPF", "TTP"}

    # Build set of 2026 contesting parties in this constituency
    # Normalize wiki party names to TCPD canonical forms for matching
    parties_2026_set = {
        normalize_party(c["party"].strip())
        for c in candidates_2026
        if c.get("party")
    }

    # --- Determine if a 'flip' should happen ---
    # Flip is warranted when:
    #   - The seat is CRITICAL/HIGH risk (tight historically), AND
    #   - Gemini detected strongly negative sentiment (anti-incumbency)
    flip_warranted = (
        risk_level in ["CRITICAL", "HIGH"]
        and live_sentiment_score < -0.15
    )

    if not flip_warranted:
        # No flip: incumbent (2021 winner) holds if they are contesting in 2026
        if winner_2021 in parties_2026_set:
            predicted_winner_party_2026 = winner_2021
        elif parties_2026_set:
            # Winner party not contesting 2026 — use best historical from 2026 list
            # Prefer LDF/UDF/NDA/TMC parties in historical dominance order
            for p in ([winner_2021, runner_up_2021] + list(parties_2026_set)):
                if p in parties_2026_set:
                    predicted_winner_party_2026 = p
                    break
            else:
                predicted_winner_party_2026 = winner_2021
        else:
            predicted_winner_party_2026 = winner_2021
    else:
        # Flip: give the seat to the strongest challenger from 2026 list
        # Priority: runner_up_2021 if they're contesting, else the biggest
        # opposition alliance party present in 2026
        if runner_up_2021 in parties_2026_set:
            predicted_winner_party_2026 = runner_up_2021
        else:
            # Find the best challenger by looking for known strong parties
            # Determine which bloc won and look for the opposing bloc
            if winner_2021 in LDF:
                opponent_bloc = UDF | NDA
            elif winner_2021 in UDF:
                opponent_bloc = LDF | NDA
            elif winner_2021 in NDA | BJP_FRONT:
                opponent_bloc = UDF | LDF | TMC_FRONT | DMK_FRONT
            elif winner_2021 in DMK_FRONT:
                opponent_bloc = ADMK_FRONT
            elif winner_2021 in ADMK_FRONT:
                opponent_bloc = DMK_FRONT
            elif winner_2021 in TMC_FRONT:
                opponent_bloc = BJP_FRONT
            else:
                opponent_bloc = parties_2026_set  # fallback

            challenger = next(
                (p for p in parties_2026_set if p in opponent_bloc),
                runner_up_2021
            )
            predicted_winner_party_2026 = challenger

    return {
        # Identity
        "state": state,
        "district": district,
        "constituency": constituency,
        
        # Historical (TCPD)
        "last_election_year": history["past_election_year"],
        "winner_2021": winner_2021,
        "runner_up_2021": runner_up_2021,
        "predicted_winner_party_2026": predicted_winner_party_2026,
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
        "prediction_basis": "Historical TCPD + ML Risk Model + Gemini Live Sentiment",
    }


async def run_full_pipeline():
    """Runs predictions for ALL constituencies across all 5 states."""
    print("=" * 70)
    print("  ELECTION PREDICTION PIPELINE — 2026 State Assembly Elections")
    print(f"  Run Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load candidate registry
    print("\n[1/3] Loading Wiki Candidate Registry...")
    wiki_registry = load_wiki_registry()
    print(f"       Found {sum(len(v) for v in wiki_registry.values())} candidates across {len(wiki_registry)} constituencies")

    # ─── BUG FIX: Scope predictions ONLY to 2026 election constituencies ──
    # The wiki_static_meta.json was scraped from the official 2026 candidate
    # lists. It is the authoritative source of WHICH constituencies exist in 2026.
    # Using get_all_constituencies() would return thousands of defunct/historical
    # seats from 1962-2009 that do NOT exist in 2026 elections.
    print("[2/4] Deriving 2026-valid constituencies from Wiki candidate registry...")
    wiki_constituency_keys = set(wiki_registry.keys())

    if not wiki_constituency_keys:
        # Fallback to TCPD (only latest year per constituency) if wiki is missing
        print("       [WARN] Wiki registry empty. Falling back to TCPD (may include defunct seats).")
        all_tcpd = get_all_constituencies()
        all_constituencies = all_tcpd
    else:
        # Build list from wiki registry — these are definitively 2026 seats
        all_constituencies = [
            {"state": state, "constituency": const, "district": ""}
            for (state, const) in wiki_constituency_keys
        ]

    print(f"       Confirmed {len(all_constituencies)} active 2026 constituencies")

    # Group by state for progress tracking
    by_state = defaultdict(list)
    for c in all_constituencies:
        by_state[c["state"]].append(c)

    print(f"\n       State Breakdown:")
    for s in sorted(by_state.keys()):
        print(f"         {s}: {len(by_state[s])} constituencies")

    # ─── 3. FETCH LIVE GEMINI DATA ───────────────────────────────────────────
    print(f"\n[3/4] Fetching Live Sentiments (Gemini 2.5 Flash + Tavily)...")
    print(f"       (This uses the 10-batch system with rate-limit delays. Est: ~15 mins)")
    targets = [{"state": c["state"], "constituency": c["constituency"]} for c in all_constituencies]
    
    try:
        live_results = await batch_analyze(targets)
        live_map = {(r["state"], r["constituency"]): r for r in live_results}
        print(f"        Successfully pulled live data for {len(live_map)} constituencies.")
    except Exception as e:
        print(f"        Failed to fetch live data: {e}")
        print("       Falling back to neutral sentiment baseline (0.0).")
        live_map = {}

    # Run predictions
    print(f"\n[4/4] Running Predictions...")
    all_predictions = []
    failed = 0
    
    for state_name in sorted(by_state.keys()):
        constituencies = by_state[state_name]
        print(f"\n  ── {state_name} ({len(constituencies)} seats) ──")
        
        # ─── 1. FETCH STATE-WIDE POLLING & MACRO TARGETS ───
        target_seats = None
        demographic_shifts = None
        try:
            print(f"      Fetching latest polling/survey data for {state_name}...")
            poll_data = await analyze_polling_data(state_name)
            if poll_data:
                target_seats, demographic_shifts = calibrate_from_polling(state_name, poll_data)
                print(f"      Polling Targets Found: {target_seats}")
                if demographic_shifts:
                    print(f"      Demographic Shifts Found: {demographic_shifts}")
        except Exception as e:
            print(f"      Failed to parse polling data: {e}")

        state_predictions = []
        sentiment_map = {}
        success_count = 0
        for c in constituencies:
            try:
                # Extract live sentiment if exists
                live_data = live_map.get((c["state"], c["constituency"]), {})
                sentiment = live_data.get("sentiment_score", 0.0)
                sentiment_map[c["constituency"]] = sentiment
                
                tags = live_data.get("event_tags", [])
                severity = 1.0 if any(t in ["protest", "scandal"] for t in tags) else 0.0

                # Resolve district from TCPD map
                district = resolve_district(c["state"], c["constituency"])
                
                result = await predict_constituency(
                    state=c["state"],
                    constituency=c["constituency"],
                    district=district,
                    wiki_registry=wiki_registry,
                    live_sentiment_score=sentiment,
                    live_severity_score=severity,
                )
                if result:
                    state_predictions.append(result)
                    success_count += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

        # ─── APPLY SWING MODEL ────────────────────────────────────────
        # This corrects the incumbency bias by flipping vulnerable seats
        # based on anti-incumbency patterns and polling-derived macro targets
        state_predictions = apply_uniform_swing(
            state_predictions,
            state_name,
            target_seats=target_seats,
            demographic_shifts=demographic_shifts,
            sentiment_map=sentiment_map,
        )
        all_predictions.extend(state_predictions)

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
        tossup = (state_df["predicted_outcome"].str.contains("TOSS-UP|FLIP")).sum()
        print(f"    {state_name:<15} {len(state_df):>4} seats | Avg Margin: {avg_margin:>5.1f}% | Critical: {critical:>3} | Toss-ups/Flips: {tossup:>3}")

    print(f"\n  ── Alliance-wise Seat Projections ──")
    for state_name in sorted(df["state"].unique()):
        state_df = df[df["state"] == state_name]
        # Classify predicted winners into alliances
        alliance_seats = defaultdict(int)
        for _, row in state_df.iterrows():
            alliance = classify_alliance(state_name, row["predicted_winner_party_2026"])
            alliance_seats[alliance] += 1
        print(f"    {state_name}:")
        for alliance, seats in sorted(alliance_seats.items(), key=lambda x: -x[1]):
            pct = seats / len(state_df) * 100
            bar = "█" * int(pct / 5)
            print(f"      {alliance:<10} {seats:>4} seats ({pct:>5.1f}%)  {bar}")
        print()

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
    asyncio.run(run_full_pipeline())
