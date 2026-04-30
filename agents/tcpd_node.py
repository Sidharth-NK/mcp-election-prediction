"""
TCPD Historical Node
====================
Reads real TCPD CSV data for all 5 states and provides:
  1. Single constituency lookup (for fusion layer)
  2. Full historical summary with rolling averages (for ML model + TFT)
"""

import os
import csv
import json
import pandas as pd
from typing import Dict, List, Optional
from functools import lru_cache

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tcpd")

# ─── Lazy-loaded Cache ───────────────────────────────────────────────────────
_tcpd_df: Optional[pd.DataFrame] = None

def _load_all_tcpd() -> pd.DataFrame:
    """Loads and caches all TCPD CSVs into one DataFrame."""
    global _tcpd_df
    if _tcpd_df is not None:
        return _tcpd_df

    all_data = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(filepath, low_memory=False)
            all_data.append(df)

    _tcpd_df = pd.concat(all_data, ignore_index=True)

    # Cast essential columns
    for col in ["Margin_Percentage", "Turnout_Percentage", "Vote_Share_Percentage"]:
        _tcpd_df[col] = pd.to_numeric(_tcpd_df[col], errors="coerce")
    _tcpd_df["Year"] = pd.to_numeric(_tcpd_df["Year"], errors="coerce")
    _tcpd_df["Position"] = pd.to_numeric(_tcpd_df["Position"], errors="coerce")

    # Normalize names for matching
    _tcpd_df["_state_lower"] = _tcpd_df["State_Name"].str.lower().str.strip()
    _tcpd_df["_const_lower"] = _tcpd_df["Constituency_Name"].str.lower().str.strip()

    return _tcpd_df


def get_historical_baseline(state: str, constituency: str) -> Dict:
    """
    Returns the most recent election data for a constituency.
    Includes rolling average margin and turnout volatility for the ML model.
    """
    if not os.path.exists(DATA_DIR):
        return {"status": "error", "message": f"TCPD directory not found: {DATA_DIR}"}

    df = _load_all_tcpd()
    s_lower = state.lower().strip()
    c_lower = constituency.lower().strip()

    # Filter to this constituency
    mask = (df["_state_lower"] == s_lower) & (df["_const_lower"] == c_lower)
    match = df[mask].copy()

    if match.empty:
        return {
            "status": "not_found",
            "message": f"No historical data for {constituency}, {state}."
        }

    # Get the MOST RECENT election year
    latest_year = match["Year"].max()
    latest = match[match["Year"] == latest_year].sort_values("Position")

    # Winners only across all years for aggregate stats
    winners = match[match["Position"] == 1].sort_values("Year")

    # Rolling average margin (across all past elections)
    margins = winners["Margin_Percentage"].dropna().tolist()
    rolling_avg_margin = sum(margins) / len(margins) if margins else 0.0

    # Turnout volatility (std of turnout across elections)
    turnouts = winners["Turnout_Percentage"].dropna().tolist()
    if len(turnouts) > 1:
        mean_t = sum(turnouts) / len(turnouts)
        turnout_std = (sum((t - mean_t) ** 2 for t in turnouts) / len(turnouts)) ** 0.5
    else:
        turnout_std = 0.0

    # Latest winner info
    winner_row = latest.iloc[0] if len(latest) > 0 else None

    # Top 3 candidates in latest election
    top3 = []
    for _, row in latest.head(3).iterrows():
        top3.append({
            "candidate": row.get("Candidate", ""),
            "party": row.get("Party", ""),
            "position": int(row["Position"]) if pd.notna(row["Position"]) else 0,
            "votes": int(row["Votes"]) if pd.notna(row.get("Votes")) else 0,
            "vote_share_percentage": float(row["Vote_Share_Percentage"]) if pd.notna(row.get("Vote_Share_Percentage")) else 0.0,
            "margin_percentage": float(row["Margin_Percentage"]) if pd.notna(row.get("Margin_Percentage")) else 0.0,
        })

    return {
        "status": "success",
        "state": state,
        "constituency": constituency,
        "past_election_year": int(latest_year),
        "voter_turnout_percentage": float(winner_row["Turnout_Percentage"]) if winner_row is not None and pd.notna(winner_row.get("Turnout_Percentage")) else 0.0,
        "winner": winner_row["Party"] if winner_row is not None else "UNKNOWN",
        "winning_margin_percentage": float(winner_row["Margin_Percentage"]) if winner_row is not None and pd.notna(winner_row.get("Margin_Percentage")) else 0.0,
        "n_elections": len(winners),
        "rolling_avg_margin": round(rolling_avg_margin, 2),
        "turnout_std": round(turnout_std, 2),
        "last_vote_share": float(winner_row["Vote_Share_Percentage"]) if winner_row is not None and pd.notna(winner_row.get("Vote_Share_Percentage")) else 50.0,
        "top_candidates": top3,
    }


def get_all_constituencies() -> List[Dict]:
    """Returns a list of all unique (state, constituency, district) tuples from TCPD data."""
    df = _load_all_tcpd()
    # Get the most recent year per constituency to find the latest district name
    latest = df.sort_values("Year", ascending=False).drop_duplicates(
        subset=["State_Name", "Constituency_Name"], keep="first"
    )
    result = []
    for _, row in latest.iterrows():
        result.append({
            "state": row["State_Name"],
            "constituency": row["Constituency_Name"],
            "district": row.get("District_Name", ""),
        })
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("TCPD Historical Node Test")
    print("=" * 50)

    res1 = get_historical_baseline("Kerala", "MANJESHWAR")
    print(json.dumps(res1, indent=2))

    print(f"\nTotal unique constituencies in TCPD: {len(get_all_constituencies())}")
