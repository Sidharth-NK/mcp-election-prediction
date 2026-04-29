import os
import csv
import json
from typing import Dict, List, Optional

# Path to the historical TCPD CSV data
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "tcpd_sample.csv")

def get_historical_baseline(state: str, constituency: str) -> Dict:
    """
    Parses the TCPD historical election CSV to find the past election math
    for a given state and constituency. Returns vote shares, turnout, and margins.
    """
    if not os.path.exists(DATA_FILE):
        return {"error": "Historical CSV dataset not found."}

    results = []
    election_year = None
    turnout = None
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["State_Name"].lower() == state.lower() and row["Constituency_Name"].lower() == constituency.lower():
                election_year = row["Election_Year"]
                turnout = float(row["Turnout_Percentage"]) if row["Turnout_Percentage"] else None
                
                results.append({
                    "candidate": row["Candidate"],
                    "party": row["Party"],
                    "position": int(row["Position"]),
                    "votes": int(row["Votes"]),
                    "vote_share_percentage": float(row["Vote_Share_Percentage"]),
                    "margin_percentage": float(row["Margin_Percentage"]) if row["Margin_Percentage"] else 0.0
                })

    if not results:
        return {
            "status": "not_found",
            "message": f"No historical data found for {constituency}, {state}."
        }

    # Sort by position just in case
    results.sort(key=lambda x: x["position"])
    
    # Calculate winner margin
    winner = results[0]
    runner_up = results[1] if len(results) > 1 else None
    
    return {
        "status": "success",
        "state": state,
        "constituency": constituency,
        "past_election_year": election_year,
        "voter_turnout_percentage": turnout,
        "winner": winner["party"],
        "winning_margin_percentage": winner["margin_percentage"],
        "top_candidates": results[:3] # Return top 3 candidates for TFT context
    }

if __name__ == "__main__":
    print("=" * 50)
    print("TCPD Historical Node Test")
    print("=" * 50)
    
    res1 = get_historical_baseline("Kerala", "Manjeshwaram")
    print(json.dumps(res1, indent=2))
