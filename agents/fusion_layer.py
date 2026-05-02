"""
Feature Fusion Layer

Aligns static, dynamic, and historical nodes into a unified multidimensional feature vector for the Temporal Fusion Transformer (TFT).

Gathering parameters from all MCP nodes and formatting them specifically for PyTorch Forecasting's TFT:
- Static Reals: Demographics, Historical margins
- Static Categoricals: Area names, Past winner
- Time-Varying Unknown Reals: Sentiment Score, Event Severity, Polling percent
- Time-Varying Unknown Categoricals: Risk Level
"""

import json
import asyncio
import datetime
from typing import Dict, Any

from agents.demographic_node import get_demographic_vector
from agents.tcpd_node import get_historical_baseline
from agents.gemini_news_agent import analyze_constituency
from agents.political_model import analyze_political_signal
from agents.polling_node import analyze_polling_data


async def build_tft_feature_vector(
    state: str, 
    district: str, 
    constituency: str,
    time_idx: int = 0
) -> Dict[str, Any]:
    """
    Calls all data ingestion nodes concurrently where possible,
    and formats their outputs into a TFT-compatible feature set.
    """
    print(f"--- Fusing Features for {constituency}, {district}, {state} ---")
    
    # Fetch Static / Historical Data (Synchronously)
    print("  > Fetching Static Variables (Demographics, History)...")
    demographics = await get_demographic_vector(state, district)
    history = get_historical_baseline(state, constituency)
    
    # Fetch Dynamic Data (Async Parallel)
    print("  > Fetching Time-Varying Variables (Sentiment, Polling)...")
    sentiment_task = analyze_constituency(state, constituency)
    polling_task = analyze_polling_data(state)
    
    raw_sentiment, polling = await asyncio.gather(sentiment_task, polling_task)
    
    # Enrich sentiment with political risk model heuristics
    sentiment_risk = analyze_political_signal(raw_sentiment, history)
    
    # Assemble the TFT Data Schema
    print("  > Assembling Matrix...")
    
    static_reals = {
        "rural_percentage": demographics.get("rural_percentage", 0.0),
        "urban_percentage": demographics.get("urban_percentage", 0.0),
        "literacy_rate": demographics.get("literacy_rate", 0.0),
        "sc_st_percentage": demographics.get("sc_st_percentage", 0.0),
        "past_winning_margin": history.get("winning_margin_percentage", 0.0),
        "past_voter_turnout": history.get("voter_turnout_percentage", 0.0),
        "party_loyalty": float(history.get("consecutive_party_wins", 0))
    }
    
    static_categoricals = {
        "state": state,
        "district": district,
        "constituency": constituency,
        "past_winner_party": history.get("winner", "UNKNOWN")
    }
    
    time_varying_unknown_reals = {
        "sentiment_score": sentiment_risk.get("sentiment_score", 0.0),
        "event_severity": sentiment_risk.get("event_severity", 0.0),
    }
    
    time_varying_unknown_categoricals = {
        "risk_level": sentiment_risk.get("ml_risk_level", "LOW")
    }
    
    # Parse polling into party-specific momentum features
    # e.g., 'NDA_polling': 42.5
    for party_math in polling.get("parties", []):
        party_name = party_math.get("party", "UNKNOWN")
        vote_share = party_math.get("projected_vote_share_percentage")
        if vote_share is not None:
            time_varying_unknown_reals[f"poll_{party_name}"] = float(vote_share)

    fusion_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "time_idx": time_idx, 
        "static_reals": static_reals,
        "static_categoricals": static_categoricals,
        "time_varying_unknown_reals": time_varying_unknown_reals,
        "time_varying_unknown_categoricals": time_varying_unknown_categoricals,
        "raw_polling_summary": polling.get("poll_consensus", ""),
        "news_events": sentiment_risk.get("event_tags", [])
    }
    
    return fusion_payload


if __name__ == "__main__":
    async def test():
        result = await build_tft_feature_vector(
            state="Kerala",
            district="Thiruvananthapuram",
            constituency="Thiruvananthapuram"
        )
        print("\n=== TFT FUSION PAYLOAD ===")
        print(json.dumps(result, indent=2))
        
    asyncio.run(test())
