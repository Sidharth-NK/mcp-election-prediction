import os 
import json
from typing import Dict, List
from mcp.server.fastmcp import FastMCP

from agents.gemini_news_agent import analyze_constituency, batch_analyze
from agents.political_model import analyze_political_signal
from wiki_agent.candidate_list_agent import WikiAgent

mcp = FastMCP("Election Analyzer")

@mcp.tool()
async def get_election_sentiment(state: str, constituency: str) -> str:
    """
    Fetch and analyze real-time political sentiment for a specific constituency.
    Return a JSON string containing sentiment_score, and key_headlines
    """
    result = await analyze_constituency(state, constituency)

    return json.dumps(result, indent=2)

@mcp.tool()
async def get_batch_political_analysis(targets: List[Dict]) -> str:
    """
    Fetch and analyze multiple constituencies efficiently.
    Returns a JSON string containing analysis for each targeted constituency.
    """
    results = await batch_analyze(targets)

    return json.dumps(results, indent=2)


@mcp.tool()
def get_political_risk_analysis(gemini_output:Dict) -> str:
    """
    Enriches the gemini sentiment data with politcal risk insights
    Returns risk level, LOW/MODERATE/HIGH/CRITICAL and event severity score
    """

    result = analyze_political_signal(gemini_output)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_candidate_metadata(state: str) -> str:
    """
    Fetch live candidate metadata for a given state from Wikipedia.
    Returns party, alliance, constituency, and candidate identity for all seats.
    Accepts state names: Kerala, Tamil Nadu, West Bengal, Puducherry, Assam.
    """
    agent = WikiAgent(states=[state])
    agent.run()
    meta = agent.get_static_meta()
    return json.dumps(meta, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
