import os 
import json
from mcp.server.fastmcp import FastMCP

from agents.gemini_news_agent import analyze_constituency, batch_analyze

mcp = FastMCP("Election Analyzer")

@mcp.tool()
def get_election_sentiment(state: str, constituency: str) -> str:
    """
    Fetch and analyze real-time political sentiment for a specific constituency.
    Return a JSON string containing sentiment_score, and key_headlines
    """
    result = analyze_constituency(state, constituency)

    return json.dumps(result, indent=2)

@mcp.tool()
def get_batch_political_analysis(targets: List[Dict]) -> str:
    """
    Fetch and analyze multiple constituencies efficiently.
    Returns a JSON string containing analysis for each targeted constituency.
    """
    results = batch_analyze(targets)

    return json.dumps(results, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")