import json
from typing import Dict, List
from mcp.server.fastmcp import FastMCP

from agents.gemini_news_agent import analyze_constituency, batch_analyze
from agents.political_model import analyze_political_signal
from wiki_agent.candidate_list_agent import WikiAgent
from agents.polling_node import analyze_polling_data
from agents.demographic_node import get_demographic_vector
from agents.tcpd_node import get_historical_baseline

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
    Enriches the gemini sentiment data with political risk insights.
    Returns risk level: LOW/MODERATE/HIGH/CRITICAL and event severity score.
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
    # live_enrich=False keeps it fast (no infobox API calls per candidate)
    # html_dir is not used — _extract_state always fetches from WIKI_REST_URLS
    agent = WikiAgent(states=[state], live_enrich=False)
    agent.run()
    meta = agent.get_static_meta()
    return json.dumps(meta, indent=2)

@mcp.tool()
async def get_polling_data(state: str) -> str:
    """
    Fetch and extract the latest polling and survey tracker numbers for a given state.
    Returns JSON containing projected vote shares and seat ranges per party.
    """
    result = await analyze_polling_data(state)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_constituency_demographics(state: str, district: str) -> str:
    """
    Retrieves the static census demographic vector (rural/urban ratio, literacy, caste proxy)
    for a given district. This acts as the sociological weighting layer for the model.
    """
    result = get_demographic_vector(state, district)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_historical_election_data(state: str, constituency: str) -> str:
    """
    Retrieves the historical election mathematical baseline for a given constituency.
    Returns past winner, margin of victory, vote shares, and voter turnout.
    """
    result = get_historical_baseline(state, constituency)
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
