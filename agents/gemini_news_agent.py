import os
import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize the Gemini Client (Assumes GEMINI_API_KEY environment variable is set)
try:
    client = genai.Client()
except Exception as e:
    # We keep a warning but the fetch function will explicitly fail if this is not fixed
    client = None
    print(f"CRITICAL: Gemini Client failed to initialize. Please ensure GEMINI_API_KEY is set. Error: {e}")

# Restrict to the 5 specified states
VALID_STATES = [
    "Kerala", 
    "Tamil Nadu", 
    "West Bengal", 
    "Puducherry", 
    "Karnataka"
]

# Supported event tags based on requirements
EVENT_TAGS = ["alliance", "protest", "scandal", "campaign activity", "general"]

class SentimentAnalysisOutput(BaseModel):
    sentiment_score: float = Field(
        description="Sentiment score ranging from -1.0 (strongly negative) to +1.0 (strongly positive)."
    )
    event_tags: List[str] = Field(
        description=f"List of relevant event tags. Choose from: {', '.join(EVENT_TAGS)}"
    )
    reasoning: str = Field(
        description="Brief reasoning for the sentiment score and event tags based on the retrieved news."
    )
    key_headlines: List[str] = Field(
        description="A list of 2-3 key headlines or news points discovered that influenced this score."
    )

def generate_bidirectional_queries(state: str, constituency: str, party: Optional[str] = None, candidate: Optional[str] = None) -> List[str]:
    """
    Generates multidimensional/bidirectional queries to fetch comprehensive news.
    """
    queries = [
        f"latest political news India {state} constituency {constituency}"
    ]
    
    if party:
        queries.append(f"{state} {constituency} {party} controversy or news")
    
    if candidate:
        queries.append(f"{state} {constituency} candidate {candidate} rally impact news")
        
    return queries

def fetch_and_analyze_news(state: str, constituency: str, party: Optional[str] = None, candidate: Optional[str] = None) -> Dict:
    """
    Uses Gemini 2.5 Flash with Google Search Grounding to simultaneously fetch recent 
    political news and analyze sentiment for a specific constituency.
    """
    if state not in VALID_STATES:
        raise ValueError(f"State '{state}' is not in the allowed list of states: {VALID_STATES}")
        
    if not client:
         raise RuntimeError("Gemini Client is not initialized. Please set the GEMINI_API_KEY environment variable.")

    # Generate our specific search queries
    queries = generate_bidirectional_queries(state, constituency, party, candidate)
    query_bullet_points = "\n".join([f"- {q}" for q in queries])
    
    prompt = f"""
    You are an expert political analyst system.
    Please use Google Search to find the absolute latest news regarding the following queries:
    
    {query_bullet_points}
    
    Based on the real-time news gathered, evaluate the current political environment in {constituency}, {state}.
    
    Provide a structured output containing:
    1. A sentiment_score from -1.0 (highly negative/controversial) to +1.0 (highly positive/strong momentum).
    2. Relevant event_tags such as alliance, protest, scandal, campaign activity, or general.
    3. A brief reasoning based on the facts you found.
    4. Provide 2-3 key headlines you discovered that back up your reasoning.
    """

    # We use Google Search grounding feature to make Gemini fetch the news for us
    response = client.models.generate_content(
        model='gemini-2.0-flash', # Updated to current available production model
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SentimentAnalysisOutput,
            temperature=0.2, 
            tools=[{"google_search": {}}], 
        ),
    )
    
    # Extract parsed Pydantic object
    analysis: SentimentAnalysisOutput = response.parsed
    
    return {
        "state": state,
        "constituency": constituency,
        "date": datetime.date.today().isoformat(),
        "sentiment_score": analysis.sentiment_score,
        "event_tags": analysis.event_tags,
        "reasoning": analysis.reasoning,
        "key_headlines": analysis.key_headlines
    }

if __name__ == "__main__":
    # Example usage for testing the live pipeline
    try:
        print("Testing Live Gemini API News Agent...")
        # Note: Replace with a real constituency you want to test
        results = fetch_and_analyze_news("Kerala", "Wayanad")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error during live test: {e}")

