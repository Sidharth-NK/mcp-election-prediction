import os
import json
import asyncio
import datetime
import httpx
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"

try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    # Using 2.5-flash as default, it's fast and handles extraction perfectly
    gemini = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    gemini = None
    print(f"CRITICAL: Gemini client failed — {e}")

# ============================================================
# SCHEMAS for Mathematical Extraction
# ============================================================

class PartyMath(BaseModel):
    party: str = Field(description="Abbreviation of the party or alliance (e.g., LDF, UDF, NDA, DMK, AIADMK).")
    projected_vote_share_percentage: Optional[float] = Field(default=None, description="The projected vote share percentage (e.g., 42.5). Use null if unknown.")
    projected_seats_min: Optional[int] = Field(default=None, description="Minimum projected seats. Use null if unknown.")
    projected_seats_max: Optional[int] = Field(default=None, description="Maximum projected seats. Use null if unknown.")

class PollingOutput(BaseModel):
    state: str = Field(description="The state being surveyed.")
    poll_agencies_referenced: List[str] = Field(description="Agencies mentioned (e.g., CVoter, Axis My India, CSDS).")
    poll_consensus: str = Field(description="A brief 1-2 sentence summary of what the polls generally agree on.")
    projected_winner: str = Field(description="Party or Alliance projected to win.")
    parties: List[PartyMath] = Field(description="Mathematical breakdown per party/alliance.")

# ============================================================
# TAVILY SEARCH (Targeted at Survey Agencies)
# ============================================================

async def _tavily_poll_search(query: str) -> str:
    """Async Tavily search tailored for polling numbers."""
    try:
        async with httpx.AsyncClient() as http:
            r = await http.post(
                TAVILY_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": 4,
                    "search_depth": "advanced", # We need detailed news text to get actual numbers
                    "include_answer": False,
                },
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
            block = f"[Query: {query}]\n"
            for res in data.get("results", []):
                block += f"  • {res.get('title', '')} — {res.get('content', '')}\n"
            return block
    except Exception as e:
        return f"[Query: {query}]\n  ERROR: {e}\n"

async def fetch_polling_data(state: str) -> str:
    """Fires multiple targeted queries to find the latest tracker polls."""
    queries = [
        f"latest opinion poll survey tracker {state} legislative assembly election 2026",
        f"CVoter Axis My India CSDS pre-poll projection {state} elections",
        f"vote share prediction seat projection {state} election latest news"
    ]
    results = await asyncio.gather(*[_tavily_poll_search(q) for q in queries])
    return "\n".join(results)

# ============================================================
# GEMINI SYNTHESIS (Zero Temperature)
# ============================================================

async def analyze_polling_data(state: str) -> Dict:
    """
    Scrapes the latest poll data via Tavily and extracts hard numbers using Gemini.
    """
    print(f"  > [{state}] Searching for latest polling trackers and surveys (Advanced Depth)...")
    raw_news = await fetch_polling_data(state)
    
    print(f"  > [{state}] Synthesizing polling numbers with Gemini (Temp: 0.0)...")
    prompt = f"""You are a specialized electoral data extractor.
Review the following web search snippets regarding opinion polls and surveys for the {state} assembly elections.
Identify any projected vote shares (%) and seat counts. 

Search Snippets:
{raw_news}

If exact polling numbers for 2026 are not yet available in the text, infer the current polling demographic consensus based on the political momentum described, but output null for the math fields to avoid hallucinations. You must return a strict JSON object mapping exactly to the schema.
"""
    response = gemini.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PollingOutput,
            temperature=0.0, # Zero temperature is critical for mathematical extraction
        ),
    )
    
    output = response.parsed
    
    # --- PYTHON MATH VALIDATION LAYER ---
    total_vote_share = sum(p.projected_vote_share_percentage for p in output.parties if p.projected_vote_share_percentage is not None)
    
    # If the LLM hallucinated numbers exceeding 100% significantly, normalize them to preserve the signal ratio
    if total_vote_share > 105.0:
        print(f"  > [{state}] WARNING: Gemini hallucinated invalid math (Sum: {total_vote_share}%). Normalizing to 100%.")
        for p in output.parties:
            if p.projected_vote_share_percentage is not None:
                p.projected_vote_share_percentage = round((p.projected_vote_share_percentage / total_vote_share) * 100, 2)
            
    # Convert Pydantic object to dictionary, adding a timestamp
    result_dict = output.model_dump()
    result_dict["date_aggregated"] = datetime.date.today().isoformat()
    return result_dict

if __name__ == "__main__":
    async def run_tests():
        print("=" * 50)
        print("Polling Node Test - Single State")
        print("=" * 50)
        res = await analyze_polling_data("Kerala")
        print(json.dumps(res, indent=2))

    import asyncio
    asyncio.run(run_tests())
