import os
import json
import asyncio
import datetime
import httpx
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"
GROQ_MODEL = "llama-3.1-8b-instant"

try:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    groq_client = None
    print(f"CRITICAL: Groq client failed — {e}")

# ============================================================
# SCHEMAS for Mathematical Extraction
# ============================================================

class PartyMath(BaseModel):
    party: str = Field(description="Abbreviation of the party or alliance (e.g., LDF, UDF, NDA, DMK, AIADMK).")
    projected_vote_share_percentage: Optional[float] = Field(default=None, description="The projected vote share percentage (e.g., 42.5). Use null if unknown.")
    projected_seats_min: Optional[int] = Field(default=None, description="Minimum projected seats. Use null if unknown.")
    projected_seats_max: Optional[int] = Field(default=None, description="Maximum projected seats. Use null if unknown.")
    demographic_shifts: Optional[Dict[str, Optional[float]]] = Field(default=None, description="Dictionary of demographics shifting to this party. Keys should be exactly 'sc_st', 'rural', or 'urban'. Value is intensity 0.0 to 1.0.")

class PollingOutput(BaseModel):
    state: str = Field(description="The state being surveyed.")
    poll_agencies_referenced: List[str] = Field(description="Agencies mentioned (e.g., CVoter, Axis My India, CSDS).")
    poll_consensus: str = Field(description="A brief 1-2 sentence summary of what the polls generally agree on.")
    projected_winner: Optional[str] = Field(default=None, description="Party or Alliance projected to win. Use null if unknown.")
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
# GROQ SYNTHESIS (Llama 3.3 70B, Zero Temperature)
# ============================================================

async def analyze_polling_data(state: str) -> Dict:
    """
    Scrapes the latest poll data via Tavily and extracts hard numbers using Groq.
    """
    if not groq_client:
        raise RuntimeError("Groq client not initialized. Check GROQ_API_KEY.")
        
    print(f"  > [{state}] Searching for latest polling trackers and surveys (Advanced Depth)...")
    raw_news = await fetch_polling_data(state)
    
    print(f"  > [{state}] Synthesizing polling numbers with Groq/Llama (Temp: 0.0)...")
    prompt = f"""You are a specialized electoral data extractor.
Review the following web search snippets regarding opinion polls and surveys for the {state} assembly elections.
Identify any projected vote shares (%), seat counts, and demographic shifts (e.g., shifts in SC/ST, rural, or urban votes).

Search Snippets:
{raw_news[:10000]}

Extract the data matching the JSON schema precisely. 
If a party is clearly taking the SC/ST, rural, or urban vote, set demographic_shifts keys like 'sc_st': 0.8, 'rural': 0.6.
If no demographic data is present, leave demographic_shifts null.

If exact polling numbers for 2026 are not yet available in the text, infer the current polling demographic consensus based on the political momentum described, but output null for the math fields to avoid hallucinations.

Return a JSON object with these exact keys:
- "state": "{state}"
- "poll_agencies_referenced": array of agency names mentioned
- "poll_consensus": brief 1-2 sentence summary
- "projected_winner": party/alliance projected to win
- "parties": array of objects, each with:
  - "party": abbreviation
  - "projected_vote_share_percentage": float or null
  - "projected_seats_min": int or null
  - "projected_seats_max": int or null
  - "demographic_shifts": object mapping 'sc_st', 'rural', or 'urban' to a float. Omit the key entirely if it is null or does not apply.
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    
    raw = json.loads(response.choices[0].message.content)
    output = PollingOutput(**raw)
    
    # --- PYTHON MATH VALIDATION LAYER ---
    total_vote_share = sum(p.projected_vote_share_percentage for p in output.parties if p.projected_vote_share_percentage is not None)
    
    # If the LLM hallucinated numbers exceeding 100% significantly, normalize them to preserve the signal ratio
    if total_vote_share > 105.0:
        print(f"  > [{state}] WARNING: LLM hallucinated invalid math (Sum: {total_vote_share}%). Normalizing to 100%.")
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
