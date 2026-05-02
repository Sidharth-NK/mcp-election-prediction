"""
Demographic Node (Live via Serper API)
======================================
Fetches the latest demographic data (rural/urban %, literacy rate) 
for a given district using Google Search via the Serper API.

Results are aggressively cached to disk to prevent exhausting API limits.
"""

import os
import json
import httpx
import asyncio
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "demographics_cache")

os.makedirs(CACHE_DIR, exist_ok=True)

def _get_cache_path(state: str, district: str) -> str:
    """Creates a safe filename for caching district demographics."""
    safe_state = state.replace(" ", "_").lower()
    safe_dist = district.replace(" ", "_").lower()
    return os.path.join(CACHE_DIR, f"{safe_state}_{safe_dist}.json")


async def _serper_search(query: str) -> str:
    """Executes a search using Serper API and returns a concatenated snippet block."""
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY is not set in the environment.")

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "gl": "in"})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, content=payload, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    # Extract text from Answer Box and top organic results
    snippets = []
    if "answerBox" in data and "snippet" in data["answerBox"]:
        snippets.append("ANSWER BOX: " + data["answerBox"]["snippet"])
        
    for res in data.get("organic", [])[:4]:
        if "snippet" in res:
            snippets.append(res["snippet"])

    return "\n".join(snippets)


async def get_demographic_vector(state: str, district: str) -> Dict:
    """
    Fetches the latest demographic vector. Checks local cache first.
    If not found, queries Serper API and uses Gemini to extract the exact numbers.
    """
    # 0. Guard: if district is empty/missing, return fallback immediately (zero API cost)
    if not district or not district.strip():
        return _fallback_vector(state, district, "No district name provided")

    cache_path = _get_cache_path(state, district)
    
    # 1. Check Cache (Zero API cost)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"  > [Demographics] No cache for {district}. Fetching live via Serper API...")
    
    # 2. Fetch live snippets from Serper
    query = f"latest rural urban population percentage and literacy rate of {district} district {state}"
    try:
        search_text = await _serper_search(query)
    except Exception as e:
        print(f"  > [Demographics] Serper API Error: {e}")
        fallback = _fallback_vector(state, district, str(e))
        # Cache the fallback so we don't retry on every constituency
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2)
        return fallback

    # 3. Extract numbers using Groq (Llama 3.3)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("  > [Demographics] Warning: GROQ_API_KEY missing. Cannot parse Serper snippets.")
        fallback = _fallback_vector(state, district, "Groq key missing.")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2)
        return fallback

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Extract the demographic percentages for {district}, {state} from these search results. If exact numbers are missing, estimate based on the text.

Snippets:
{search_text}

Return a JSON object with these exact keys:
- "rural_percentage": float (0-100), default 50.0
- "urban_percentage": float (0-100), default 50.0
- "literacy_rate": float (0-100), default 75.0
- "sc_st_percentage": float (0-100), default 20.0
- "major_demographic_note": brief note on data year
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        parsed = json.loads(response.choices[0].message.content)
        
        result = {
            "status": "success",
            "state": state,
            "district": district,
            "rural_percentage": float(parsed.get("rural_percentage", 50.0)),
            "urban_percentage": float(parsed.get("urban_percentage", 50.0)),
            "literacy_rate": float(parsed.get("literacy_rate", 75.0)),
            "sc_st_percentage": float(parsed.get("sc_st_percentage", 20.0)),
            "major_demographic_note": parsed.get("major_demographic_note", ""),
            "data_source": "Live Serper API + Groq/Llama",
        }
        
        # Save to Cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            
        return result

    except Exception as e:
        print(f"  > [Demographics] Extraction failed: {e}")
        fallback = _fallback_vector(state, district, str(e))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2)
        return fallback


def _fallback_vector(state: str, district: str, error_msg: str) -> Dict:
    """Returns a safe fallback vector if APIs fail, preventing pipeline crash."""
    return {
        "status": "fallback",
        "state": state,
        "district": district,
        "rural_percentage": 50.0,
        "urban_percentage": 50.0,
        "literacy_rate": 75.0,
        "sc_st_percentage": 20.0,
        "major_demographic_note": f"Fallback used due to error: {error_msg}",
        "data_source": "Heuristic Fallback",
    }


if __name__ == "__main__":
    async def test():
        print("=" * 50)
        print("Demographic Node Test (Serper API)")
        print("=" * 50)
        
        # Ensure your .env has SERPER_API_KEY and GEMINI_API_KEY
        res = await get_demographic_vector("Kerala", "Thiruvananthapuram")
        print(json.dumps(res, indent=2))

    asyncio.run(test())
