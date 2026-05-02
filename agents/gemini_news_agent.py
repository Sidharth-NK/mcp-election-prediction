"""
Political Sentiment Analyzer 
  1. External search grounding (Tavily) 
  2. Constituency batching  — 10 constituencies per Gemini call
  3. 6-hour TTL file cache  — repeat runs within same half-day cost 0 API calls

Request math for 50 constituencies, polled twice/day: approx 10 Gemini calls/day 
"""

import os
import json
import time
import asyncio
import hashlib
import datetime
import httpx

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY") or os.getenv("NEWS_API_KEY")
CACHE_DIR       = os.getenv("CACHE_DIR", ".cache/sentiment")
CACHE_TTL_HOURS = 6           # Results older than this are considered stale
BATCH_SIZE      = 5           # Reduced from 10 to stay under Groq TPM limits
RESULTS_PER_QUERY = 3         # Tavily results per search query
INTER_BATCH_DELAY = 4.0       # Increased for safer rate-limit compliance

VALID_STATES = [
    "Kerala",
    "Tamil Nadu",
    "West Bengal",
    "Puducherry",
      "Assam",
]

# Local language keywords to force Tavily into grabbing regional media
REGIONAL_SEARCH_TERMS = {
    "Kerala": "തിരഞ്ഞെടുപ്പ് വാർത്തകൾ (election news) ജനവികാരം (public mood)",
    "Tamil Nadu": "தேர்தல் செய்திகள் (election news) மக்கள் கருத்து (public opinion)",
    "West Bengal": "নির্বাচনের খবর (election news) স্থানীয় রাজনীতি (local politics)",
    "Puducherry": "தேர்தல் செய்திகள் (election news)",
    "Assam": "নিৰ্বাচনৰ খবৰ (election news)",
}

EVENT_TAGS = ["alliance", "protest", "scandal", "campaign activity", "general"]

TAVILY_URL = "https://api.tavily.com/search"

GROQ_MODEL = "llama-3.1-8b-instant"

try:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    groq_client = None
    print(f"CRITICAL: Groq client failed — {e}")

os.makedirs(CACHE_DIR, exist_ok=True)

class ConstituencyResult(BaseModel):
    """Sentiment result for a single constituency."""
    constituency: str = Field(description="Exact constituency name as provided.")
    sentiment_score: float = Field(
        description="Score from -1.0 (strongly negative) to +1.0 (strongly positive)."
    )
    event_tags: List[str] = Field(
        description=f"Relevant tags. Choose from: {', '.join(EVENT_TAGS)}"
    )
    reasoning: str = Field(description="Brief reasoning for the score.")
    key_headlines: List[str] = Field(description="2-3 key headlines that influenced the score.")


class BatchAnalysisOutput(BaseModel):
    """Gemini returns one of these per batch call — a list of results."""
    results: List[ConstituencyResult]


def _cache_key(state: str, constituency: str) -> str:
    """
    Cache key encodes state + constituency + current 6-hour window.
    Window 0 = 00:00–05:59,  Window 1 = 06:00–11:59, etc.
    Two polls within the same window return the cached result.
    """
    now   = datetime.datetime.now()
    window = now.hour // CACHE_TTL_HOURS
    raw   = f"{state}|{constituency}|{now.date()}|w{window}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(state: str, constituency: str) -> Optional[Dict]:
    path = os.path.join(CACHE_DIR, f"{_cache_key(state, constituency)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def cache_set(state: str, constituency: str, data: Dict) -> None:
    path = os.path.join(CACHE_DIR, f"{_cache_key(state, constituency)}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# parallel tavily search
def _build_queries(
    state: str,
    constituency: str,
    party: Optional[str] = None,
    candidate: Optional[str] = None,
) -> List[str]:
    queries = [
        f"latest political news {constituency} {state}",
        f"voter sentiment public mood {constituency} {state} election",
        f"election issues local grievances {constituency} {state}",
        f"political alliance candidate defections {state} {constituency}",
    ]
    
    #  INDIAN CONTEXT: Inject native language search to catch regional media
    regional_terms = REGIONAL_SEARCH_TERMS.get(state)
    if regional_terms:
        queries.append(f"{constituency} {regional_terms}")

    if party:
        queries.append(f"{party} campaign momentum conflicts {constituency} {state}")
    if candidate:
        queries.append(f"{candidate} rally impact response {constituency}")
    return queries


async def _tavily_search(http: httpx.AsyncClient, query: str) -> str:
    """Single async Tavily search — returns formatted text block."""
    try:
        r = await http.post(
            TAVILY_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": RESULTS_PER_QUERY,
                "search_depth": "basic",
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


async def fetch_search_results_for_constituency(
    state: str,
    constituency: str,
    party: Optional[str] = None,
    candidate: Optional[str] = None,
) -> str:
    """Fire all queries for one constituency in parallel. Returns combined text."""
    queries = _build_queries(state, constituency, party, candidate)
    async with httpx.AsyncClient() as http:
        results = await asyncio.gather(*[_tavily_search(http, q) for q in queries])
    return "\n".join(results)


async def fetch_search_results_for_batch(batch: List[Dict]) -> Dict[str, str]:
    """
    Fire ALL search queries for ALL constituencies in a batch simultaneously.
    e.g. 10 constituencies × 4 queries = 40 Tavily calls fired in parallel.
    Returns dict: { constituency_name -> combined_search_text }
    """
    async def fetch_one(target: Dict):
        text = await fetch_search_results_for_constituency(
            state=target["state"],
            constituency=target["constituency"],
            party=target.get("party"),
            candidate=target.get("candidate"),
        )
        return target["constituency"], text

    pairs = await asyncio.gather(*[fetch_one(t) for t in batch])
    return dict(pairs)



# BATCHED GEMINI SYNTHESIS  (1 call per 10 constituencies)
def synthesize_batch(
    news_by_constituency: Dict[str, str],
    state: str,
) -> List[ConstituencyResult]:
    """
    Sends all constituencies' news in ONE Groq (Llama 3.3) call.
    Returns a list of structured ConstituencyResult objects.
    """
    # Built a clearly delimited prompt so Llama doesn't mix up constituencies
    sections = []
    constituency_names = list(news_by_constituency.keys())
    for i, (constituency, news) in enumerate(news_by_constituency.items(), 1):
        sections.append(
            f"--- CONSTITUENCY {i}: {constituency}, {state} ---\n{news[:1500]}\n"
        )

    prompt = f"""You are an expert Indian political data scientist.
Below are news summaries for {len(sections)} constituencies in {state}.
Analyze EACH ONE independently.

{''.join(sections)}

Return a JSON object with a single key "results" containing an array of objects.
Each object MUST have these exact keys:
- "constituency": exact constituency name as listed above
- "sentiment_score": float from -1.0 (strongly negative) to +1.0 (strongly positive)
- "event_tags": array of strings from: {EVENT_TAGS}
- "reasoning": brief reasoning for the score
- "key_headlines": array of 2-3 key headlines

Return one result per constituency in the same order. Base analysis ONLY on provided news.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = json.loads(response.choices[0].message.content)
    results = []
    for i, item in enumerate(raw.get("results", [])):
        try:
            results.append(ConstituencyResult(
                constituency=item.get("constituency", constituency_names[i] if i < len(constituency_names) else "UNKNOWN"),
                sentiment_score=float(item.get("sentiment_score", 0.0)),
                event_tags=[t for t in item.get("event_tags", ["general"]) if t in EVENT_TAGS] or ["general"],
                reasoning=item.get("reasoning", "No reasoning provided."),
                key_headlines=item.get("key_headlines", []),
            ))
        except Exception:
            pass
    return results


async def analyze_constituency(
    state: str,
    constituency: str,
    party: Optional[str] = None,
    candidate: Optional[str] = None,
) -> Dict:
    """
    Analyze a single constituency.
    Checks cache first — if fresh result exists, returns it immediately (0 API calls).
    Otherwise: search via Tavily + synthesize via Gemini.
    NOTE: For analyzing many constituencies at once, use batch_analyze() instead
          — it's far more efficient (10 constituencies per Gemini call vs 1).
    """
    # Cache check
    cached = cache_get(state, constituency)
    if cached:
        print(f"  > [{constituency}] Cache HIT — skipping API calls.")
        return cached

    # Search
    print(f"  > [{constituency}] Searching via Tavily...")
    news = await fetch_search_results_for_constituency(state, constituency, party, candidate)

    # Synthesize (single-item batch)
    print(f"  > [{constituency}] Synthesizing with Gemini...")
    results = synthesize_batch({constituency: news}, state)
    r = results[0]

    output = {
        "state":           state,
        "constituency":    constituency,
        "date":            datetime.date.today().isoformat(),
        "sentiment_score": r.sentiment_score,
        "event_tags":      r.event_tags,
        "reasoning":       r.reasoning,
        "key_headlines":   r.key_headlines,
    }

    cache_set(state, constituency, output)
    return output


async def batch_analyze(targets: List[Dict]) -> List[Dict]:
    """
    Analyze multiple constituencies efficiently.

    targets format:
        [
          {"state": "Kerala",     "constituency": "Thiruvananthapuram"},
          {"state": "Tamil Nadu", "constituency": "Chennai Central", "party": "DMK"},
          ...
        ]

    Process:
      1. Separate cached vs uncached targets  → 0 calls for cached ones
      2. Split uncached into batches of BATCH_SIZE (default 10)
      3. For each batch:
           a. Fire ALL Tavily searches in parallel   (0 Gemini calls)
           b. Send all results in ONE Gemini call    (1 Gemini call per 10 constituencies)
      4. Cache all fresh results
      5. Return combined list in original order

    Gemini calls = ceil(uncached_count / BATCH_SIZE)
    """
    if not groq_client:
        raise RuntimeError("Groq client not initialized. Check GROQ_API_KEY.")
    if not TAVILY_API_KEY:
        raise RuntimeError("Tavily API key not set. Check TAVILY_API_KEY.")

    # Validate states upfront
    for t in targets:
        if t["state"] not in VALID_STATES:
            raise ValueError(f"Invalid state '{t['state']}'. Allowed: {VALID_STATES}")

    # split into cached and uncached 
    cached_results: Dict[str, Dict] = {}
    uncached: List[Dict] = []

    for t in targets:
        hit = cache_get(t["state"], t["constituency"])
        if hit:
            print(f"  > [{t['constituency']}] Cache HIT")
            cached_results[t["constituency"]] = hit
        else:
            uncached.append(t)

    print(f"\nCache: {len(cached_results)} hits, {len(uncached)} misses")

    #  Process uncached in batches (Grouped by state!) 
    fresh_results: Dict[str, Dict] = {}
    
    # Group targets by state
    from collections import defaultdict
    targets_by_state = defaultdict(list)
    for t in uncached:
        targets_by_state[t["state"]].append(t)

    #  Build purely separated batches
    batches = []
    for state_group in targets_by_state.values():
        for i in range(0, len(state_group), BATCH_SIZE):
            batches.append(state_group[i : i + BATCH_SIZE])

    for batch_num, batch in enumerate(batches, 1):
        state = batch[0]["state"]  # Now safely guaranteed to be identical for the whole batch

        print(f"\n[Batch {batch_num}/{len(batches)}] Searching {len(batch)} {state} constituencies in parallel...")
        news_map = await fetch_search_results_for_batch(batch)

        print(f"[Batch {batch_num}/{len(batches)}] Synthesizing with 1 Gemini call...")
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, synthesize_batch, news_map, state)

        today = datetime.date.today().isoformat()
        # Fallback safeguard in case Gemini skips one or misaligns
        result_map = {r.constituency: r for r in results}
        
        for t in batch:
            constituency_key = t["constituency"]
            if constituency_key not in result_map:
                print(f"  [!] Warning: Gemini failed to return data for {constituency_key}")
                continue
                
            r = result_map[constituency_key]
            output = {
                "state":           t["state"],
                "constituency":    t["constituency"],
                "date":            today,
                "sentiment_score": r.sentiment_score,
                "event_tags":      r.event_tags,
                "reasoning":       r.reasoning,
                "key_headlines":   r.key_headlines,
            }
            cache_set(t["state"], t["constituency"], output)
            fresh_results[t["constituency"]] = output

        # Polite delay between batches to stay under RPM
        if batch_num < len(batches):
            print(f"  Waiting {INTER_BATCH_DELAY}s before next batch...")
            await asyncio.sleep(INTER_BATCH_DELAY)

    all_results = {**cached_results, **fresh_results}
    return [all_results[t["constituency"]] for t in targets if t["constituency"] in all_results]


#usage example
if __name__ == "__main__":
    async def run_tests():
        # single constituency 
        print("=" * 50)
        print("Single constituency test")
        print("=" * 50)
        result = await analyze_constituency("Kerala", "Thiruvananthapuram")
        print(json.dumps(result, indent=2))

        # Multiple constituencies (efficient batch mode) 
        print("\n" + "=" * 50)
        print("Batch test — 3 constituencies, 1-2 Gemini calls")
        print("=" * 50)

        targets = [
            {"state": "Kerala",     "constituency": "Thiruvananthapuram"},
            {"state": "Kerala",     "constituency": "Thrissur"},
            {"state": "Tamil Nadu", "constituency": "Chennai Central"},
        ]

        results = await batch_analyze(targets)
        for r in results:
            print(f"\n{r['constituency']} ({r['state']})")
            print(f"  Score : {r['sentiment_score']}")
            print(f"  Tags  : {r['event_tags']}")
            print(f"  Reason: {r['reasoning'][:120]}...")

    import asyncio
    asyncio.run(run_tests())