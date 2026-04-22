"""
Political Sentiment Analyzer — v2 (Optimized)
=============================================
Optimizations applied vs v1:
  1. External search (Tavily) — removes Gemini's search grounding multiplier
  2. Constituency batching  — 10 constituencies per Gemini call (10× fewer LLM calls)
  3. 6-hour TTL file cache  — repeat runs within same half-day cost 0 API calls

Request math for 50 constituencies, polled twice/day:
  Before (v1 original) : 200 Gemini calls/day
  After  (v2 this file): ~10 Gemini calls/day  (5 batches × 2 polls, cached after first run)
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
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
CACHE_DIR       = os.getenv("CACHE_DIR", ".cache/sentiment")
CACHE_TTL_HOURS = 6          # Results older than this are considered stale
BATCH_SIZE      = 10         # Constituencies per Gemini call (keep ≤ 10)
RESULTS_PER_QUERY = 3        # Tavily results per search query
INTER_BATCH_DELAY = 4.0      # Seconds to wait between Gemini batch calls (RPM safety)

VALID_STATES = [
    "Kerala",
    "Tamil Nadu",
    "West Bengal",
    "Puducherry",
    "Karnataka",
]

EVENT_TAGS = ["alliance", "protest", "scandal", "campaign activity", "general"]

TAVILY_URL = "https://api.tavily.com/search"

# ============================================================
# CLIENT INIT
# ============================================================

try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    gemini = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    gemini = None
    print(f"CRITICAL: Gemini client failed — {e}")

os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# SCHEMAS
# ============================================================

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


# ============================================================
# CACHE  (file-based, 6-hour TTL)
# ============================================================

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


# ============================================================
# STEP 1 — PARALLEL TAVILY SEARCH  (0 Gemini calls)
# ============================================================

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


# ============================================================
# STEP 2 — BATCHED GEMINI SYNTHESIS  (1 call per 10 constituencies)
# ============================================================

def synthesize_batch(
    news_by_constituency: Dict[str, str],
    state: str,
) -> List[ConstituencyResult]:
    """
    Sends all constituencies' news in ONE Gemini call.
    Returns a list of structured ConstituencyResult objects.
    """
    # Build a clearly delimited prompt so Gemini doesn't mix up constituencies
    sections = []
    for i, (constituency, news) in enumerate(news_by_constituency.items(), 1):
        sections.append(
            f"--- CONSTITUENCY {i}: {constituency}, {state} ---\n{news}\n"
        )

    prompt = f"""You are an expert Indian political data scientist.
Below are news summaries for {len(sections)} constituencies in {state}.
Analyze EACH ONE independently and return a JSON array of results.

{''.join(sections)}

Return one result object per constituency in the same order as listed above.
Base your analysis ONLY on the provided news data.
"""

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchAnalysisOutput,
            temperature=0.1,
        ),
    )

    return response.parsed.results


# ============================================================
# MAIN PUBLIC API
# ============================================================

def analyze_constituency(
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
    news = asyncio.run(
        fetch_search_results_for_constituency(state, constituency, party, candidate)
    )

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


def batch_analyze(targets: List[Dict]) -> List[Dict]:
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
    if not gemini:
        raise RuntimeError("Gemini client not initialized. Check GEMINI_API_KEY.")
    if not TAVILY_API_KEY:
        raise RuntimeError("Tavily API key not set. Check TAVILY_API_KEY.")

    # Validate states upfront
    for t in targets:
        if t["state"] not in VALID_STATES:
            raise ValueError(f"Invalid state '{t['state']}'. Allowed: {VALID_STATES}")

    # --- Split into cached and uncached ---
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

    # --- Process uncached in batches (Grouped by state!) ---
    fresh_results: Dict[str, Dict] = {}
    
    # 1. Group targets by state
    from collections import defaultdict
    targets_by_state = defaultdict(list)
    for t in uncached:
        targets_by_state[t["state"]].append(t)

    # 2. Build purely separated batches
    batches = []
    for state_group in targets_by_state.values():
        for i in range(0, len(state_group), BATCH_SIZE):
            batches.append(state_group[i : i + BATCH_SIZE])

    for batch_num, batch in enumerate(batches, 1):
        state = batch[0]["state"]  # Now safely guaranteed to be identical for the whole batch

        print(f"\n[Batch {batch_num}/{len(batches)}] Searching {len(batch)} {state} constituencies in parallel...")
        news_map = asyncio.run(fetch_search_results_for_batch(batch))

        print(f"[Batch {batch_num}/{len(batches)}] Synthesizing with 1 Gemini call...")
        results = synthesize_batch(news_map, state)

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
            time.sleep(INTER_BATCH_DELAY)

    # --- Merge and return in original order ---
    all_results = {**cached_results, **fresh_results}
    return [all_results[t["constituency"]] for t in targets if t["constituency"] in all_results]


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":

    # --- Single constituency ---
    print("=" * 50)
    print("Single constituency test")
    print("=" * 50)
    result = analyze_constituency("Kerala", "Thiruvananthapuram")
    print(json.dumps(result, indent=2))

    # --- Multiple constituencies (efficient batch mode) ---
    print("\n" + "=" * 50)
    print("Batch test — 3 constituencies, 1-2 Gemini calls")
    print("=" * 50)

    targets = [
        {"state": "Kerala",     "constituency": "Thiruvananthapuram"},
        {"state": "Kerala",     "constituency": "Thrissur"},
        {"state": "Tamil Nadu", "constituency": "Chennai Central"},
    ]

    results = batch_analyze(targets)
    for r in results:
        print(f"\n{r['constituency']} ({r['state']})")
        print(f"  Score : {r['sentiment_score']}")
        print(f"  Tags  : {r['event_tags']}")
        print(f"  Reason: {r['reasoning'][:120]}...")