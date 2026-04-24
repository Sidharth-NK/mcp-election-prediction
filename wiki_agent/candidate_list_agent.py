"""
Wiki Agent — 2026 Assembly Election Candidate Metadata Extractor
=================================================================
Node 1 of the Multi-Horizon Forecast System.

Role in pipeline
----------------
  Provides "Static Meta" to the Model Context Protocol (Central Orchestrator).
  This is the semantic connective tissue between:
    - ECI Node  (hard numbers / results)
    - Gemini    (news sentiment)
    - TFT       (temporal forecasting)

  Without this agent, the downstream models have numbers and sentiment
  but no entity identity — they cannot know *who* a sentiment spike
  refers to, *what party* they belong to, or *which constituency* is
  being discussed.

Three Core Functions
--------------------
  1. Entity Identification
       Maps each candidate name → Wikipedia page URL (biographical anchor).
       Flags whether a Wikipedia article exists (has_wiki_page).

  2. Party Contextualization
       Extracts party full name + Wikipedia URL for every party referenced.
       Deduplicates across all states into a shared party registry.

  3. Constituency Mapping
       Extracts constituency name + Wikipedia URL for every seat.
       Includes district and reservation status where available.

Data Sources
------------
  Saved Wikipedia HTML files (offline-safe, no API rate limits):
    klae_2026.html  — Kerala
    wbae_2026.html  — West Bengal
    tnae_2026.html  — Tamil Nadu
    plae_2026.html  — Puducherry
    alae_2026.html  — Assam

  Live MediaWiki Action API (when --live flag is used):
    Fetches infobox wikitext for candidates/constituencies that have
    Wikipedia pages, enriching records with biographical + geographic data.

Output — Static Meta JSON (fed to orchestrator)
-----------------------------------------------
  {
    "meta": { source, generated_at, total_candidates, per_state },

    "candidates": [
      {
        "entity_id":            "KL-001-LDF",
        "state":                "Kerala",
        "district":             "Kasaragod",
        "constituency_number":  1,
        "constituency":         "Manjeshwaram",
        "constituency_wiki_url":"https://en.wikipedia.org/wiki/Manjeshwaram_Assembly_constituency",
        "alliance":             "LDF",
        "party":                "CPI(M)",
        "party_wiki_url":       "https://en.wikipedia.org/wiki/Communist_Party_of_India_(Marxist)",
        "candidate_name":       "K. R. Jayanandan",
        "candidate_wiki_url":   null,
        "has_wiki_page":        false,
        "remarks":              null
      }, ...
    ],

    "party_registry": {
      "CPI(M)": {
        "full_name": "Communist Party of India (Marxist)",
        "wiki_url":  "https://en.wikipedia.org/wiki/Communist_Party_of_India_(Marxist)",
        "states":    ["Kerala", "Tamil Nadu", ...]
      }, ...
    },

    "constituency_registry": {
      "Manjeshwaram": {
        "wiki_url":    "https://en.wikipedia.org/wiki/Manjeshwaram_Assembly_constituency",
        "state":       "Kerala",
        "district":    "Kasaragod",
        "reservation": null
      }, ...
    }
  }

Requirements
------------
  pip install beautifulsoup4

Usage
-----
  # Basic (offline, from HTML files)
  agent = WikiAgent(html_dir=".")
  meta  = agent.run()
  agent.to_json("output/wiki_static_meta.json")
  agent.to_csv("output/candidates_2026.csv")

  # With live API enrichment (fetches infoboxes for known Wikipedia pages)
  agent = WikiAgent(html_dir=".", live_enrich=True, api_delay=1.0)
  agent.run()
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise ImportError(
        "beautifulsoup4 is not installed.\n"
        "Run:  pip install beautifulsoup4"
    ) from exc


# ══════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════

@dataclass
class CandidateRecord:
    """
    Atomic unit of Static Meta delivered to the orchestrator.
    Each field is an 'anchor' the TFT uses to resolve entity identity.
    """
    # ── Identity ──────────────────────────────────────────────
    entity_id:              str             # e.g. "KL-001-LDF"  (unique across pipeline)
    state:                  str
    district:               str
    constituency_number:    Optional[int]
    constituency:           str
    constituency_wiki_url:  Optional[str]   # geographic context anchor

    # ── Alliance / Party ──────────────────────────────────────
    alliance:               str             # LDF | UDF | NDA | SPA | AIADMK+ | N/A
    party:                  str
    party_wiki_url:         Optional[str]   # party identity anchor

    # ── Candidate ─────────────────────────────────────────────
    candidate_name:         str
    candidate_wiki_url:     Optional[str]   # biographical anchor
    has_wiki_page:          bool            # True if Wikipedia URL was found in HTML

    # ── Optional enrichment (from live API) ───────────────────
    remarks:                Optional[str]   = None
    reservation:            Optional[str]   = None   # GEN / SC / ST
    infobox:                Optional[dict]  = None   # populated by live enrichment


@dataclass
class PartyRecord:
    """Deduplicated party entry in the party registry."""
    abbreviation:   str
    full_name:      str
    wiki_url:       Optional[str]
    states:         List[str]       = field(default_factory=list)
    alliances:      List[str]       = field(default_factory=list)


@dataclass
class ConstituencyRecord:
    """Deduplicated constituency entry in the constituency registry."""
    name:           str
    wiki_url:       Optional[str]
    state:          str
    district:       str
    reservation:    Optional[str]   = None   # GEN / SC / ST


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

HTML_FILES: Dict[str, str] = {
    "Kerala":       "klae_2026.html",
    "West Bengal":  "wbae_2026.html",
    "Tamil Nadu":   "tnae_2026.html",
    "Puducherry":   "plae_2026.html",
    "Assam":        "alae_2026.html",
}

# State abbreviations used in entity_id generation
STATE_ABBR: Dict[str, str] = {
    "Kerala":       "KL",
    "West Bengal":  "WB",
    "Tamil Nadu":   "TN",
    "Puducherry":   "PY",
    "Assam":        "AS",
}

WIKI_API    = "https://en.wikipedia.org/w/api.php"
WIKI_BASE   = "https://en.wikipedia.org/wiki/"
HEADERS     = {"User-Agent": "WikiAgent-PoliticalAnalysis/2.0 (Multi-Horizon Forecast System)"}


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def _cell(cells: list, i: int) -> str:
    return cells[i].strip() if i < len(cells) else ""

def _to_int(val: str) -> Optional[int]:
    return int(val) if str(val).isdigit() else None

def _load_soup(path: Path) -> BeautifulSoup:
    with open(path, encoding="utf-8") as f:
        return BeautifulSoup(f, "html.parser")

def _wikitables(soup: BeautifulSoup) -> list:
    return soup.find_all("table", class_="wikitable")

def _extract_link(cell) -> Optional[str]:
    """Return the first Wikipedia article URL found in a BeautifulSoup cell."""
    for a in cell.find_all("a"):
        href = a.get("href", "")
        if "en.wikipedia.org/wiki/" in href:
            return href
    return None

def _slug_to_title(url: str) -> str:
    """Convert Wikipedia URL to readable page title."""
    slug = url.split("/wiki/")[-1]
    return unquote(slug).replace("_", " ")


# ══════════════════════════════════════════════════════════════
# STATE PARSERS
# (each returns a list of raw dicts with text + URL per field)
# ══════════════════════════════════════════════════════════════

def _parse_kerala(soup: BeautifulSoup) -> list:
    """
    Table 6 | 2 alliances headers + 140 data rows
    Cell layout (after optional district cell):
      [0]no [1]constituency [2]img [3]LDF_party [4]LDF_cand [5]img
      [6]UDF_party [7]UDF_cand [8]img [9]NDA_party [10]NDA_cand
    """
    rows_out = []
    t = _wikitables(soup)[6]
    district = ""
    district_url = None

    for row in t.find_all("tr")[2:]:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if not texts:
            continue

        # District cell (rowspan)
        if texts[0] and not texts[0].isdigit():
            district     = texts[0]
            district_url = _extract_link(cells[0])
            cells  = cells[1:]
            texts  = texts[1:]

        if len(texts) < 11:
            continue

        const_no  = texts[0]
        const_name = texts[1]
        const_url  = _extract_link(cells[1])

        for alliance, pi, ni in [("LDF", 3, 4), ("UDF", 6, 7), ("NDA", 9, 10)]:
            rows_out.append({
                "state": "Kerala", "district": district,
                "district_wiki_url": district_url,
                "constituency_number": _to_int(const_no),
                "constituency": const_name, "constituency_wiki_url": const_url,
                "reservation": None,
                "alliance": alliance,
                "party": texts[pi],        "party_wiki_url": _extract_link(cells[pi]),
                "candidate_name": texts[ni], "candidate_wiki_url": _extract_link(cells[ni]),
                "remarks": None,
            })
    return rows_out


def _parse_west_bengal(soup: BeautifulSoup) -> list:
    """
    Table 2 | 1 header + 317 data rows
    Cell layout (after optional district cell):
      [0]no [1]constituency [2]name [3]img [4]party [5]img [6]remarks
    """
    rows_out = []
    t = _wikitables(soup)[2]
    district = ""
    district_url = None

    for row in t.find_all("tr")[1:]:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if not texts:
            continue

        if texts[0] and not texts[0].isdigit():
            district     = texts[0]
            district_url = _extract_link(cells[0])
            cells  = cells[1:]
            texts  = texts[1:]

        if len(texts) < 3:
            continue

        rows_out.append({
            "state": "West Bengal", "district": district,
            "district_wiki_url": district_url,
            "constituency_number": _to_int(texts[0]),
            "constituency": texts[1], "constituency_wiki_url": _extract_link(cells[1]),
            "reservation": None,
            "alliance": "N/A",
            "party": _cell(texts, 4),     "party_wiki_url": _extract_link(cells[4]) if len(cells) > 4 else None,
            "candidate_name": texts[2],   "candidate_wiki_url": _extract_link(cells[2]),
            "remarks": _cell(texts, 6) or None,
        })
    return rows_out


def _parse_tamil_nadu(soup: BeautifulSoup) -> list:
    """
    Table 7 | 2 headers + 234 data rows
    Cell layout (after optional district cell):
      [0]no [1]constituency [2]img [3]SPA_party [4]SPA_cand [5]img
      [6]AIADMK_party [7]AIADMK_cand
    """
    rows_out = []
    t = _wikitables(soup)[7]
    district = ""
    district_url = None

    for row in t.find_all("tr")[2:]:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if not texts:
            continue

        if texts[0] and not texts[0].isdigit():
            district     = texts[0]
            district_url = _extract_link(cells[0])
            cells  = cells[1:]
            texts  = texts[1:]

        if len(texts) < 6:
            continue

        const_no   = texts[0]
        const_name = texts[1]
        const_url  = _extract_link(cells[1])

        for alliance, pi, ni in [("SPA", 3, 4), ("AIADMK+", 6, 7)]:
            party = _cell(texts, pi)
            name  = _cell(texts, ni)
            if party or name:
                rows_out.append({
                    "state": "Tamil Nadu", "district": district,
                    "district_wiki_url": district_url,
                    "constituency_number": _to_int(const_no),
                    "constituency": const_name, "constituency_wiki_url": const_url,
                    "reservation": None,
                    "alliance": alliance,
                    "party": party,   "party_wiki_url": _extract_link(cells[pi]) if len(cells) > pi else None,
                    "candidate_name": name, "candidate_wiki_url": _extract_link(cells[ni]) if len(cells) > ni else None,
                    "remarks": None,
                })
    return rows_out


def _parse_puducherry(soup: BeautifulSoup) -> list:
    """
    Table 1 | 3 headers + 30 data rows
    Cell layout:
      [0]no [1]constituency [2]reservation [3]name [4]party [5]img [6]alliance [7]img [8]remarks
    """
    rows_out = []
    t = _wikitables(soup)[1]

    for row in t.find_all("tr")[3:]:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if len(texts) < 5:
            continue

        rows_out.append({
            "state": "Puducherry", "district": "Puducherry",
            "district_wiki_url": None,
            "constituency_number": _to_int(texts[0]),
            "constituency": texts[1], "constituency_wiki_url": _extract_link(cells[1]),
            "reservation": texts[2] if texts[2] in ("SC", "ST") else "GEN",
            "alliance": _cell(texts, 6),
            "party": texts[4],     "party_wiki_url": _extract_link(cells[4]),
            "candidate_name": texts[3], "candidate_wiki_url": _extract_link(cells[3]),
            "remarks": _cell(texts, 8) or None,
        })
    return rows_out


def _parse_assam(soup: BeautifulSoup) -> list:
    """
    Table 1 | 1 header + 134 data rows
    Cell layout (after optional district cell):
      [0]no [1]constituency [2]name [3]img [4]party [5]img [6]alliance [7]img [8]remarks
    """
    rows_out = []
    t = _wikitables(soup)[1]
    district = ""
    district_url = None

    for row in t.find_all("tr")[1:]:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if not texts:
            continue

        if texts[0] and not texts[0].isdigit():
            district     = texts[0]
            district_url = _extract_link(cells[0])
            cells  = cells[1:]
            texts  = texts[1:]

        if len(texts) < 3:
            continue

        # Detect reservation suffix e.g. "Ratabari(SC)"
        const_raw = texts[1]
        reservation = None
        m = re.search(r'\((SC|ST)\)', const_raw)
        if m:
            reservation = m.group(1)
            const_clean = const_raw[:m.start()].strip()
        else:
            const_clean = const_raw

        rows_out.append({
            "state": "Assam", "district": district,
            "district_wiki_url": district_url,
            "constituency_number": _to_int(texts[0]),
            "constituency": const_clean, "constituency_wiki_url": _extract_link(cells[1]),
            "reservation": reservation,
            "alliance": _cell(texts, 6),
            "party": _cell(texts, 4),     "party_wiki_url": _extract_link(cells[4]) if len(cells) > 4 else None,
            "candidate_name": texts[2],   "candidate_wiki_url": _extract_link(cells[2]),
            "remarks": _cell(texts, 8) or None,
        })
    return rows_out


PARSERS = {
    "Kerala":      _parse_kerala,
    "West Bengal": _parse_west_bengal,
    "Tamil Nadu":  _parse_tamil_nadu,
    "Puducherry":  _parse_puducherry,
    "Assam":       _parse_assam,
}


# ══════════════════════════════════════════════════════════════
# LIVE API ENRICHMENT (optional)
# ══════════════════════════════════════════════════════════════

def _fetch_infobox(title: str, delay: float = 1.0) -> Optional[dict]:
    """
    Fetch and parse infobox wikitext for a Wikipedia page.
    Returns a dict of infobox key-value pairs, or None on failure.
    """
    try:
        import requests
        params = {
            "action": "query", "prop": "revisions",
            "rvprop": "content", "rvslots": "main",
            "titles": title, "format": "json", "formatversion": "2",
        }
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", [])
        if not pages:
            return None
        wikitext = pages[0].get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
        return _parse_infobox_wikitext(wikitext)
    except Exception:
        return None
    finally:
        time.sleep(delay)


def _parse_infobox_wikitext(wikitext: str) -> Optional[dict]:
    """
    Parse {{Infobox ...}} from raw wikitext into a clean dict.
    Extracts: birth_date, birth_place, occupation, alma_mater,
              office, term_start, term_end, predecessor, party history.
    """
    match = re.search(r'\{\{Infobox[^|]*\|(.+?)\}\}(?:\s*\{\{)', wikitext, re.DOTALL)
    if not match:
        match = re.search(r'\{\{Infobox[^|]*\|(.+)', wikitext, re.DOTALL)
    if not match:
        return None

    body = match.group(1)
    result = {}
    # Split on | followed by key =
    for part in re.split(r'\n\s*\|', body):
        if "=" in part:
            key, _, val = part.partition("=")
            key = key.strip().lower().replace(" ", "_")
            # Clean wiki markup from value
            val = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', val)
            val = re.sub(r'\{\{[^}]*\}\}', '', val)
            val = re.sub(r'<[^>]+>', '', val)
            val = re.sub(r'\s+', ' ', val).strip()
            if key and val:
                result[key] = val

    # Keep only biographical fields relevant to political analysis
    relevant = {
        "birth_date", "birth_place", "occupation", "alma_mater",
        "office", "term_start", "term_end", "predecessor", "successor",
        "party", "other_party", "nationality", "spouse", "children",
        "residence", "website",
    }
    return {k: v for k, v in result.items() if k in relevant} or None


# ══════════════════════════════════════════════════════════════
# WIKI AGENT
# ══════════════════════════════════════════════════════════════

class WikiAgent:
    """
    Wiki Agent — Node 1 of the Multi-Horizon Forecast System.

    Extracts and structures candidate metadata from saved Wikipedia HTML
    files for 5 Indian state assembly elections (2026).

    Outputs
    -------
    Static Meta JSON  →  Model Context Protocol (Orchestrator)
      - candidates[]         : full candidate records with all wiki anchors
      - party_registry{}     : deduplicated party identity map
      - constituency_registry{} : deduplicated constituency map
    """

    def __init__(
        self,
        states:       Optional[List[str]] = None,
        html_dir:     str  = ".",
        live_enrich:  bool = False,
        api_delay:    float = 1.0,
    ):
        self.states      = states or list(HTML_FILES.keys())
        self.html_dir    = Path(html_dir)
        self.live_enrich = live_enrich
        self.api_delay   = api_delay

        self.candidates:            List[CandidateRecord] = []
        self.party_registry:        Dict[str, PartyRecord] = {}
        self.constituency_registry: Dict[str, ConstituencyRecord] = {}
        self.errors:                Dict[str, str] = {}
        self._counters:             Dict[str, int] = {}   # per-state entity_id counters

    # ── Public API ────────────────────────────────────────────

    def run(self) -> "WikiAgent":
        """Execute full extraction pipeline. Returns self for chaining."""
        print("=" * 65)
        print("  WIKI AGENT  —  2026 Assembly Election Static Meta Extractor")
        print("=" * 65)

        for state in self.states:
            print(f"\n[{state}]")
            try:
                self._extract_state(state)
            except FileNotFoundError as e:
                self.errors[state] = f"File not found: {e}"
                print(f"  ✗  {self.errors[state]}")
            except Exception as e:
                self.errors[state] = str(e)
                print(f"  ✗  Parse error: {e}")

        self._print_summary()
        return self

    def get_static_meta(self) -> dict:
        """
        Returns the full Static Meta payload for the orchestrator.
        This is what flows along the 'Static Meta' edge in the system design.
        """
        return {
            "meta": {
                "source":          "Wikipedia (saved HTML + MediaWiki Action API)",
                "generated_at":    datetime.now(timezone.utc).isoformat(),
                "states":          self.states,
                "total_candidates": len(self.candidates),
                "per_state":       {
                    s: sum(1 for c in self.candidates if c.state == s)
                    for s in self.states
                },
                "wiki_coverage": {
                    "candidates_with_wiki_page": sum(1 for c in self.candidates if c.has_wiki_page),
                    "candidates_total":          len(self.candidates),
                    "constituency_coverage":     len(self.constituency_registry),
                    "party_coverage":            len(self.party_registry),
                },
                "errors": self.errors,
            },
            "candidates": [asdict(c) for c in self.candidates],
            "party_registry": {
                k: asdict(v) for k, v in sorted(self.party_registry.items())
            },
            "constituency_registry": {
                k: asdict(v) for k, v in sorted(self.constituency_registry.items())
            },
        }

    def to_json(self, path: str = "wiki_static_meta.json") -> None:
        """Export full Static Meta to JSON (primary orchestrator handoff file)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_static_meta(), f, ensure_ascii=False, indent=2)
        print(f"\n  JSON  → {path}  ({len(self.candidates)} candidates, "
              f"{len(self.party_registry)} parties, "
              f"{len(self.constituency_registry)} constituencies)")

    def to_csv(self, path: str = "candidates_2026.csv") -> None:
        """Export flat candidate list to CSV."""
        if not self.candidates:
            print("  No candidates to export.")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Flatten infobox dict into columns
        fieldnames = [
            f for f in CandidateRecord.__dataclass_fields__ if f != "infobox"
        ] + ["infobox_json"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.candidates:
                row = asdict(c)
                infobox_data = row.pop("infobox", None)
                row["infobox_json"] = json.dumps(infobox_data) if infobox_data else ""
                writer.writerow(row)
        print(f"  CSV   → {path}  ({len(self.candidates)} records)")

    def print_sample(self, n: int = 3) -> None:
        """Print sample records per state."""
        print(f"\n{'─'*65}")
        print(f"  SAMPLE OUTPUT  (first {n} per state)")
        print(f"{'─'*65}")
        seen = {}
        for c in self.candidates:
            seen.setdefault(c.state, [])
            if len(seen[c.state]) < n:
                seen[c.state].append(c)
        for state, records in seen.items():
            print(f"\n  {state}")
            print(f"  {'ID':<12} {'Constituency':<22} {'Alliance':<9} {'Party':<10} {'Candidate':<25} {'Wiki?'}")
            print(f"  {'─'*85}")
            for c in records:
                wiki = "✓" if c.has_wiki_page else "–"
                print(f"  {c.entity_id:<12} {c.constituency:<22} {c.alliance:<9} "
                      f"{c.party:<10} {c.candidate_name:<25} {wiki}")

    # ── Internal ──────────────────────────────────────────────

    def _extract_state(self, state: str) -> None:
        path = self.html_dir / HTML_FILES[state]
        if not path.exists():
            raise FileNotFoundError(path)

        soup   = _load_soup(path)
        parser = PARSERS[state]
        rows   = parser(soup)
        abbr   = STATE_ABBR[state]
        self._counters[state] = 0

        for row in rows:
            self._counters[state] += 1
            entity_id = f"{abbr}-{row['constituency_number'] or self._counters[state]:03d}-{row['alliance']}"

            candidate = CandidateRecord(
                entity_id             = entity_id,
                state                 = row["state"],
                district              = row["district"],
                constituency_number   = row["constituency_number"],
                constituency          = row["constituency"],
                constituency_wiki_url = row.get("constituency_wiki_url"),
                alliance              = row["alliance"],
                party                 = row["party"],
                party_wiki_url        = row.get("party_wiki_url"),
                candidate_name        = row["candidate_name"],
                candidate_wiki_url    = row.get("candidate_wiki_url"),
                has_wiki_page         = bool(row.get("candidate_wiki_url")),
                remarks               = row.get("remarks"),
                reservation           = row.get("reservation"),
                infobox               = None,
            )

            # Live API enrichment
            if self.live_enrich and candidate.has_wiki_page:
                title = _slug_to_title(candidate.candidate_wiki_url)
                candidate.infobox = _fetch_infobox(title, self.api_delay)

            self.candidates.append(candidate)
            self._update_party_registry(row, state)
            self._update_constituency_registry(row, state)

        wiki_count = sum(1 for c in self.candidates if c.state == state and c.has_wiki_page)
        total      = sum(1 for c in self.candidates if c.state == state)
        print(f"  ✓  {total} candidate records  |  "
              f"Wikipedia pages found: {wiki_count}/{total} candidates  |  "
              f"{len([c for c in self.constituency_registry.values() if c.state == state])} constituencies")

    def _update_party_registry(self, row: dict, state: str) -> None:
        party = row.get("party", "").strip()
        if not party or party in ("-", "—", ""):
            return
        if party not in self.party_registry:
            self.party_registry[party] = PartyRecord(
                abbreviation = party,
                full_name    = party,
                wiki_url     = row.get("party_wiki_url"),
                states       = [],
                alliances    = [],
            )
        rec = self.party_registry[party]
        if state not in rec.states:
            rec.states.append(state)
        alliance = row.get("alliance", "")
        if alliance and alliance not in rec.alliances and alliance != "N/A":
            rec.alliances.append(alliance)
        # Prefer longer full_name (wiki URL title is more descriptive)
        if row.get("party_wiki_url") and not rec.wiki_url:
            rec.wiki_url   = row["party_wiki_url"]
            rec.full_name  = _slug_to_title(row["party_wiki_url"])

    def _update_constituency_registry(self, row: dict, state: str) -> None:
        const = row.get("constituency", "").strip()
        if not const:
            return
        key = f"{state}::{const}"
        if key not in self.constituency_registry:
            self.constituency_registry[key] = ConstituencyRecord(
                name        = const,
                wiki_url    = row.get("constituency_wiki_url"),
                state       = state,
                district    = row.get("district", ""),
                reservation = row.get("reservation"),
            )

    def _print_summary(self) -> None:
        total      = len(self.candidates)
        with_wiki  = sum(1 for c in self.candidates if c.has_wiki_page)
        print(f"\n{'='*65}")
        print(f"  EXTRACTION COMPLETE")
        print(f"{'─'*65}")
        print(f"  {'State':<16}  {'Records':>7}  {'Wiki Pages':>10}  {'Coverage':>9}")
        print(f"  {'─'*50}")
        for state in self.states:
            sc = [c for c in self.candidates if c.state == state]
            sw = sum(1 for c in sc if c.has_wiki_page)
            pct = f"{100*sw//len(sc)}%" if sc else "N/A"
            err = "  ← ERROR" if state in self.errors else ""
            print(f"  {state:<16}  {len(sc):>7}  {sw:>10}  {pct:>9}{err}")
        print(f"  {'─'*50}")
        print(f"  {'TOTAL':<16}  {total:>7}  {with_wiki:>10}  {100*with_wiki//total if total else 0:>8}%")
        print(f"\n  Party registry     : {len(self.party_registry)} unique parties")
        print(f"  Constituency registry: {len(self.constituency_registry)} unique constituencies")
        print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wiki Agent — 2026 Election Static Meta Extractor")
    parser.add_argument("html_dir",    nargs="?", default=".",    help="Directory containing HTML files")
    parser.add_argument("--states",    nargs="+", default=None,   help="States to process (default: all)")
    parser.add_argument("--live",      action="store_true",       help="Enable live Wikipedia API enrichment")
    parser.add_argument("--delay",     type=float, default=1.0,   help="API delay in seconds (default: 1.0)")
    parser.add_argument("--out-dir",   default="output",          help="Output directory (default: output/)")
    parser.add_argument("--sample",    type=int,   default=3,     help="Sample rows to print per state")
    args = parser.parse_args()

    agent = WikiAgent(
        states      = args.states,
        html_dir    = args.html_dir,
        live_enrich = args.live,
        api_delay   = args.delay,
    )

    agent.run()
    agent.print_sample(n=args.sample)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    agent.to_json(f"{args.out_dir}/wiki_static_meta.json")
    agent.to_csv(f"{args.out_dir}/candidates_2026.csv")

    print(f"\n  Ready for orchestrator handoff.")
    print(f"  Load with:  json.load(open('{args.out_dir}/wiki_static_meta.json'))")