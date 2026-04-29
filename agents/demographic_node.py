"""
Demographic Node
================
Reads the static census demographic dataset and returns a structured
vector for a given (state, district) pair.

This acts as the sociological weighting layer for the TFT forecasting model.
Data source: Census 2011 (current) — see TODO.md for upgrade path to SHRUG.

Known Limitation:
  Demographics are based on Census 2011 data, which is ~15 years old.
  Fast-urbanising constituencies may have drifted significantly.
  Fix: Integrate DevDataLab SHRUG dataset (nighttime satellite lights)
  as a proxy for modern urbanisation ratios.
"""

import os
import json
from typing import Dict, Optional

# Path to the bundled demographics JSON sample
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "demographics_sample.json")

# Lazily loaded cache — parsed once, reused for every subsequent call
_demographics_cache: Optional[Dict] = None


def _load_data() -> Dict:
    """Load and cache the demographics JSON from disk."""
    global _demographics_cache
    if _demographics_cache is None:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"Demographics dataset not found at: {DATA_FILE}\n"
                "Ensure agents/data/demographics_sample.json is present."
            )
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            _demographics_cache = json.load(f)
    return _demographics_cache


def get_demographic_vector(state: str, district: str) -> Dict:
    """
    Retrieve the static census demographic vector for a given state and district.

    Returns a dict with:
      - rural_percentage        : % of population in rural areas
      - urban_percentage        : % of population in urban areas
      - literacy_rate           : overall literacy rate (%)
      - sc_st_percentage        : Scheduled Caste + Scheduled Tribe population (%)
      - major_demographic_note  : qualitative context string
      - data_source             : provenance tag
      - warning                 : data-age caveat (Census 2011)

    Falls back gracefully if state or district is not found in the dataset,
    returning a structured error dict instead of raising an exception —
    this keeps the MCP tool from crashing the entire server on a cache miss.
    """
    try:
        data = _load_data()
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}

    # Case-insensitive state lookup
    state_data: Optional[Dict] = None
    for s_key in data:
        if s_key.lower() == state.lower():
            state_data = data[s_key]
            break

    if state_data is None:
        available_states = list(data.keys())
        return {
            "status": "not_found",
            "message": f"State '{state}' not found in demographics dataset.",
            "available_states": available_states,
        }

    # Case-insensitive district lookup
    district_data: Optional[Dict] = None
    for d_key in state_data:
        if d_key.lower() == district.lower():
            district_data = state_data[d_key]
            break

    if district_data is None:
        available_districts = list(state_data.keys())
        return {
            "status": "not_found",
            "message": f"District '{district}' not found for state '{state}'.",
            "available_districts": available_districts,
        }

    return {
        "status": "success",
        "state": state,
        "district": district,
        "rural_percentage": district_data.get("rural_percentage"),
        "urban_percentage": district_data.get("urban_percentage"),
        "literacy_rate": district_data.get("literacy_rate"),
        "sc_st_percentage": district_data.get("sc_st_percentage"),
        "major_demographic_note": district_data.get("major_demographic_note", ""),
        "data_source": "Census 2011 (India)",
        "warning": (
            "Data is based on Census 2011. May not reflect current urbanisation "
            "in fast-growing constituencies. Upgrade path: SHRUG dataset (DevDataLab)."
        ),
    }


if __name__ == "__main__":
    import json as _json

    print("=" * 50)
    print("Demographic Node Test")
    print("=" * 50)

    # Test 1: Valid lookup
    result = get_demographic_vector("Kerala", "Thiruvananthapuram")
    print("\n[Test 1] Kerala / Thiruvananthapuram:")
    print(_json.dumps(result, indent=2))

    # Test 2: Valid lookup
    result2 = get_demographic_vector("Tamil Nadu", "Chennai")
    print("\n[Test 2] Tamil Nadu / Chennai:")
    print(_json.dumps(result2, indent=2))

    # Test 3: Unknown district — graceful fallback
    result3 = get_demographic_vector("Kerala", "UnknownDistrict")
    print("\n[Test 3] Unknown district (should return not_found):")
    print(_json.dumps(result3, indent=2))

    # Test 4: Unknown state — graceful fallback
    result4 = get_demographic_vector("Goa", "Panaji")
    print("\n[Test 4] Unknown state (should return not_found):")
    print(_json.dumps(result4, indent=2))
