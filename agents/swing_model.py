"""
Uniform Swing Model

Implements alliance-level seat prediction using:
1. Proper party → alliance classification for all 5 states
2. Historical alternation pattern detection
3. Uniform swing: flip the most vulnerable seats first
4. Polling calibration: use live poll projections to set swing targets
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ALLIANCE CLASSIFICATION (comprehensive for all 5 states)

# Kerala: LDF vs UDF vs NDA
KERALA_LDF = {
    "CPM", "CPI", "RSP", "JD(S)", "KC(M)", "NCP", "INL", "KTP", "C(S)",
    "SJD", "NLP", "KRSP", "LJD", "CPS", "CONG(S)", "SP", "SJP",
    "KEC(B)", "JD-S", "JDS", "RJD", "RMPOI", "NSC",
    "Janadhipathiya Kerala Congress",
}
KERALA_UDF = {
    "INC", "IUML", "MUL", "KEC", "KEC(M)", "KEC(J)", "KC", "KC(J)",
    "RMPI", "ML", "KSP", "JD(U)", "KCM", "CMPC", "SRP", "DCK",
    "Muslim League", "Indian Union Muslim League", "IND(UDF)",
}
KERALA_NDA = {"BJP", "BDJS", "NSS", "BDML", "TTP", "NDA"}

# Tamil Nadu: DMK+ vs ADMK+
TN_DMK_FRONT = {
    "DMK", "INC", "CPM", "CPI", "VCK", "MDMK", "MMK", "KMDK",
    "AIFB", "MNM", "CPI(M)", "IUML", "MUL",
}
TN_ADMK_FRONT = {
    "ADMK", "AIADMK", "AI-ADMK", "PMK", "BJP", "DMDK", "TTP",
    "TMC(M)", "PT", "BDJS",
}
TN_TVK = {"TVK", "Tamilaga Vettri Kazhagam"}

# West Bengal: TMC vs BJP+ vs Left+
WB_TMC = {"TMC", "AITC", "Trinamool Congress", "All India Trinamool Congress"}
WB_BJP = {
    "BJP", "Bharatiya Janata Party", "AGP", "BGPM",
    "Bharatiya Gorkha Prajatantrik Morcha",
}
WB_LEFT = {
    "CPM", "CPI", "CPI(M)", "AIFB", "RSP", "INC",
    "Indian National Congress", "ISF", "Indian Secular Front",
    "RSSCOMJP", "RSSCMJP",
}

# Assam: NDA (BJP+) vs UPA (INC+) vs Regional
ASSAM_NDA = {"BJP", "AGP", "UPPL", "BPF", "BOPF", "TTP", "Bharatiya Janata Party"}
ASSAM_UPA = {"INC", "AIUDF", "CPM", "CPI", "CPI(M)", "Indian National Congress"}

# Puducherry: NDA vs DMK+
PUDU_NDA = {
    "All India N.R. Congress", "AINRC", "BJP", "Bharatiya Janata Party",
    "ADMK", "AIADMK", "NMK", "Namadhu Makkal Kazhagam",
}
PUDU_DMK = {
    "DMK", "Dravida Munnetra Kazhagam", "INC", "Indian National Congress",
    "VCK", "CPM", "CPI", "CPI(M)", "MDMK",
}


def classify_alliance(state: str, party: str) -> str:
    """Classify a party into its alliance bloc for a given state."""
    if not party:
        return "OTHER"
    p = party.strip()

    if state == "Kerala":
        if p in KERALA_LDF: return "LDF"
        if p in KERALA_UDF: return "UDF"
        if p in KERALA_NDA: return "NDA"
    elif state in ("Tamil Nadu", "Tamil_Nadu"):
        if p in TN_DMK_FRONT: return "DMK+"
        if p in TN_ADMK_FRONT: return "ADMK+"
        if p in TN_TVK: return "TVK"
    elif state in ("West Bengal", "West_Bengal"):
        if p in WB_TMC: return "TMC"
        if p in WB_BJP: return "BJP+"
        if p in WB_LEFT: return "LEFT+"
    elif state == "Assam":
        if p in ASSAM_NDA: return "NDA"
        if p in ASSAM_UPA: return "UPA"
    elif state == "Puducherry":
        if p in PUDU_NDA: return "NDA"
        if p in PUDU_DMK: return "DMK+"

    return "OTHER"


def get_ruling_alliance(state: str) -> str:
    """Returns the alliance that won in 2021 (the current incumbent)."""
    return {
        "Kerala": "LDF",
        "Tamil Nadu": "DMK+",
        "West Bengal": "TMC",
        "Assam": "NDA",
        "Puducherry": "NDA",
    }.get(state, "UNKNOWN")


def get_opposition_alliance(state: str) -> str:
    """Returns the main challenger alliance for each state."""
    return {
        "Kerala": "UDF",
        "Tamil Nadu": "ADMK+",
        "West Bengal": "BJP+",
        "Assam": "UPA",
        "Puducherry": "DMK+",
    }.get(state, "UNKNOWN")



# ANTI-INCUMBENCY STRENGTH (based on historical alternation)

# Computed from actual TCPD historical alternation rates (modern era):
# Kerala (1987+): 57% alternation, but 2021 was an exception (LDF retained)
# historically very strong, adjust upward since 2021 was first non-alternation since 1982
# Tamil Nadu (1989+): 86% alternation → very strong
# West Bengal (2001+): 25% alternation → TMC dominance
# Assam (1991+): 50% alternation → moderate
# Puducherry (1990+): 43% alternation → moderate
ANTI_INCUMBENCY_STRENGTH = {
    "Kerala": 0.75,       # Strong: 2021 was exception, 2026 reversion expected
    "Tamil Nadu": 0.80,   # Very strong: near-perfect alternation
    "West Bengal": 0.20,  # Weak: TMC dominance since 2011
    "Assam": 0.50,        # Moderate: could go either way
    "Puducherry": 0.45,   # Moderate-weak
}


# UNIFORM SWING MODEL

def apply_uniform_swing(
    predictions: List[Dict],
    state: str,
    target_seats: Optional[Dict[str, int]] = None,
    demographic_shifts: Optional[Dict[str, Dict[str, float]]] = None,
    sentiment_map: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Applies a uniform swing model to a state's constituency predictions.
    """
    if not predictions:
        return predictions

    # Classify each prediction into alliance
    for p in predictions:
        winner_2021 = p.get("winner_2021", "")
        p["_alliance_2021"] = classify_alliance(state, winner_2021)
        runner_up = p.get("runner_up_2021", "")
        p["_alliance_runner"] = classify_alliance(state, runner_up)

    # If the polling_node provides macro targets (e.g., predicting a third-front emergence),
    # use the macro allocation engine to intelligently distribute seats based on vulnerability.
    if target_seats:
        return _apply_macro_target_allocation(predictions, state, target_seats, demographic_shifts, sentiment_map)

    # FALLBACK: Pure historical ruling/opposition swing 
    ruling = get_ruling_alliance(state)
    opposition = get_opposition_alliance(state)
    anti_inc = ANTI_INCUMBENCY_STRENGTH.get(state, 0.5)

    # Count current seat distribution by alliance
    current_seats = defaultdict(int)
    for p in predictions:
        current_seats[p["_alliance_2021"]] += 1

    total = len(predictions)
    ruling_seats = current_seats.get(ruling, 0)
    oppo_seats = current_seats.get(opposition, 0)

    # Use anti-incumbency model to estimate swing
    HISTORICAL_OPPOSITION_WIN_SHARE = {
        "Puducherry": 0.63,
    }

    if anti_inc > 0.5:
        oppo_win_share = HISTORICAL_OPPOSITION_WIN_SHARE.get(state, 0.45)
        swing_intensity = (anti_inc - 0.5) * 2
        target_oppo_pct = oppo_seats/total + swing_intensity * (oppo_win_share - oppo_seats/total)
        target_oppo = int(target_oppo_pct * total)
        target_ruling = total - target_oppo - (total - ruling_seats - oppo_seats)
        seats_to_flip = max(0, ruling_seats - target_ruling)
    else:
        swing_intensity = (0.5 - anti_inc) * 2
        seats_to_gain = int(swing_intensity * oppo_seats * 0.1)
        target_ruling = ruling_seats + seats_to_gain
        target_oppo = max(oppo_seats - seats_to_gain, int(total * 0.15))
        seats_to_flip = 0

    print(f"\n    [Swing Model] {state}:")
    print(f"      2021 result: {ruling}={ruling_seats}, {opposition}={oppo_seats}, Other={total - ruling_seats - oppo_seats}")
    print(f"      Anti-incumbency strength: {anti_inc:.0%}")
    print(f"      Target 2026: {ruling}≈{target_ruling}, {opposition}≈{target_oppo}")
    print(f"      Seats to flip from {ruling}→{opposition}: {seats_to_flip}")

    if seats_to_flip <= 0:
        for p in predictions:
            p["swing_applied"] = False
        return predictions

    ruling_seats_list = [p for p in predictions if p["_alliance_2021"] == ruling]

    for p in ruling_seats_list:
        sent = sentiment_map.get(p["constituency"], 0.0) if sentiment_map else 0.0
        p["_vulnerability"] = p["rolling_avg_margin"] + (sent * 5.0)

    ruling_seats_list.sort(key=lambda x: x["_vulnerability"])

    flipped = 0
    for p in ruling_seats_list:
        if flipped >= seats_to_flip:
            break

        runner_alliance = p["_alliance_runner"]
        if runner_alliance == opposition:
            p["predicted_winner_party_2026"] = p["runner_up_2021"]
        else:
            parties_2026 = p.get("parties_2026", "")
            opp_party = _find_opposition_party(state, parties_2026, opposition)
            if opp_party:
                p["predicted_winner_party_2026"] = opp_party
            else:
                p["predicted_winner_party_2026"] = p["runner_up_2021"]

        p["predicted_outcome"] = "SWING / FLIP"
        p["prediction_confidence"] = "MODERATE"
        p["swing_applied"] = True
        flipped += 1

    flipped_set = {id(p) for p in ruling_seats_list[:flipped]}
    for p in predictions:
        if id(p) not in flipped_set:
            p["swing_applied"] = False

    print(f"      Actually flipped: {flipped} seats")
    return predictions


def _apply_macro_target_allocation(
    predictions: List[Dict], 
    state: str, 
    targets: Dict[str, int], 
    demographic_shifts: Optional[Dict[str, Dict[str, float]]],
    sentiment_map: Optional[Dict[str, float]]
) -> List[Dict]:
    """Allocates seats across multiple alliances to exactly match macro targets."""
    current_seats = defaultdict(int)
    for p in predictions:
        current_seats[p["_alliance_2021"]] += 1
        
    surplus_alliances = {}
    deficit_queue = []
    
    # Identify which alliances need to lose seats and which need to gain
    for alliance, current in current_seats.items():
        target = targets.get(alliance, 0)
        if current > target:
            surplus_alliances[alliance] = current - target
            
    for alliance, target in targets.items():
        current = current_seats.get(alliance, 0)
        if target > current:
            deficit_queue.extend([alliance] * (target - current))
            
    print(f"\n    [Macro Allocation Model] {state}:")
    print(f"      Current: {dict(current_seats)}")
    print(f"      Target:  {targets}")
    
    if not surplus_alliances or not deficit_queue:
        for p in predictions: p["swing_applied"] = False
        return predictions
        
    # Rank seats in surplus alliances by vulnerability
    # Vulnerability = Historical Margin + News Penalty - Demographic Wave Bonus
    vulnerable_seats = []
    for p in predictions:
        alliance = p["_alliance_2021"]
        if alliance in surplus_alliances and surplus_alliances[alliance] > 0:
            sent = sentiment_map.get(p["constituency"], 0.0) if sentiment_map else 0.0
            margin = p.get("rolling_avg_margin", 0.0)
            
            # Base vulnerability
            # Higher margin = less vulnerable. Negative sentiment = more vulnerable.
            vuln = margin + (sent * 5.0)
            
            # Apply demographic bonus if polling indicates a wave among specific groups
            demo_bonus = 0.0
            if demographic_shifts:
                for target_alliance, shifts in demographic_shifts.items():
                    if target_alliance in deficit_queue:
                        if "sc_st" in shifts:
                            demo_bonus += shifts["sc_st"] * (p.get("sc_st_pct", 0.0) * 0.1)
                        if "rural" in shifts:
                            demo_bonus += shifts["rural"] * (p.get("rural_pct", 0.0) * 0.1)
                        if "urban" in shifts:
                            demo_bonus += shifts["urban"] * (p.get("urban_pct", 0.0) * 0.1)
            
            # Final Vulnerability: Low score = Flip Target
            p["_vulnerability"] = vuln - demo_bonus
            vulnerable_seats.append(p)
            
    vulnerable_seats.sort(key=lambda x: x["_vulnerability"])
    
    flipped = 0
    for p in vulnerable_seats:
        if not deficit_queue:
            break
            
        alliance_to_lose = p["_alliance_2021"]
        if surplus_alliances[alliance_to_lose] <= 0:
            continue
            
        target_alliance = deficit_queue.pop(0)
        surplus_alliances[alliance_to_lose] -= 1
        
        parties_2026 = p.get("parties_2026", "")
        opp_party = _find_opposition_party(state, parties_2026, target_alliance)
        
        if p["_alliance_runner"] == target_alliance:
            p["predicted_winner_party_2026"] = p["runner_up_2021"]
        elif opp_party:
            p["predicted_winner_party_2026"] = opp_party
        else:
            p["predicted_winner_party_2026"] = f"{target_alliance} Candidate"
            
        p["predicted_outcome"] = "SWING / FLIP"
        p["prediction_confidence"] = "MODERATE"
        p["swing_applied"] = True
        flipped += 1
        
    for p in predictions:
        if "swing_applied" not in p:
            p["swing_applied"] = False
            
    print(f"      Actually flipped: {flipped} seats")
    return predictions


def _find_opposition_party(state: str, parties_2026_str: str, opposition: str) -> Optional[str]:
    """Find the best opposition party from the 2026 candidate list."""
    if not parties_2026_str:
        return None

    parties = [p.strip() for p in parties_2026_str.split(",")]
    for p in parties:
        if classify_alliance(state, p) == opposition:
            return p
    return None



# POLLING CALIBRATION
def calibrate_from_polling(state: str, polling_data: Optional[Dict]) -> tuple[Optional[Dict[str, int]], Optional[Dict[str, Dict[str, float]]]]:
    """
    Converts polling output (vote share projections) into target seat counts and demographic shifts.
    Returns (targets, demographic_shifts) or (None, None) if no data.
    """
    if not polling_data or not polling_data.get("parties"):
        return None, None

    targets = {}
    demographic_shifts = {}
    
    for party_data in polling_data["parties"]:
        party = party_data.get("party", "")
        alliance = classify_alliance(state, party)
        if alliance == "OTHER":
            # Try matching by party name directly to alliance names
            p_upper = party.upper().replace(" ", "")
            if p_upper in ("LDF", "UDF", "NDA", "DMK+", "ADMK+", "TMC", "BJP+", "LEFT+", "UPA", "TVK"):
                alliance = p_upper
            elif "AIADMK" in p_upper:
                alliance = "ADMK+"
            elif "CONG+" in p_upper:
                alliance = "UPA"
            elif "TMC+" in p_upper:
                alliance = "TMC"

        seats_min = party_data.get("projected_seats_min")
        seats_max = party_data.get("projected_seats_max")

        if seats_min is not None and seats_max is not None:
            avg = (seats_min + seats_max) // 2
            targets[alliance] = targets.get(alliance, 0) + avg
        elif seats_min is not None:
            targets[alliance] = targets.get(alliance, 0) + seats_min
            
        shifts = party_data.get("demographic_shifts")
        if shifts:
            demographic_shifts[alliance] = shifts

    return (targets if targets else None), (demographic_shifts if demographic_shifts else None)
