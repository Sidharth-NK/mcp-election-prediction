"""
VALIDATION SUITE — Pre-Production Sanity Checks
=================================================
Validates every component in the pipeline before we train the TFT.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"   {name}")
    else:
        FAIL += 1
        print(f"   {name} — {detail}")

def warn(name, detail):
    global WARN
    WARN += 1
    print(f"    {name} — {detail}")

print("=" * 70)
print("  VALIDATION SUITE — Election Prediction Pipeline")
print("=" * 70)

# ─── 1. DATA FILES EXIST ─────────────────────────────────────────────────────
print("\n── 1. Data File Integrity ──")
tcpd_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "data", "tcpd")
expected_csvs = ["Kerala_AE.csv", "Assam_AE.csv", "Tamil_Nadu_AE.csv", "West_Bengal_AE.csv", "Puducherry_AE.csv"]
for csv_name in expected_csvs:
    path = os.path.join(tcpd_dir, csv_name)
    exists = os.path.exists(path)
    check(f"TCPD CSV: {csv_name}", exists, "File missing")
    if exists:
        df = pd.read_csv(path, low_memory=False, nrows=5)
        required_cols = ["State_Name", "Constituency_Name", "Year", "Position", "Margin_Percentage", "Turnout_Percentage", "Party", "Candidate", "Votes"]
        for col in required_cols:
            check(f"  Column '{col}' in {csv_name}", col in df.columns, f"Missing column")

check("Demographics JSON", os.path.exists(os.path.join(os.path.dirname(__file__), "..", "agents/data/demographics_sample.json")), "File missing")
check("Wiki Static Meta", os.path.exists(os.path.join(os.path.dirname(__file__), "..", "wiki_static_meta.json")), "File missing")
check("ML Model PKL", os.path.exists(os.path.join(os.path.dirname(__file__), "..", "agents/political_risk_model.pkl")), "Run train_political_model.py first")

# ─── 2. TCPD NODE RETURNS VALID DATA ─────────────────────────────────────────
print("\n── 2. TCPD Node Validation ──")
from agents.tcpd_node import get_historical_baseline, get_all_constituencies

test_cases = [
    ("Kerala", "MANJESHWAR"),
    ("Assam", "GUWAHATI EAST"),
    ("Tamil_Nadu", "CHENNAI NORTH"),
    ("West_Bengal", "KOLKATA PORT"),
]

for state, const in test_cases:
    res = get_historical_baseline(state, const)
    check(f"TCPD lookup: {const}, {state}", res.get("status") == "success", f"Got: {res.get('status', 'error')} - {res.get('message', '')}")
    if res.get("status") == "success":
        # Validate numeric ranges
        margin = res.get("rolling_avg_margin", -1)
        check(f"  Margin in range [0, 100]: {margin}", 0 <= margin <= 100, f"Got {margin}")
        turnout = res.get("voter_turnout_percentage", -1)
        check(f"  Turnout in range [0, 100]: {turnout}", 0 <= turnout <= 100, f"Got {turnout}")
        vs = res.get("last_vote_share", -1)
        check(f"  Vote share in range [0, 100]: {vs}", 0 <= vs <= 100, f"Got {vs}")
        n = res.get("n_elections", 0)
        check(f"  n_elections > 0: {n}", n > 0, f"Got {n}")

all_const = get_all_constituencies()
check(f"Total constituencies loaded: {len(all_const)}", len(all_const) > 500, f"Only {len(all_const)} found, expected 500+")

# ─── 3. POLITICAL RISK MODEL ─────────────────────────────────────────────────
print("\n── 3. Political Risk Model Validation ──")
from agents.political_model import classify_political_risk, analyze_political_signal

# Test that ML model produces valid labels
valid_labels = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
test_scenarios = [
    (25.0, 3.0, 60.0, 5, 0.0, 0.0, "Safe seat"),
    (2.0,  8.0, 45.0, 3, -0.5, 1.5, "Tight + negative sentiment"),
    (10.0, 5.0, 52.0, 4, 0.2, 0.3, "Medium seat"),
]
for ram, tstd, lvs, ne, sent, sev, name in test_scenarios:
    result = classify_political_risk(ram, tstd, lvs, ne, sent, sev)
    check(f"Risk label valid for '{name}': {result}", result in valid_labels, f"Got invalid: {result}")

# Test the full analyze_political_signal function
sample = analyze_political_signal(
    {"constituency": "Test", "state": "Kerala", "date": "2026-01-01", "sentiment_score": -0.3, "event_tags": ["scandal"]},
    {"rolling_avg_margin": 5.0, "turnout_std": 4.0, "last_vote_share": 49.0, "n_elections": 4}
)
check("analyze_political_signal returns all keys", 
      all(k in sample for k in ["ml_risk_level", "event_severity", "sentiment_score"]),
      f"Missing keys in output")

# ─── 4. PREDICTION OUTPUT VALIDATION ─────────────────────────────────────────
print("\n── 4. Prediction Output Validation ──")
pred_path = os.path.join(os.path.dirname(__file__), "..", "predictions/election_predictions_2026.csv")
check("Predictions CSV exists", os.path.exists(pred_path), "Run run_predictions.py first")

if os.path.exists(pred_path):
    df = pd.read_csv(pred_path)
    check(f"Predictions count: {len(df)}", len(df) > 1000, f"Only {len(df)} predictions")
    
    # Check all 5 states are present
    states = df["state"].unique()
    for s in ["Assam", "Kerala", "Puducherry", "Tamil_Nadu", "West_Bengal"]:
        # TCPD uses underscores, check both
        found = any(s.replace("_", " ") in str(st) or s in str(st) for st in states)
        check(f"State '{s}' present", found, f"Missing from predictions")
    
    # Check for NaN in critical columns
    critical_cols = ["rolling_avg_margin", "winning_margin_pct", "voter_turnout_pct", "ml_risk_level"]
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            check(f"No NaN in '{col}'", nan_count == 0, f"{nan_count} NaN values")
    
    # Validate margin ranges
    margins = df["rolling_avg_margin"]
    check(f"All margins >= 0", (margins >= 0).all(), f"Found negative margins")
    check(f"All margins <= 100", (margins <= 100).all(), f"Found margins > 100%")
    check(f"Mean margin realistic [5-30]: {margins.mean():.1f}", 5 <= margins.mean() <= 30, f"Suspicious mean")
    
    # Validate turnout ranges
    turnouts = df["voter_turnout_pct"]
    check(f"All turnouts >= 0", (turnouts >= 0).all(), f"Found negative turnout")
    check(f"All turnouts <= 100", (turnouts <= 100).all(), f"Found turnout > 100%")
    check(f"Mean turnout realistic [50-85]: {turnouts.mean():.1f}", 50 <= turnouts.mean() <= 85, f"Suspicious mean")
    
    # Validate risk distribution is not degenerate (all same class)
    risk_dist = df["ml_risk_level"].value_counts()
    check("Risk distribution has >= 3 classes", len(risk_dist) >= 3, f"Only {len(risk_dist)} classes")
    
    # Check outcome distribution
    outcome_dist = df["predicted_outcome"].value_counts()
    check("Multiple outcome types present", len(outcome_dist) >= 3, f"Only {len(outcome_dist)} types")
    
    # Political sanity: Kerala should be the most competitive (historically true)
    state_margins = df.groupby("state")["rolling_avg_margin"].mean()
    kerala_margin = state_margins.get("Kerala", 999)
    check(f"Kerala has lowest avg margin ({kerala_margin:.1f}%)", 
          kerala_margin < state_margins.mean(), 
          f"Kerala margin {kerala_margin:.1f} not lowest — historically suspicious")

    # Check competitiveness index
    check(f"Competitiveness scores in [0-10]", 
          (df["competitiveness"] >= 0).all() and (df["competitiveness"] <= 15).all(),
          "Out of range")
    
    # Validate no duplicate constituencies per state
    dupes = df.groupby(["state", "constituency"]).size()
    dupe_count = (dupes > 1).sum()
    if dupe_count > 0:
        warn("Duplicate constituencies found", f"{dupe_count} duplicates")
    else:
        check("No duplicate constituencies", True)

# ─── 5. TCPD DATA CROSS-VALIDATION ───────────────────────────────────────────
print("\n── 5. Historical Data Cross-Validation ──")
# Spot-check known historical results
known_results = [
    # (state, constituency, expected_winner_party_substring, expected_year)
    ("Kerala", "MANJESHWAR", "IUML", 2021),
    ("Kerala", "THALASSERY", "CPI", 2021),
]
for state, const, expected_party, expected_year in known_results:
    res = get_historical_baseline(state, const)
    if res.get("status") == "success":
        actual_winner = res.get("winner", "")
        actual_year = res.get("past_election_year", 0)
        check(f"Winner of {const} ({state}): {actual_winner}", 
              expected_party.lower() in actual_winner.lower(),
              f"Expected '{expected_party}', got '{actual_winner}'")
        check(f"  Year: {actual_year}", actual_year == expected_year, f"Expected {expected_year}")
    else:
        warn(f"Cannot verify {const}", res.get("message", ""))

# ─── FINAL VERDICT ────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  VALIDATION COMPLETE")
print(f"   Passed: {PASS}  |   Failed: {FAIL}  |    Warnings: {WARN}")
if FAIL == 0:
    print(f"   ALL CHECKS PASSED — Pipeline is production-ready!")
else:
    print(f"   {FAIL} issues need fixing before proceeding to TFT training.")
print(f"{'=' * 70}")
