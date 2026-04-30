"""
TFT Historical Dataset Builder (Real Data)
============================================
Converts raw TCPD CSV data into a proper time-series DataFrame
for PyTorch Forecasting's Temporal Fusion Transformer.

Each constituency has multiple election years → these form the time series.
For each (state, constituency, year) we have:
  - Static: state, district, constituency type
  - Time-varying known: year, election number
  - Time-varying unknown: margin, vote share, turnout, n_candidates, ENOP
  - Target: winning margin for the NEXT election (shifted forward by 1)
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "data", "tcpd")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "data")


def build_historical_timeseries() -> pd.DataFrame:
    """
    Builds a clean time-series DataFrame from all TCPD state CSVs.
    Each row = one constituency in one election year (winner-level summary).
    """
    all_data = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath, low_memory=False)
        all_data.append(df)

    raw = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(raw):,} raw candidate records from {len(all_data)} state CSVs")

    # Cast columns
    numeric_cols = ["Year", "Position", "Margin_Percentage", "Turnout_Percentage",
                    "Vote_Share_Percentage", "N_Cand", "ENOP", "Votes", "Electors"]
    for col in numeric_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Drop rows without essential data
    raw = raw.dropna(subset=["Year", "Position", "Margin_Percentage", "Constituency_Name"])

    # ─── Aggregate to constituency-year level (winner row) ────────────────
    winners = raw[raw["Position"] == 1].copy()
    winners = winners.sort_values(["State_Name", "Constituency_Name", "Year"])

    # Create a unique group identifier
    winners["group_id"] = winners["State_Name"] + "::" + winners["Constituency_Name"]

    # ─── Build time_idx: sequential election number per constituency ──────
    winners["time_idx"] = winners.groupby("group_id").cumcount()

    # ─── Features ─────────────────────────────────────────────────────────
    # Static (don't change across time for a constituency)
    winners["state"] = winners["State_Name"].astype(str)
    winners["constituency"] = winners["Constituency_Name"].astype(str)
    winners["district"] = winners["District_Name"].fillna("UNKNOWN").astype(str)
    winners["constituency_type"] = winners["Constituency_Type"].fillna("GEN").astype(str)

    # Time-varying known reals (we know these before the election)
    winners["year"] = winners["Year"].astype(float)
    winners["election_number"] = winners["time_idx"].astype(float) + 1.0

    # Time-varying unknown reals (we discover these AFTER the election)
    winners["margin_pct"] = winners["Margin_Percentage"].fillna(0.0).astype(float)
    winners["turnout_pct"] = winners["Turnout_Percentage"].fillna(70.0).astype(float)
    winners["vote_share_pct"] = winners["Vote_Share_Percentage"].fillna(50.0).astype(float)
    winners["n_candidates"] = winners["N_Cand"].fillna(5).astype(float)
    winners["enop"] = winners["ENOP"].fillna(2.0).astype(float)  # Effective Number of Parties
    winners["total_votes"] = winners["Votes"].fillna(50000).astype(float)
    winners["total_electors"] = winners["Electors"].fillna(150000).astype(float)

    # Winner party as categorical
    winners["winner_party"] = winners["Party"].fillna("IND").astype(str)

    # Incumbency feature
    winners["is_incumbent"] = winners["Incumbent"].fillna(False)
    winners["is_incumbent"] = winners["is_incumbent"].map(
        {True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0, "True": 1.0, "False": 0.0}
    ).fillna(0.0).astype(float)

    # ─── Target: margin of the NEXT election (what we want to predict) ────
    # Shift margin forward by 1 within each constituency group
    winners["target_margin"] = (
        winners.groupby("group_id")["margin_pct"]
        .shift(-1)
    )

    # Drop the last observation per group (no next election to predict)
    before_drop = len(winners)
    winners = winners.dropna(subset=["target_margin"])
    print(f"Dropped {before_drop - len(winners)} rows (last election per constituency, no future target)")

    # ─── Filter: only constituencies with >= 3 elections ──────────────────
    group_counts = winners.groupby("group_id").size()
    valid_groups = group_counts[group_counts >= 2].index
    winners = winners[winners["group_id"].isin(valid_groups)]
    print(f"Kept {winners['group_id'].nunique()} constituencies with >= 3 total elections")

    # Re-index time_idx per group after filtering
    winners["time_idx"] = winners.groupby("group_id").cumcount()

    # ─── Select final columns ────────────────────────────────────────────
    columns = [
        "time_idx", "group_id",
        # Static categoricals
        "state", "constituency", "district", "constituency_type",
        # Time-varying known reals
        "year", "election_number",
        # Time-varying unknown reals
        "margin_pct", "turnout_pct", "vote_share_pct",
        "n_candidates", "enop", "total_votes", "total_electors",
        "is_incumbent",
        # Time-varying unknown categoricals
        "winner_party",
        # Target
        "target_margin",
    ]
    final = winners[columns].copy().reset_index(drop=True)

    # Ensure no NaN
    for col in final.select_dtypes(include=[np.number]).columns:
        final[col] = final[col].fillna(0.0)

    print(f"\nFinal dataset: {len(final):,} rows × {len(final.columns)} columns")
    print(f"Unique constituencies: {final['group_id'].nunique()}")
    print(f"Time steps per constituency: {final.groupby('group_id')['time_idx'].max().describe()}")

    # Summary stats
    print(f"\n── Target (next-election margin) stats ──")
    print(f"  Mean:   {final['target_margin'].mean():.2f}%")
    print(f"  Median: {final['target_margin'].median():.2f}%")
    print(f"  Min:    {final['target_margin'].min():.2f}%")
    print(f"  Max:    {final['target_margin'].max():.2f}%")

    return final


if __name__ == "__main__":
    df = build_historical_timeseries()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tft_historical_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))
