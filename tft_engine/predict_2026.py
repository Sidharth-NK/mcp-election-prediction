"""
TFT 2026 Prediction Generator
===============================
Uses the trained TFT model to generate probabilistic predictions 
for all constituencies in the 2026 state assembly elections.

For each constituency, outputs:
  - P10, P25, P50 (median), P75, P90 quantile predictions of winning margin
  - Risk classification based on predicted margin uncertainty
"""

import os
import sys
import warnings
import json

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tft_historical_dataset.csv")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")


def find_best_checkpoint():
    """Find the best model checkpoint."""
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    # Sort by val_loss in filename
    ckpts.sort()
    return os.path.join(CHECKPOINT_DIR, ckpts[0])


def margin_to_outcome(p50: float, p10: float, p90: float) -> tuple:
    """Converts predicted margin quantiles to outcome labels and confidence."""
    spread = p90 - p10  # Prediction uncertainty

    if p50 > 15.0:
        outcome = "SAFE HOLD"
        confidence = "HIGH" if spread < 15 else "MODERATE"
    elif p50 > 7.0:
        outcome = "LIKELY HOLD"
        confidence = "MODERATE" if spread < 20 else "LOW"
    elif p50 > 3.0:
        outcome = "LEAN / TOSS-UP"
        confidence = "LOW"
    else:
        outcome = "TOSS-UP / POSSIBLE FLIP"
        confidence = "VERY LOW"

    return outcome, confidence, spread


def generate_predictions():
    print("=" * 70)
    print("  TFT 2026 PREDICTIONS — Indian Assembly Elections")
    print("=" * 70)

    # ─── Load Data ────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    df["time_idx"] = df["time_idx"].astype(int)
    for col in ["state", "constituency", "district", "constituency_type", "winner_party", "group_id"]:
        df[col] = df[col].astype(str)
    df["target_margin"] = df["target_margin"].clip(0.0, 60.0)

    # Filter same as training
    max_time = df.groupby("group_id")["time_idx"].max()
    valid_groups = max_time[max_time >= 3].index
    df = df[df["group_id"].isin(valid_groups)].reset_index(drop=True)

    # Top parties (same as training)
    top_parties = df["winner_party"].value_counts().head(20).index.tolist()
    df["winner_party"] = df["winner_party"].where(df["winner_party"].isin(top_parties), "OTHER")

    print(f"Loaded {df['group_id'].nunique()} constituencies")

    # ─── Reconstruct Training Dataset (needed for schema) ────────────────
    max_encoder_length = 6
    max_prediction_length = 1
    training_cutoff = df.groupby("group_id")["time_idx"].transform("max") - max_prediction_length
    train_df = df[df["time_idx"] <= training_cutoff].copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target_margin",
        group_ids=["group_id"],
        min_encoder_length=2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["state", "constituency_type"],
        static_reals=[],
        time_varying_known_reals=["time_idx", "year", "election_number"],
        time_varying_known_categoricals=[],
        time_varying_unknown_reals=[
            "margin_pct", "turnout_pct", "vote_share_pct",
            "n_candidates", "enop", "is_incumbent",
        ],
        time_varying_unknown_categoricals=["winner_party"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Build prediction dataset (full data, uses last time steps as encoder context)
    prediction_data = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    pred_loader = prediction_data.to_dataloader(train=False, batch_size=128, num_workers=0)

    # ─── Load Trained Model ──────────────────────────────────────────────
    ckpt_path = find_best_checkpoint()
    print(f"Loading checkpoint: {os.path.basename(ckpt_path)}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    # ─── Generate Predictions ─────────────────────────────────────────────
    print("Generating quantile predictions...")
    predictions = best_tft.predict(pred_loader, mode="quantiles", return_x=True)
    pred_values = predictions.output.numpy()  # (N, 1, 7) → 7 quantiles

    # Quantile indices: 0=P5, 1=P10, 2=P25, 3=P50, 4=P75, 5=P90, 6=P95
    quantile_names = ["P05", "P10", "P25", "P50_median", "P75", "P90", "P95"]

    # ─── Map predictions back to constituencies ──────────────────────────
    # Get the group_id for each prediction sample
    # The prediction dataset iterates in order of the dataframe groups
    last_rows = df.groupby("group_id").last().reset_index()
    # Match predictions count  
    n_preds = pred_values.shape[0]
    print(f"Generated {n_preds} predictions for {last_rows.shape[0]} constituencies")

    # If counts don't match, limit to min
    n = min(n_preds, len(last_rows))
    
    results = []
    for i in range(n):
        row = last_rows.iloc[i]
        preds = pred_values[i, 0, :]  # 7 quantiles

        p10 = float(preds[1])
        p25 = float(preds[2])
        p50 = float(preds[3])
        p75 = float(preds[4])
        p90 = float(preds[5])

        outcome, confidence, spread = margin_to_outcome(p50, p10, p90)

        results.append({
            "state": row["state"],
            "constituency": row["constituency"],
            "district": row["district"],
            "last_election_year": int(row["year"]),
            "last_winner_party": row["winner_party"],
            "last_margin_pct": round(float(row["margin_pct"]), 2),
            "last_turnout_pct": round(float(row["turnout_pct"]), 2),

            # TFT Quantile Predictions for 2026
            "tft_P10": round(p10, 2),
            "tft_P25": round(p25, 2),
            "tft_P50_median": round(p50, 2),
            "tft_P75": round(p75, 2),
            "tft_P90": round(p90, 2),
            "tft_uncertainty_spread": round(spread, 2),

            # Derived outcome
            "tft_predicted_outcome": outcome,
            "tft_confidence": confidence,
        })

    results_df = pd.DataFrame(results)
    
    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TFT PREDICTION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total Predictions: {len(results_df)}")

    print(f"\n  ── State-wise TFT Median Predictions ──")
    for state in sorted(results_df["state"].unique()):
        s_df = results_df[results_df["state"] == state]
        print(f"    {state:<18} {len(s_df):>4} seats | "
              f"Median P50: {s_df['tft_P50_median'].mean():>5.1f}% | "
              f"Avg Uncertainty: {s_df['tft_uncertainty_spread'].mean():>5.1f}% | "
              f"Toss-ups: {(s_df['tft_predicted_outcome'].str.contains('TOSS')).sum()}")

    print(f"\n  ── Outcome Distribution ──")
    for outcome, cnt in results_df["tft_predicted_outcome"].value_counts().items():
        pct = cnt / len(results_df) * 100
        print(f"    {outcome:<30} {cnt:>4} seats ({pct:>5.1f}%)")

    print(f"\n  ── TOP 10 MOST COMPETITIVE (lowest predicted margin) ──")
    top10 = results_df.nsmallest(10, "tft_P50_median")
    print(f"  {'State':<15} {'Constituency':<25} {'P50':<8} {'P10-P90':<12} {'Outcome'}")
    print(f"  {'-'*75}")
    for _, r in top10.iterrows():
        print(f"  {r['state']:<15} {r['constituency']:<25} {r['tft_P50_median']:<8.1f} "
              f"{r['tft_P10']:.1f}-{r['tft_P90']:.1f}    {r['tft_predicted_outcome']}")

    # ─── Save ─────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "tft_predictions_2026.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  📊 Full predictions: {csv_path}")

    json_path = os.path.join(OUT_DIR, "tft_top_competitive_2026.json")
    results_df.nsmallest(30, "tft_P50_median").to_json(json_path, orient="records", indent=2)
    print(f"  🔥 Top 30 competitive: {json_path}")

    print(f"\n{'=' * 70}")
    print(f"  2026 TFT Forecasting Complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    generate_predictions()
