"""
TFT Model Training Engine (Real Historical Data)
==================================================
Trains PyTorch Forecasting's Temporal Fusion Transformer on REAL
TCPD election data spanning 1951–2021 across 5 Indian states.

The model learns:
  "Given a constituency's historical election pattern (margins, turnout,
   party dynamics, competitiveness), predict the winning margin of the 
   NEXT election."

Output: Multi-horizon quantile predictions (P10, P25, P50, P75, P90)
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available.*")

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tft_historical_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_and_prepare() -> tuple:
    """Loads the historical dataset and creates PyTorch Forecasting datasets."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df['group_id'].nunique()} constituencies")

    # Ensure types
    df["time_idx"] = df["time_idx"].astype(int)
    for col in ["state", "constituency", "district", "constituency_type", "winner_party", "group_id"]:
        df[col] = df[col].astype(str)

    # Clamp target to reasonable range
    df["target_margin"] = df["target_margin"].clip(0.0, 60.0)

    # ─── Determine encoder/prediction lengths ────────────────────────────
    # Most constituencies have ~8 elections. Use last 2 as prediction window.
    max_time = df.groupby("group_id")["time_idx"].max()
    
    # Only keep constituencies with enough history (>= 4 elections for 2 encoder + 1 pred + 1 buffer)
    valid_groups = max_time[max_time >= 3].index
    df = df[df["group_id"].isin(valid_groups)].reset_index(drop=True)
    print(f"After filtering (>= 4 elections): {df['group_id'].nunique()} constituencies, {len(df):,} rows")

    max_encoder_length = 6   # Look back up to 6 past elections
    max_prediction_length = 1  # Predict 1 election ahead

    # Limit top-N parties to keep categorical embedding manageable
    top_parties = df["winner_party"].value_counts().head(20).index.tolist()
    df["winner_party"] = df["winner_party"].where(df["winner_party"].isin(top_parties), "OTHER")

    # ─── Training / Validation split ─────────────────────────────────────
    # Use the last time step of each group as validation
    training_cutoff = df.groupby("group_id")["time_idx"].transform("max") - max_prediction_length
    train_df = df[df["time_idx"] <= training_cutoff].copy()
    
    print(f"Training set: {len(train_df):,} rows")
    print(f"Target range: [{train_df['target_margin'].min():.1f}, {train_df['target_margin'].max():.1f}]")

    # ─── Build TimeSeriesDataSet ─────────────────────────────────────────
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

        target_normalizer=GroupNormalizer(
            groups=["group_id"],
            transformation="softplus",
        ),
        
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create validation dataset from same data structure
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    return training, validation, df


def train_tft():
    """Trains the TFT and saves the best checkpoint."""
    print("=" * 70)
    print("  TFT MODEL TRAINING — Indian Assembly Elections")
    print("=" * 70)

    training, validation, df = load_and_prepare()

    # Dataloaders
    batch_size = 64
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ─── Callbacks ────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-4,
        mode="min",
    )
    
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-election-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # ─── Trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=10,
    )

    # ─── TFT Model ────────────────────────────────────────────────────────
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.15,
        hidden_continuous_size=16,
        output_size=7,           # 7 quantiles: P5, P10, P25, P50, P75, P90, P95
        loss=QuantileLoss(),
        reduce_on_plateau_patience=3,
        log_interval=5,
    )

    n_params = sum(p.numel() for p in tft.parameters())
    print(f"\nModel parameters: {n_params:,} ({n_params/1e3:.1f}k)")

    # ─── Train ────────────────────────────────────────────────────────────
    print("\n--- Starting Training ---")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("--- Training Complete ---")

    # ─── Load best model ──────────────────────────────────────────────────
    best_model_path = checkpoint.best_model_path
    print(f"\nBest model checkpoint: {best_model_path}")
    print(f"Best val_loss: {checkpoint.best_model_score:.4f}")

    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # ─── Generate Predictions ─────────────────────────────────────────────
    print("\n--- Generating Predictions on Validation Set ---")
    predictions = best_tft.predict(val_loader, mode="quantiles", return_x=True)
    
    pred_values = predictions.output
    print(f"Prediction shape: {pred_values.shape}")
    print(f"  (samples × prediction_horizon × quantiles)")

    # Extract median (P50) predictions
    median_idx = 3  # Index 3 = P50 in 7-quantile output
    medians = pred_values[:, 0, median_idx].numpy()  # First prediction step

    print(f"\n── Prediction Summary (P50 / Median) ──")
    print(f"  Count:  {len(medians)}")
    print(f"  Mean:   {np.mean(medians):.2f}%")
    print(f"  Median: {np.median(medians):.2f}%")
    print(f"  Std:    {np.std(medians):.2f}%")
    print(f"  Min:    {np.min(medians):.2f}%")
    print(f"  Max:    {np.max(medians):.2f}%")

    # ─── Feature Importance (TFT's built-in interpretability) ─────────────
    print("\n--- Variable Importance (TFT Attention Weights) ---")
    try:
        interpretation = best_tft.interpret_output(predictions.output, reduction="sum")
        for key in ["encoder_variables", "decoder_variables", "static_variables"]:
            if key in interpretation:
                print(f"\n  {key}:")
                imp = interpretation[key]
                if hasattr(imp, 'items'):
                    for var, weight in sorted(imp.items(), key=lambda x: -x[1]):
                        bar = "█" * int(weight * 40)
                        print(f"    {var:<25} {bar} ({weight:.3f})")
    except Exception as e:
        print(f"  (Interpretation not available: {e})")

    print(f"\n{'=' * 70}")
    print(f"  TFT Training Complete!")
    print(f"  Best checkpoint: {best_model_path}")
    print(f"  Val Loss: {checkpoint.best_model_score:.4f}")
    print(f"{'=' * 70}")

    return best_tft


if __name__ == "__main__":
    train_tft()
