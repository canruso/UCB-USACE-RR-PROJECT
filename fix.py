import pandas as pd
from pathlib import Path

from UCB_training.UCB_utils import save_hparams, _artifact_root

# ================================
# CONFIG
# ================================

BASIN = "guerneville"
MODE = "mts"
RUN_LABEL = "CROSS_VAL_V4"
RUN_STAMP = "20260307T122221Z"

# ================================
# Locate CSVs
# ================================

root = _artifact_root(BASIN, MODE)
archive = root / "hyperparams" / "archive"

prefix = f"{BASIN}_{MODE}_{RUN_LABEL}"

csv_no_phys = archive / f"{prefix}_no_physics_gridsearch_{RUN_STAMP}.csv"
csv_phys = archive / f"{prefix}_physics_gridsearch_{RUN_STAMP}.csv"

print("Loading:")
print(csv_no_phys)
print(csv_phys)

df_no = pd.read_csv(csv_no_phys)
df_phys = pd.read_csv(csv_phys)

print(f"\nLoaded {len(df_no)} no-physics trials")
print(f"Loaded {len(df_phys)} physics trials")

# ================================
# Find best configs
# ================================

df_no = df_no.sort_values("_rank_score", ascending=False).reset_index(drop=True)
df_phys = df_phys.sort_values("_rank_score", ascending=False).reset_index(drop=True)

best_no = df_no.iloc[0].copy()
best_phys = df_phys.iloc[0].copy()

best_no["model_type"] = "no_physics"
best_phys["model_type"] = "physics"

best_params_df = pd.DataFrame([best_no, best_phys]).reset_index(drop=True)

print("\nBEST NO-PHYSICS:")
print(best_no)

print("\nBEST PHYSICS:")
print(best_phys)

# ================================
# Save final hyperparameter file
# ================================

save_hparams(
    best_df=best_params_df,
    basin=BASIN,
    mode=MODE,
    label=RUN_LABEL,
    run_stamp=RUN_STAMP,
    df_no=df_no,
    df_phys=df_phys,
)

print("\nSaved final hyperparameters.")