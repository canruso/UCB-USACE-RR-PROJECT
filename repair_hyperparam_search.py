import pandas as pd
from pathlib import Path
from UCB_training.UCB_utils import save_hparams, _artifact_root

BASIN = "warm_springs"
MODE = "mts"
RUN_LABEL = "EXTREME_YEARS"
RUN_STAMP = "20260307T122221Z"

root = _artifact_root(BASIN, MODE)
arch = root / "hyperparams" / "archive"

prefix = f"{BASIN}_{MODE}_{RUN_LABEL}"

df_no = pd.read_csv(arch / f"{prefix}_no_physics_gridsearch_{RUN_STAMP}.csv")
df_phys = pd.read_csv(arch / f"{prefix}_physics_gridsearch_{RUN_STAMP}.csv")

best_no = df_no.sort_values("_rank_score", ascending=False).iloc[0]
best_phys = df_phys.sort_values("_rank_score", ascending=False).iloc[0]

best_no["model_type"] = "no_physics"
best_phys["model_type"] = "physics"

best_df = pd.DataFrame([best_no, best_phys])

save_hparams(
    best_df=best_df,
    basin=BASIN,
    mode=MODE,
    label=RUN_LABEL,
    run_stamp=RUN_STAMP,
    df_no=df_no,
    df_phys=df_phys
)

print("Final hyperparameters saved.")