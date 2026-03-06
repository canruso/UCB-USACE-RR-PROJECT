"""Integrated Gradients attribution analysis for trained LSTM models.

Uses Captum to compute feature importance via Integrated Gradients.
Single-frequency (daily/hourly CudaLSTM) only — MTS not yet supported.

Typical usage
-------------
>>> from UCB_training.ucb_captum import run_ig_analysis, plot_feature_importance
>>> df = run_ig_analysis("outputs/hopland/daily_shared/runs/hopland_daily/testing_run_XXX")
>>> plot_feature_importance(df, title="Hopland Daily — No Physics")

Cached workflow (compute once, analyze many times)
--------------------------------------------------
>>> df, attrs = run_ig_analysis(run_dir, return_attributions=True)
>>> save_attributions(attrs, feature_names, "cache/hopland_np.pt", basin="hopland", experiment="BASELINE", phys_type="NP")
>>> attrs2, meta = load_attributions("cache/hopland_np.pt")
>>> df2 = rank_features(attrs2, meta["feature_names"])

Combined DataFrame for cross-run analysis
------------------------------------------
>>> results = {("hopland", "BASELINE", "NP"): df1, ("hopland", "NOBC", "NP"): df2, ...}
>>> combined = build_combined_df(results)
>>> plot_ig_heatmap(combined, basins=["hopland"], experiments=["BASELINE", "NOBC"], phys_type="NP")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo import get_model
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_model_for_ig(
    run_dir: Union[str, Path],
    epoch: Optional[int] = None,
    device: str = "cpu",
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[torch.nn.Module, Config, list[str]]:
    """Load a trained model ready for Integrated Gradients.

    Parameters
    ----------
    run_dir  : path to a single run directory containing config.yml and model_epoch*.pt
    epoch    : specific epoch to load (None = last available)
    device   : "cpu" or "cuda:0" etc.
    data_dir : override data_dir in saved config (for runs trained on a different machine)

    Returns
    -------
    model         : nn.Module in eval mode, weights loaded
    cfg           : Config object for this run
    feature_names : ordered list of dynamic input feature names
    """
    run_dir = Path(run_dir)
    cfg = Config(run_dir / "config.yml")

    if data_dir is not None:
        data_dir = Path(data_dir)
        cfg._cfg["data_dir"] = data_dir
        # Fix hardcoded physics_data_file paths from other machines
        phys_file = cfg._cfg.get("physics_data_file")
        if phys_file is not None and phys_file != "None" and not Path(str(phys_file)).exists():
            cfg._cfg["physics_data_file"] = data_dir / Path(str(phys_file)).name

    model = get_model(cfg).to(device)

    if epoch is None:
        # Prefer best checkpoint if available, fall back to last epoch
        best_file = run_dir / "model_best.pt"
        weight_file = best_file if best_file.is_file() else sorted(run_dir.glob("model_epoch*.pt"))[-1]
    else:
        weight_file = run_dir / f"model_epoch{epoch:03d}.pt"

    state = torch.load(weight_file, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    feature_names = list(cfg.dynamic_inputs)

    return model, cfg, feature_names


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------

def load_input_data(
    cfg: Config,
    run_dir: Union[str, Path],
    period: str = "validation",
    basin: Optional[str] = None,
    n_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load scaled input sequences from the dataset pipeline.

    Parameters
    ----------
    cfg       : Config from the run
    run_dir   : path to run directory (for scaler)
    period    : "validation" or "test"
    basin     : basin name (inferred from cfg if None)
    n_samples : number of sequences to return (None = all)

    Returns
    -------
    x_d   : tensor of shape [n_samples, seq_len, n_features]
    dates : array of shape [n_samples, seq_len] (datetime64)
    """
    run_dir = Path(run_dir)
    if basin is None:
        basin_attr = getattr(cfg, f"{period}_basin_file", None)
        basin = str(basin_attr).replace(".txt", "").strip() if basin_attr else None

    scaler = load_scaler(run_dir)
    ds = get_dataset(
        cfg=cfg,
        is_train=False,
        period=period,
        basin=basin,
        scaler=scaler,
    )

    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=ds.collate_fn)
    batch = next(iter(loader))

    x_d = batch["x_d"]      # [N, seq_len, features]
    dates = batch["date"]    # [N, seq_len] numpy datetime64

    if n_samples is not None and n_samples < x_d.shape[0]:
        # evenly spaced samples across the period
        indices = np.linspace(0, x_d.shape[0] - 1, n_samples, dtype=int)
        x_d = x_d[indices]
        dates = dates[indices]

    return x_d, dates


# ---------------------------------------------------------------------------
# 3. Integrated Gradients computation
# ---------------------------------------------------------------------------

def _make_captum_forward(model: torch.nn.Module):
    """Build a Captum-compatible forward function.

    Captum expects:  f(input_tensor) -> scalar or tensor
    Our model expects: model({'x_d': tensor}) -> {'y_hat': tensor}

    We return the mean prediction (scalar per sample) so IG produces
    per-feature attributions for the average output.
    """
    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        out = model({"x_d": x})
        y_hat = out["y_hat"]  # [batch, seq_len, n_targets]
        return y_hat.mean(dim=(1, 2))  # [batch] — mean prediction
    return forward_fn


def compute_ig(
    model: torch.nn.Module,
    x_data: torch.Tensor,
    baseline: str = "zero",
    n_steps: int = 50,
    internal_batch_size: int = 8,
) -> torch.Tensor:
    """Compute Integrated Gradients attributions.

    Parameters
    ----------
    model               : trained model in eval mode
    x_data              : input tensor [n_samples, seq_len, features]
    baseline            : "zero" (default) or "mean" (dataset mean as reference)
    n_steps             : integration steps (higher = more accurate, slower)
    internal_batch_size : batch size for IG integration steps

    Returns
    -------
    attributions : tensor [n_samples, seq_len, features]
    """
    from captum.attr import IntegratedGradients

    forward_fn = _make_captum_forward(model)

    if baseline == "zero":
        baselines = torch.zeros_like(x_data)
    elif baseline == "mean":
        baselines = x_data.mean(dim=0, keepdim=True).expand_as(x_data)
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    x_data = x_data.requires_grad_(True)

    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(
        x_data,
        baselines=baselines,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )

    return attributions.detach()


# ---------------------------------------------------------------------------
# 4. Ranking
# ---------------------------------------------------------------------------

def rank_features(
    attributions: torch.Tensor,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by mean absolute attribution.

    Parameters
    ----------
    attributions  : tensor [n_samples, seq_len, features]
    feature_names : list of feature names (length = features dim)

    Returns
    -------
    DataFrame with columns: feature, mean_abs_attr, rank
        sorted by mean_abs_attr descending
    """
    # mean over samples and time steps → [features]
    mean_abs = attributions.abs().mean(dim=(0, 1)).numpy()

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_attr": mean_abs,
    })
    df = df.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# 4b. Attribution caching
# ---------------------------------------------------------------------------

def save_attributions(
    attributions: torch.Tensor,
    feature_names: list[str],
    path: Union[str, Path],
    **metadata,
) -> Path:
    """Save raw attributions tensor + metadata to disk.

    Parameters
    ----------
    attributions  : tensor [n_samples, seq_len, features]
    feature_names : ordered feature names
    path          : output .pt file path
    **metadata    : extra keys stored alongside (basin, experiment, phys_type, etc.)

    Returns
    -------
    Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attributions": attributions,
        "feature_names": feature_names,
        **metadata,
    }
    torch.save(payload, path)
    return path


def load_attributions(
    path: Union[str, Path],
) -> Tuple[torch.Tensor, dict]:
    """Load cached attributions from disk.

    Returns
    -------
    attributions : tensor [n_samples, seq_len, features]
    metadata     : dict with at least 'feature_names', plus any extras from save
    """
    payload = torch.load(Path(path), map_location="cpu")
    attributions = payload.pop("attributions")
    return attributions, payload


# ---------------------------------------------------------------------------
# 4c. Combined DataFrame builder
# ---------------------------------------------------------------------------

def build_combined_df(
    results: dict[tuple, pd.DataFrame],
) -> pd.DataFrame:
    """Tag and concatenate ranked DataFrames from multiple runs.

    Parameters
    ----------
    results : dict mapping (basin, experiment, phys_type) -> DataFrame from rank_features()

    Returns
    -------
    Combined DataFrame with columns: feature, mean_abs_attr, rank, basin, experiment, phys_type
    """
    frames = []
    for (basin, experiment, phys_type), df in results.items():
        tagged = df.copy()
        tagged["basin"] = basin
        tagged["experiment"] = experiment
        tagged["phys_type"] = phys_type
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_feature_importance(
    df: pd.DataFrame,
    title: Optional[str] = None,
    top_n: Optional[int] = None,
    ax=None,
    figsize: Tuple[int, int] = (12, 8),
):
    """Horizontal bar chart of feature importance.

    Parameters
    ----------
    df     : DataFrame from rank_features()
    title  : plot title
    top_n  : only show top N features (None = all)
    ax     : matplotlib Axes (creates new figure if None)
    figsize: figure size if creating new figure
    """
    import matplotlib.pyplot as plt

    plot_df = df.head(top_n) if top_n else df
    plot_df = plot_df.iloc[::-1]  # reverse for bottom-to-top

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n = len(plot_df)
    colors = ["steelblue"] * n
    # top 3 in red shades (they're at the end after reversal)
    red_shades = ["#CD5C5C", "#B22222", "#8B0000"]
    for i, c in enumerate(red_shades[:min(3, n)]):
        colors[n - 1 - i] = c

    ax.barh(range(n), plot_df["mean_abs_attr"].values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=8)
    ax.set_xlabel("Mean |Attribution|", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # value labels on top 10
    vals = plot_df["mean_abs_attr"].values
    for i in range(max(0, n - 10), n):
        ax.text(vals[i], i, f" {vals[i]:.4f}", va="center", fontsize=7)

    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# 6. Convenience wrapper
# ---------------------------------------------------------------------------

def run_ig_analysis(
    run_dir: Union[str, Path],
    period: str = "validation",
    n_steps: int = 50,
    n_samples: Optional[int] = 50,
    baseline: str = "zero",
    device: str = "cpu",
    epoch: Optional[int] = None,
    data_dir: Optional[Union[str, Path]] = None,
    return_attributions: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, torch.Tensor, list[str]]]:
    """End-to-end IG analysis: load model + data, compute attributions, rank.

    Parameters
    ----------
    run_dir              : path to trained run directory
    period               : "validation" or "test"
    n_steps              : IG integration steps
    n_samples            : number of input sequences to average over (None = all, 50 is reasonable)
    baseline             : "zero" or "mean"
    device               : torch device
    epoch                : model epoch to load (None = last)
    data_dir             : override data_dir in saved config (for runs trained on a different machine)
    return_attributions  : if True, also return raw attribution tensor and feature names

    Returns
    -------
    If return_attributions=False: DataFrame from rank_features()
    If return_attributions=True:  (DataFrame, attributions tensor, feature_names list)
    """
    model, cfg, feature_names = load_model_for_ig(run_dir, epoch=epoch, device=device, data_dir=data_dir)
    x_data, dates = load_input_data(cfg, run_dir, period=period, n_samples=n_samples)
    x_data = x_data.to(device)

    attributions = compute_ig(
        model, x_data, baseline=baseline, n_steps=n_steps,
    )

    df = rank_features(attributions, feature_names)

    if return_attributions:
        return df, attributions, feature_names
    return df


# ---------------------------------------------------------------------------
# 7. Comparison plotting
# ---------------------------------------------------------------------------

def plot_ig_side_by_side(
    dfs: list[pd.DataFrame],
    labels: list[str],
    top_n: int = 15,
    figsize: Tuple[int, int] = (16, 7),
    title: Optional[str] = None,
):
    """Side-by-side horizontal bar charts comparing feature importance across runs.

    Parameters
    ----------
    dfs    : list of DataFrames from rank_features()
    labels : display labels for each DataFrame (same length as dfs)
    top_n  : show top N features per panel
    figsize: figure size
    title  : overall suptitle
    """
    import matplotlib.pyplot as plt

    n_panels = len(dfs)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=False)
    if n_panels == 1:
        axes = [axes]

    palette = ["#2196F3", "#FF6D00", "#4CAF50", "#9C27B0"]

    for idx, (df, label) in enumerate(zip(dfs, labels)):
        ax = axes[idx]
        plot_df = df.head(top_n).iloc[::-1]
        n = len(plot_df)
        color = palette[idx % len(palette)]

        ax.barh(range(n), plot_df["mean_abs_attr"].values, color=color, alpha=0.85,
                edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(n))
        ax.set_yticklabels(plot_df["feature"].values, fontsize=7)
        ax.set_xlabel("Mean |Attribution|", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        vals = plot_df["mean_abs_attr"].values
        for i in range(max(0, n - 5), n):
            ax.text(vals[i], i, f" {vals[i]:.4f}", va="center", fontsize=6)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_ig_rank_shift(
    df_base: pd.DataFrame,
    df_compare: pd.DataFrame,
    label_base: str = "BASELINE",
    label_compare: str = "NOBC",
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
):
    """Rank-shift (bump chart) showing how feature ranks change between experiments.

    Parameters
    ----------
    df_base    : DataFrame from rank_features() for the base experiment
    df_compare : DataFrame from rank_features() for the comparison experiment
    label_base, label_compare : display labels for left/right columns
    top_n      : include features that rank in top N of either experiment
    title      : plot title
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    # Union of top-N features from both experiments
    top_base = set(df_base.head(top_n)["feature"])
    top_comp = set(df_compare.head(top_n)["feature"])
    features = top_base | top_comp

    rank_base = df_base.set_index("feature")["rank"]
    rank_comp = df_compare.set_index("feature")["rank"]

    rows = []
    for f in features:
        rb = int(rank_base[f]) if f in rank_base.index else len(df_base) + 1
        rc = int(rank_comp[f]) if f in rank_comp.index else len(df_compare) + 1
        rows.append({"feature": f, "rank_base": rb, "rank_comp": rc, "shift": rb - rc})
    shift_df = pd.DataFrame(rows).sort_values("rank_base")

    fig, ax = plt.subplots(figsize=figsize)
    n = len(shift_df)
    max_rank = max(shift_df["rank_base"].max(), shift_df["rank_comp"].max())

    for _, row in shift_df.iterrows():
        rb, rc = row["rank_base"], row["rank_comp"]
        if row["shift"] > 0:
            color = "#4CAF50"  # improved (lower rank number = more important)
        elif row["shift"] < 0:
            color = "#F44336"  # worsened
        else:
            color = "#9E9E9E"
        ax.plot([0, 1], [rb, rc], color=color, alpha=0.6, linewidth=1.5)
        ax.plot(0, rb, "o", color=color, markersize=5)
        ax.plot(1, rc, "o", color=color, markersize=5)

    # Labels on left and right
    for _, row in shift_df.iterrows():
        ax.text(-0.02, row["rank_base"], row["feature"], ha="right", va="center", fontsize=6.5)
        ax.text(1.02, row["rank_comp"], row["feature"], ha="left", va="center", fontsize=6.5)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(max_rank + 1, 0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_base, label_compare], fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature Rank", fontsize=11)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    return fig, ax, shift_df


def plot_ig_heatmap(
    ig_df: pd.DataFrame,
    basins: list[str],
    experiments: list[str],
    phys_type: str = "NP",
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
):
    """Heatmap of normalized feature attributions across basins × experiments.

    Parameters
    ----------
    ig_df       : combined DataFrame with columns [feature, mean_abs_attr, rank, basin, experiment, phys_type]
    basins      : list of basin names to include
    experiments : list of experiment labels to include
    phys_type   : "NP" or "PHYS"
    top_n       : show top N features (union across all combos)
    title       : plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = ig_df[(ig_df["phys_type"] == phys_type) &
                (ig_df["basin"].isin(basins)) &
                (ig_df["experiment"].isin(experiments))].copy()

    # Normalize within each (basin, experiment) combo: fraction of total attribution
    sub["norm_attr"] = sub.groupby(["basin", "experiment"])["mean_abs_attr"].transform(
        lambda x: x / x.sum()
    )

    # Get top-N features by max normalized attribution across any combo
    top_feats = (sub.groupby("feature")["norm_attr"].max()
                 .nlargest(top_n).index.tolist())
    sub = sub[sub["feature"].isin(top_feats)]

    # Pivot for heatmap: features × (basin, experiment)
    sub["col"] = sub["basin"].str.title() + "\n" + sub["experiment"]
    pivot = sub.pivot_table(index="feature", columns="col", values="norm_attr", aggfunc="first")

    # Order features by mean across columns (descending)
    feat_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[feat_order]

    fig, ax = plt.subplots(figsize=figsize)
    annot = pivot.map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    sns.heatmap(pivot.fillna(0), annot=annot, fmt="", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Normalized Attribution"})
    ax.set_ylabel("")
    ax.set_xlabel("")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    return fig, ax
