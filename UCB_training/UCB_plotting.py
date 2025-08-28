# UCB_training/UCB_plotting.py
from __future__ import annotations

from typing import Sequence, Literal, Union, List
from pathlib import Path

import json
import numpy as np
import pandas as pd
import xarray as xr

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.dates import DateFormatter

from UCB_training.UCB_utils import (prepare_out_path, ensure_output_tree, set_active_context, data_dir, repo_root,
                                    clean_df)

_BASINS = ("calpella", "guerneville", "hopland", "warm_springs")
_ORDER_FIXED = ("Calpella", "Hopland", "Warm Springs", "Guerneville")


def plot_timeseries_comparison(source: Union[pd.DataFrame, Sequence[Path] | Sequence[str]], title: str, *,
                               start_date: str | None = None, end_date: str | None = None,
                               metrics: list[str] | None = None, metrics_out: str = "metrics.csv",
                               ts_out: str | None = None, fig_out: str | None = None,
                               backend: Literal["mpl", "plotly"] = "mpl", figsize: tuple[int, int] = (30, 10),
                               height: int = 600, legend_font: int = 18, axis_font: int = 20, linewidth: float = 2.0,
                               alpha_pred: float = 0.75, show: bool = True) -> pd.DataFrame:
    """
    Accept either a pre-merged DataFrame or a 3‑tuple of CSV paths:
      (lstm_csv, pilstm_csv, hms_raw_csv)

    The two LSTM CSVs are expected to have: Date, Observed, Predicted
    The HMS raw CSV is HEC-style; it is cleaned and the first data column is used.
    Outputs (if provided) are routed via prepare_out_path.
    """
    from neuralhydrology.evaluation.metrics import calculate_all_metrics

    m_path = prepare_out_path(metrics_out, kind="metrics") if metrics_out else None
    t_path = prepare_out_path(ts_out, kind="csv") if ts_out else None
    f_path = prepare_out_path(fig_out, kind="plot_timeseries") if fig_out else None

    if isinstance(source, pd.DataFrame):
        df = source.copy()

    else:
        lstm_p, pilstm_p, hms_p = map(Path, source)

        lstm_df = pd.read_csv(lstm_p).rename(columns={'Predicted': 'LSTM_Predicted'})
        pilstm_df = pd.read_csv(pilstm_p).rename(columns={'Predicted': 'PLSTM_Predicted'})
        pilstm_df.drop(columns=['Observed'], errors='ignore', inplace=True)

        hms_raw = pd.read_csv(hms_p)
        hms_df = clean_df(hms_raw).reset_index().rename(columns={'date': 'Date'})
        non_date_cols = [c for c in hms_df.columns if c != "Date"]

        if not non_date_cols:
            raise ValueError("HMS CSV appears empty after cleaning.")
        hms_df = hms_df.rename(columns={non_date_cols[0]: 'HMS_Predicted'})

        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"], errors="coerce")
        pilstm_df["Date"] = pd.to_datetime(pilstm_df["Date"], errors="coerce")
        hms_df["Date"] = pd.to_datetime(hms_df["Date"], errors="coerce")

        df = lstm_df.merge(hms_df, on="Date", how="inner").merge(pilstm_df, on="Date", how="inner")

    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col.endswith("_Predicted"):
                df.loc[df[col] < 0, col] = 0

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    if t_path:
        t_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(t_path, index=False)
        if ts_out and Path(ts_out).name != t_path.name:
            Path(ts_out).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(ts_out), index=False)

    to_da = lambda c: xr.DataArray(df[c].values, dims=["date"], coords={"date": df["Date"]})
    obs_da, hms_da, lstm_da, pilstm_da = map(to_da, ["Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"])

    big = {"HMS": calculate_all_metrics(obs_da, hms_da), "LSTM": calculate_all_metrics(obs_da, lstm_da),
           "PILSTM": calculate_all_metrics(obs_da, pilstm_da)}

    for name, da in zip(big, [hms_da, lstm_da, pilstm_da]):
        big[name]["PBIAS"] = ((obs_da - da).sum() / obs_da.sum() * 100).item()

    if m_path:
        m_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(big).to_csv(m_path)
        if metrics_out and Path(metrics_out).name != m_path.name:
            Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(big).to_csv(Path(metrics_out))

    if metrics:
        for mdl, vals in big.items():
            print(mdl)
            for m in metrics:
                if m in vals:
                    print(f"  {m} = {vals[m]:.3f}")

    def _fmt(model):
        return ", ".join(f"{m}={big[model][m]:.3f}" for m in metrics) if metrics else ""

    labels = {"HMS": "HMS Prediction", "LSTM": "LSTM Prediction", "PILSTM": "Physics Informed LSTM"}
    if metrics:
        for k in labels:
            labels[k] += f" ({_fmt(k)})"

    if backend == "plotly":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["Observed"], mode="lines", name="Observed", line=dict(width=linewidth))
        )
        for col, key in [("HMS_Predicted", "HMS"), ("LSTM_Predicted", "LSTM"), ("PLSTM_Predicted", "PILSTM")]:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=labels[key], opacity=alpha_pred,
                                     line=dict(width=linewidth)))

        fig.update_layout(title=title, height=height, xaxis_title="Date", yaxis_title="Inflow(cfs)", template="seaborn",
                          hovermode="x unified",
                          legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)",
                                      font=dict(size=legend_font)),
                          xaxis=dict(rangeslider=dict(visible=True), type="date"), margin=dict(r=300))

        if f_path:
            try:
                f_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_image(str(f_path), scale=2)
            except Exception:
                fig.write_html(str(f_path.with_suffix(".html")))
        if show:
            fig.show()
    else:
        plt.figure(figsize=figsize)
        plt.rcParams.update(
            {"axes.labelsize": axis_font, "xtick.labelsize": axis_font - 2, "ytick.labelsize": axis_font - 2,
             "legend.fontsize": legend_font})

        plt.plot(df["Date"], df["Observed"], lw=linewidth, label="Observed")
        plt.plot(df["Date"], df["HMS_Predicted"], lw=linewidth, alpha=alpha_pred, label=labels["HMS"])
        plt.plot(df["Date"], df["LSTM_Predicted"], lw=linewidth, alpha=alpha_pred, label=labels["LSTM"])
        plt.plot(df["Date"], df["PLSTM_Predicted"], lw=linewidth, alpha=alpha_pred, label=labels["PILSTM"])
        plt.title(title, fontsize=axis_font + 10)
        plt.xlabel("Date")
        plt.ylabel("Inflow (cfs)")
        plt.grid(alpha=.4)
        plt.legend(loc="upper right")
        plt.tight_layout()

        if f_path:
            f_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(f_path, dpi=600, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    return pd.DataFrame(big)


def plot_timeseries_panel(ax, sub_df: pd.DataFrame, *, legend_font: int, legend_boxpad: float, axis_font: int,
                          date_fmt: str, row_title: str | None = None, joint_metrics: bool = True) -> None:
    date = pd.to_datetime(sub_df["Date"])
    specs = [("HMS_Predicted", "HMS Prediction", 0.70), ("LSTM_Predicted", "LSTM Prediction", 0.80),
             ("PLSTM_Predicted", "Physics Informed LSTM", 0.70)]

    if joint_metrics:
        metric_cols = ["Observed"] + [c for c, _, _ in specs if c in sub_df]
        metric_df = sub_df[metric_cols].dropna(how="any")

    obs_full = sub_df["Observed"].to_numpy(float)
    ax.plot(date, obs_full, lw=2, label="Observed")

    for col, lbl, alpha in specs:
        if col not in sub_df or sub_df[col].isna().all():
            continue

        if joint_metrics:
            obs = metric_df["Observed"].to_numpy(float)
            sim = metric_df[col].to_numpy(float)
        else:
            obs = obs_full
            sim = sub_df[col].to_numpy(float)

        nse, pb = _nse(obs, sim), _pbias(obs, sim)
        ax.plot(date, sub_df[col].to_numpy(float), lw=2, alpha=alpha, label=f"{lbl} (NSE={nse:.3f}, PBIAS={pb:.2f}%)")

    ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
    ax.set_ylabel("Inflow [cfs]", fontsize=axis_font)
    ax.tick_params(axis="both", labelsize=axis_font - 2)
    ax.grid(alpha=.3)
    ax.legend(fontsize=legend_font, fancybox=True, framealpha=0.9, borderpad=legend_boxpad)

    if row_title:
        ax.set_title(row_title, fontsize=axis_font + 1, pad=6)


def ts_triptych_v3(df, *, wet_start: str, wet_end: str, dry_start: str, dry_end: str, save_path: str | Path,
                   legend_font: int = 26, legend_boxpad: float = 0.5, axis_font: int = 18, date_fmt: str = "%d-%b-%Y",
                   figsize: tuple[int, int] = (12, 10), dpi: int = 600,
                   hspace: float = 0.02, row_titles: tuple[str, str, str] | None = None, main_title: str | None = None,
                   main_title_font: int = 26, main_title_y: float = 0.98, main_title_pad: float = 0.03) -> None:
    """
    Render three vertically-stacked panels containing the entire, period, wettest water-year, driest water-year
     Df may be a pre-merged DataFrame with columns [Date, Observed, HMS_Predicted, LSTM_Predicted, PLSTM_Predicted]
    OR a 3-tuple of CSV paths: (lstm_csv, pilstm_csv, hms_raw_csv).
    """

    if isinstance(df, (list, tuple)) and len(df) == 3:
        lstm_p, pilstm_p, hms_p = map(Path, df)

        lstm_df = pd.read_csv(lstm_p).rename(columns={'Predicted': 'LSTM_Predicted'})
        pilstm_df = pd.read_csv(pilstm_p).rename(columns={'Predicted': 'PLSTM_Predicted'})
        pilstm_df.drop(columns=['Observed'], errors='ignore', inplace=True)

        hms_raw = pd.read_csv(hms_p)
        hms_df = clean_df(hms_raw).reset_index().rename(columns={'date': 'Date'})
        non_date_cols = [c for c in hms_df.columns if c != "Date"]
        if not non_date_cols:
            raise ValueError("HMS CSV appears empty after cleaning.")
        hms_df = hms_df.rename(columns={non_date_cols[0]: 'HMS_Predicted'})

        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"], errors="coerce")
        pilstm_df["Date"] = pd.to_datetime(pilstm_df["Date"], errors="coerce")
        hms_df["Date"] = pd.to_datetime(hms_df["Date"], errors="coerce")

        df = lstm_df.merge(hms_df, on="Date", how="inner").merge(pilstm_df, on="Date", how="inner")
    else:
        df = df.copy()

    if df["Date"].dtype != "datetime64[ns]":
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    wet_df = df[(df["Date"] >= wet_start) & (df["Date"] <= wet_end)]
    dry_df = df[(df["Date"] >= dry_start) & (df["Date"] <= dry_end)]

    row_titles = row_titles or (None, None, None)
    if len(row_titles) != 3:
        raise ValueError("row_titles must be a 3-tuple (whole, wet, dry)")

    _rc = plt.rcParams.copy()
    plt.rcParams.update({"legend.fontsize": legend_font, "legend.borderpad": legend_boxpad, "axes.labelsize": axis_font,
                         "xtick.labelsize": axis_font - 2, "ytick.labelsize": axis_font - 2})

    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)

    if main_title:
        fig.suptitle(main_title, fontsize=main_title_font, y=main_title_y)

    plot_timeseries_panel(axes[0], df, legend_font=legend_font, legend_boxpad=legend_boxpad, axis_font=axis_font,
                          date_fmt=date_fmt, row_title=row_titles[0])

    plot_timeseries_panel(axes[1], wet_df, legend_font=legend_font, legend_boxpad=legend_boxpad, axis_font=axis_font,
                          date_fmt=date_fmt, row_title=row_titles[1])

    plot_timeseries_panel(axes[2], dry_df, legend_font=legend_font, legend_boxpad=legend_boxpad, axis_font=axis_font,
                          date_fmt=date_fmt, row_title=row_titles[2])

    axes[2].set_xlabel("Date", fontsize=axis_font)

    top_margin = (main_title_y - main_title_pad) if main_title else 0.995
    fig.subplots_adjust(hspace=hspace, top=top_margin, bottom=0.002, left=0.07, right=0.995)

    save_path = prepare_out_path(save_path, kind="plot_triptych")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    plt.rcParams.update(_rc)


def _ascii_hyphens(txt: str | None) -> str | None:
    if txt is None:
        return None
    return txt.translate({c: 0x2D for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015)})


def _scatter_row(df, outfile: str | Path, *, resolution: str = "hourly", layout: str = "horizontal",
                 square_side: float = 4.0, point_size: float = 28, legend_font: int = 14, axis_font: int = 14,
                 panel_title_pad: float = 6, top_pad: float = .93, bottom_pad: float = .08, inter_pad: float = .02,
                 suptitle_y: float | None = None, dpi: int = 600, plot_title: str | None = None):
    resolution = resolution.lower().strip()
    prefix = "MTS-" if resolution.startswith("mts_") else ""
    lbl_lstm = f"{prefix}LSTM"
    lbl_pilstm = f"{prefix}PILSTM"

    specs = [("HMS_Predicted", "HMS Predicted Flow"), ("LSTM_Predicted", f"{lbl_lstm} Predicted Flow"),
             ("PLSTM_Predicted", f"{lbl_pilstm} Predicted Flow")]

    specs = [(c, _ascii_hyphens(l)) for c, l in specs if c in df.columns]
    if not specs:
        raise ValueError("No recognised prediction columns in DataFrame.")

    n_pan = len(specs)
    layout = layout.lower()

    if layout == "vertical":
        nrows, ncols = n_pan, 1
        figsize = (square_side, square_side * n_pan)
    else:
        nrows, ncols = 1, n_pan
        figsize = (square_side * n_pan, square_side)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(left=.06, right=.98, top=top_pad, bottom=bottom_pad, wspace=inter_pad, hspace=inter_pad)

    if plot_title:
        fig.suptitle(_ascii_hyphens(plot_title),
                     fontsize=axis_font + 2, y=suptitle_y if suptitle_y is not None else min(top_pad + .03, .98))

    c_sim = "#e41a1c"
    h_sim = Line2D([], [], ls="", marker="o", color=c_sim, label="Predicted")

    for idx, (ax, (col, title)) in enumerate(zip(axes, specs)):
        o = df["Observed"].astype(float).to_numpy()
        p = df[col].astype(float).to_numpy()
        keep = np.isfinite(o) & np.isfinite(p)
        o, p = o[keep], p[keep]

        ax.scatter(o, p, s=point_size, c=c_sim, alpha=.70)

        lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
        if lo == hi:
            hi += 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        if layout == "vertical":
            if idx == n_pan - 1:
                ax.set_xlabel("Observed flow [cfs]", fontsize=axis_font)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            ax.set_ylabel("Predicted flow [cfs]", fontsize=axis_font)
        else:
            ax.set_xlabel("Observed flow [cfs]", fontsize=axis_font)
            if idx == 0:
                ax.set_ylabel("Predicted flow [cfs]", fontsize=axis_font)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

        if idx == 0:
            ax.legend(handles=[h_sim], loc="upper left", fontsize=legend_font, framealpha=.9, borderpad=.4)

        ax.set_title(title, fontsize=axis_font + 1, pad=panel_title_pad)

        nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        pb = 100.0 * (o - p).sum() / o.sum()
        ax.text(0.98, 0.02, f"NSE = {nse:.3f}\nPBIAS = {pb:.2f}%", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=legend_font, bbox=dict(fc="white", ec="0.7", alpha=.9, pad=.35))

        ax.tick_params(labelsize=axis_font - 2)

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def scatter_triptych_pngs_v3(df, *, wet_start: str, wet_end: str, dry_start: str, dry_end: str,
                             out_dir: str | Path = "scatter_pngs", resolution: str = "hourly",
                             layout: str = "horizontal", square_side: float = 4.0, point_size: float = 28,
                             legend_font: int = 14, axis_font: int = 14, panel_title_pad: float = 6,
                             top_pad: float = .93, bottom_pad: float = .08, inter_pad: float = .02,
                             suptitle_y: float | None = None, dpi: int = 600,
                             row_titles: tuple[str, str, str] | None = None):
    if isinstance(df, (list, tuple)) and len(df) == 3:
        lstm_p, pilstm_p, hms_p = map(Path, df)
        lstm_df = pd.read_csv(lstm_p).rename(columns={'Predicted': 'LSTM_Predicted'})
        pilstm_df = pd.read_csv(pilstm_p).rename(columns={'Predicted': 'PLSTM_Predicted'})
        pilstm_df.drop(columns=['Observed'], errors='ignore', inplace=True)
        hms_raw = pd.read_csv(hms_p)
        hms_df = clean_df(hms_raw).reset_index().rename(columns={'date': 'Date'})
        non_date_cols = [c for c in hms_df.columns if c != "Date"]

        if not non_date_cols:
            raise ValueError("HMS CSV appears empty after cleaning.")
        hms_df = hms_df.rename(columns={non_date_cols[0]: 'HMS_Predicted'})

        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"], errors="coerce")
        pilstm_df["Date"] = pd.to_datetime(pilstm_df["Date"], errors="coerce")
        hms_df["Date"] = pd.to_datetime(hms_df["Date"], errors="coerce")

        df = lstm_df.merge(hms_df, on="Date", how="inner").merge(pilstm_df, on="Date", how="inner")

    else:
        df = df.copy()

    if df["Date"].dtype != "datetime64[ns]":
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    keep = df[["Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"]].notna().any(axis=1)
    df = df[keep]

    if row_titles is None:
        row_titles = ("Whole period", "Wettest year", "Driest year")
    if len(row_titles) != 3:
        raise ValueError("row_titles must be a 3-tuple")

    prefix = str(out_dir)
    files: list[Path] = []

    def _write(sub_df: pd.DataFrame, suffix: str, title: str):
        routed = prepare_out_path(f"{prefix}_{suffix}", kind="plot_scatter")
        routed.parent.mkdir(parents=True, exist_ok=True)
        _scatter_row(sub_df, routed, resolution=resolution, layout=layout, square_side=square_side,
                     point_size=point_size, legend_font=legend_font, axis_font=axis_font,
                     panel_title_pad=panel_title_pad, top_pad=top_pad, bottom_pad=bottom_pad, inter_pad=inter_pad,
                     suptitle_y=suptitle_y, dpi=dpi, plot_title=title)

        files.append(routed)

    _write(df, "scatter_whole.png", row_titles[0])
    _write(df[(df["Date"] >= wet_start) & (df["Date"] <= wet_end)], "scatter_wet.png", row_titles[1])
    _write(df[(df["Date"] >= dry_start) & (df["Date"] <= dry_end)], "scatter_dry.png", row_titles[2])

    return files


def _nse(o: np.ndarray, s: np.ndarray) -> float:
    keep = np.isfinite(o) & np.isfinite(s)
    if not keep.any():
        return np.nan
    o_, s_ = o[keep], s[keep]
    return 1.0 - np.sum((s_ - o_) ** 2) / np.sum((o_ - o_.mean()) ** 2)


def _pbias(o: np.ndarray, s: np.ndarray) -> float:
    keep = np.isfinite(o) & np.isfinite(s)
    if not keep.any():
        return np.nan
    o_, s_ = o[keep], s[keep]
    return 100.0 * np.sum(o_ - s_) / np.sum(o_)


def _load_four_basin_csvs(models_dir: Path, resolution: str, period: str, *, prefer_newest: bool = True,
                          stamp: str | None = None, run_label: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load four basin combined timeseries CSVs.

    - models_dir: root to search
    - resolution: 'daily' | 'hourly' | 'mts_daily' | 'mts_hourly'
    - period: 'val' | 'test'   (as used in filenames)
    - stamp/run_label: optional filter to pick a specific archived run folder '<RUN_LABEL>_<STAMP>'
    """
    dfs: dict[str, pd.DataFrame] = {}
    resolution = resolution.lower().strip()

    # Target filename suffix
    if resolution.startswith("mts_"):
        track = "1D" if resolution == "mts_daily" else "1H"
        suffix = f"mts*_{period.lower()}_{track}_combined_ts.csv"
    else:
        suffix = f"{resolution}_{period.lower()}_combined_ts.csv"

    for basin in _BASINS:
        hits: list[Path] = []

        legacy_root = models_dir / f"{basin}_all"
        if legacy_root.exists():
            hits.extend(legacy_root.glob(f"*{suffix}"))

        modern_root = models_dir / basin
        if modern_root.exists():
            hits.extend(modern_root.rglob(f"*{suffix}"))

        if not hits and models_dir.exists():
            hits.extend(models_dir.rglob(f"{basin}*{suffix}"))

        # Optional filter to specific archived run folder
        if stamp:
            token = f"{(run_label + '_') if run_label else ''}{stamp}"
            filt = [h for h in hits if token in str(h)]
            if filt:
                hits = filt

        if not hits:
            print(f"[WARN] {basin}: no file matching '*{suffix}' under {models_dir}")
            continue

        chosen = max(hits, key=lambda p: p.stat().st_mtime) if prefer_newest else sorted(hits)[0]
        df = pd.read_csv(chosen)

        # Parse dates
        if "Date" not in df and "date" in df:
            df = df.rename(columns={"date": "Date"})
        if df.get("Date") is not None and df["Date"].dtype != "datetime64[ns]":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # --- Normalize column names for MTS (_1D / _1H → plain names) ---
        if resolution.startswith("mts_"):
            trk = "1D" if resolution == "mts_daily" else "1H"
            # Accept a few common variants
            candidates = [
                (f"Observed_{trk}", "Observed"),
                (f"Observed_{trk.lower()}", "Observed"),
                (f"LSTM_Predicted_{trk}", "LSTM_Predicted"),
                (f"LSTM_Predicted_{trk.lower()}", "LSTM_Predicted"),
                (f"PLSTM_Predicted_{trk}", "PLSTM_Predicted"),
                (f"PLSTM_Predicted_{trk.lower()}", "PLSTM_Predicted"),
                (f"HMS_Predicted_{trk}", "HMS_Predicted"),
                (f"HMS_Predicted_{trk.lower()}", "HMS_Predicted"),
            ]
            rename_map = {old: new for (old, new) in candidates if old in df.columns}
            if rename_map:
                df = df.rename(columns=rename_map)

        # Ensure numeric and clip negatives for model outputs
        for col in ("Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if col.endswith("_Predicted"):
                    df.loc[df[col] < 0, col] = 0.0

        dfs[basin.replace("_", " ").title()] = df

    return dfs


def four_basin_period_strip(dfs: dict[str, pd.DataFrame], *, resolution: str, period_label: str, save_path: Path | str,
                            individual_xaxis: bool = False, figsize_per_basin: tuple[int, int] = (12, 5),
                            legend_font: int = 20, axis_font: int = 18, date_fmt: str = "%d‑%b‑%Y", dpi: int = 600,
                            point_alpha: float = .75, linewidth: float = 2.5, main_title: str | None = None,
                            main_title_font: int = 20, main_title_y: float = .975):
    resolution = resolution.lower().strip()
    is_mts = resolution.startswith("mts_")
    freq = resolution.split('_', 1)[1] if is_mts else resolution
    title_mode = "MTS‑LSTM" if is_mts else ""
    lstm_lbl = "MTS‑LSTM" if is_mts else "LSTM"
    pil_lbl = "MTS‑PILSTM" if is_mts else "PILSTM"

    n = len(_ORDER_FIXED)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_basin[0], figsize_per_basin[1] * n), dpi=dpi,
                             sharex=not individual_xaxis)

    if main_title:
        fig.suptitle(main_title, fontsize=main_title_font, y=main_title_y)

    for idx, (ax, basin) in enumerate(zip(axes, _ORDER_FIXED)):
        if basin not in dfs:
            ax.set_visible(False)
            continue

        df = dfs[basin]
        if "Date" not in df:
            ax.set_visible(False)
            continue

        # Parse/ensure numerics
        date = pd.to_datetime(df["Date"], errors="coerce")
        obs = pd.to_numeric(df.get("Observed"), errors="coerce") if "Observed" in df else None

        ax.plot(date, obs, lw=linewidth, label="Observed") if obs is not None else None

        # Which prediction columns are present?
        specs = []
        if "HMS_Predicted" in df:
            specs.append(("HMS_Predicted", "HMS"))
        if "LSTM_Predicted" in df:
            specs.append(("LSTM_Predicted", lstm_lbl))
        if "PLSTM_Predicted" in df:
            specs.append(("PLSTM_Predicted", pil_lbl))

        labels_with_metrics = []
        for col, lbl in specs:
            sim = pd.to_numeric(df[col], errors="coerce")
            if obs is None:
                nse_val, pb_val = np.nan, np.nan
            else:
                keep = np.isfinite(obs.to_numpy()) & np.isfinite(sim.to_numpy())
                if keep.any():
                    nse_val = _nse(obs.to_numpy()[keep], sim.to_numpy()[keep])
                    pb_val  = _pbias(obs.to_numpy()[keep], sim.to_numpy()[keep])
                else:
                    nse_val, pb_val = np.nan, np.nan

            labels_with_metrics.append((col, f"{lbl} (NSE={nse_val:.3f}, PBIAS={pb_val:.2f}%)"))

        # Plot models
        for col, full_lbl in labels_with_metrics:
            ax.plot(date, pd.to_numeric(df[col], errors="coerce"), lw=linewidth, alpha=point_alpha, label=full_lbl)

        title = f"{basin} {title_mode} {freq.capitalize()} – {period_label}".replace("  ", " ").strip()
        ax.set_title(title, fontsize=axis_font + 2, pad=8)
        ax.set_ylabel("Inflow (cfs)", fontsize=axis_font)
        ax.grid(alpha=.4)
        ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        ax.tick_params(axis="both", labelsize=axis_font - 2)
        ax.legend(fontsize=legend_font, fancybox=True, framealpha=.9, borderpad=1.2)

        if individual_xaxis or idx == n - 1:
            ax.set_xlabel("Date", fontsize=axis_font)

    if not individual_xaxis:
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = prepare_out_path(save_path, kind="plot_triptych")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def four_basin_period_triptych(dfs: dict[str, pd.DataFrame], *, resolution: str, out_dir: Path | str, wet_start: str,
                               wet_end: str, dry_start: str, dry_end: str, **strip_kwargs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    f = out_dir / "four_basin_whole.png"
    f = four_basin_period_strip(dfs, resolution=resolution, period_label="All Years", save_path=f, **strip_kwargs)
    files.append(f)

    dfs_wet = {k: v[(v["Date"] >= wet_start) & (v["Date"] <= wet_end)] for k, v in dfs.items()}
    f = out_dir / "four_basin_wet.png"
    f = four_basin_period_strip(dfs_wet, resolution=resolution, period_label="Wet Year", save_path=f, **strip_kwargs)
    files.append(f)

    dfs_dry = {k: v[(v["Date"] >= dry_start) & (v["Date"] <= dry_end)] for k, v in dfs.items()}
    f = out_dir / "four_basin_dry.png"
    f = four_basin_period_strip(dfs_dry, resolution=resolution, period_label="Dry Year", save_path=f, **strip_kwargs)
    files.append(f)

    return files


def _scatter_panel(ax, df: pd.DataFrame, *, model_col: str, model_label: str, legend_font: int, axis_font: int,
                   point_size: float = 28, show_xlabel: bool = True, show_ylabel: bool = True,
                   show_legend: bool = True):
    obs = df["Observed"].astype(float).to_numpy()
    pred = df[model_col].astype(float).to_numpy()
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]

    c_sim = "#e41a1c"
    ax.scatter(obs, pred, s=point_size, c=c_sim, alpha=.70, label=model_label if show_legend else None)

    lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
    if lo == hi:
        hi += 1

    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    nse = _nse(obs, pred)
    pb = _pbias(obs, pred)
    ax.text(0.98, 0.02, f"NSE = {nse:.3f}\nPBIAS = {pb:.2f}%", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=legend_font, bbox=dict(fc="white", ec="0.7", alpha=0.9, pad=0.35))

    if show_xlabel:
        ax.set_xlabel("Observed flow [cfs]", fontsize=axis_font)

    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("Predicted flow [cfs]", fontsize=axis_font)

    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.tick_params(labelsize=axis_font - 2)
    if show_legend:
        ax.legend(loc="upper left", fontsize=legend_font, framealpha=.9, borderpad=.4)


def four_basin_scatter_strip(dfs: dict[str, pd.DataFrame], *, save_path: str | Path, period_label: str,
                             model: str = "lstm", panel_height: float = 2.4, panel_width: float | None = None,
                             legend_font: int = 10, axis_font: int = 9, point_size: float = 22, dpi: int = 600,
                             hspace: float = 0.02, row_title_pad: float = 4, compact: bool = True,
                             main_title: str | None = None, main_title_font: int = 14, main_title_y: float = 0.995,
                             pad_top: float = 0.005, pad_bottom: float = 0.03, pad_left: float = 0.07,
                             pad_right: float = 0.995, basins_order: tuple[str, ...] =
                             ("Calpella", "Guerneville", "Hopland", "Warm Springs")) -> None:
    _MAP = {"hms": ("HMS_Predicted", "HMS Prediction"), "lstm": ("LSTM_Predicted", "LSTM Prediction"),
            "plstm": ("PLSTM_Predicted", "Physics Informed LSTM")}

    if model.lower() not in _MAP:
        raise ValueError(f"model must be one of {list(_MAP)}")
    model_col, model_lbl = _MAP[model.lower()]

    if panel_width is None:
        panel_width = panel_height

    figsize = (panel_width, panel_height * 4)
    fig, axes = plt.subplots(4, 1, figsize=figsize, dpi=dpi)

    if main_title:
        fig.suptitle(main_title, fontsize=main_title_font, y=main_title_y)

    for i, (ax, basin) in enumerate(zip(axes, basins_order)):
        if basin not in dfs or model_col not in dfs[basin]:
            ax.set_visible(False)
            continue

        sub = dfs[basin].copy()
        _scatter_panel(ax, sub, model_col=model_col, model_label=model_lbl, legend_font=legend_font,
                       axis_font=axis_font, point_size=point_size, show_xlabel=(i == len(axes) - 1) or not compact,
                       show_ylabel=True, show_legend=(i == 0) or not compact)

        ax.set_title(f"{basin} – {period_label}", fontsize=axis_font + 1, pad=row_title_pad)

    fig.subplots_adjust(hspace=hspace, top=1 - pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)

    save_path = prepare_out_path(save_path, kind="plot_scatter")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def four_basin_scatter_triptych(dfs: dict[str, pd.DataFrame], *, wet_start: str, wet_end: str, dry_start: str,
                                dry_end: str, out_dir: str | Path, **strip_kwargs) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    f = out_dir / "four_basin_scatter_whole.png"
    four_basin_scatter_strip(dfs, save_path=f, period_label="All Years", **strip_kwargs)
    files.append(prepare_out_path(f, kind="plot_scatter"))

    dfs_wet = {k: v[(v["Date"] >= wet_start) & (v["Date"] <= wet_end)] for k, v in dfs.items()}
    f = out_dir / "four_basin_scatter_wet.png"
    four_basin_scatter_strip(dfs_wet, save_path=f, period_label="Wet Year", **strip_kwargs)
    files.append(prepare_out_path(f, kind="plot_scatter"))

    dfs_dry = {k: v[(v["Date"] >= dry_start) & (v["Date"] <= dry_end)] for k, v in dfs.items()}
    f = out_dir / "four_basin_scatter_dry.png"
    four_basin_scatter_strip(dfs_dry, save_path=f, period_label="Dry Year", **strip_kwargs)
    files.append(prepare_out_path(f, kind="plot_scatter"))

    return files


def _load_trainval_predict_csvs(models_dir: Path, resolution: str, *, prefer_newest: bool = True) -> dict[str, pd.DataFrame]:
    print(f"\n[DIAG] models_dir = {models_dir.resolve()}")
    dfs: dict[str, pd.DataFrame] = {}

    resolution = resolution.lower().strip()
    is_mts = resolution.startswith("mts_")
    if resolution == "mts_daily":
        track = "1D"
    elif resolution == "mts_hourly":
        track = "1H"
    else:
        track = ""

    def _choose(cands: list[Path]) -> Path:
        if len(cands) == 1 or not prefer_newest:
            return cands[0]
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]

    for basin in _BASINS:
        if is_mts:
            if track:
                pats = [f"{basin}_mts*_{track}_trainval_ts_predict.csv",
                        f"{basin}_mts*_{track}_train_val_ts_predict.csv"]
            else:
                pats = [f"{basin}_mts*_*_trainval_ts_predict.csv",
                        f"{basin}_mts*_*_train_val_ts_predict.csv"]
        else:
            pats = [f"{basin}_{resolution}_trainval_ts_predict.csv",
                    f"{basin}_{resolution}_train_val_ts_predict.csv"]

        roots = [models_dir / f"{basin}_all", models_dir / basin]
        hits: list[Path] = []

        for pat in pats:
            for root in roots:
                if not root.exists():
                    print(f"[skip] {root.relative_to(models_dir)}  (folder absent)")
                    continue
                hits.extend(root.glob(pat))
            if not hits:
                hits.extend(models_dir.rglob(pat))
            if hits:  # stop at first pattern that finds anything
                break

        if not hits:
            print(f"[WARN] NO match for any of {pats}")
            continue

        chosen = _choose(hits)
        print(f"[INFO] chosen → {chosen.relative_to(models_dir)}")

        df = pd.read_csv(chosen)
        if df["Date"].dtype != "datetime64[ns]":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ("LSTM_Predicted", "PLSTM_Predicted"):
            if col in df:
                df.loc[df[col] < 0, col] = 0.0
        dfs[basin.replace('_', ' ').title()] = df

    return dfs


def four_basin_quadrant_grid(models_dir: Path, *, resolution: str = "hourly", period_label: str = "Train+Val period",
                             out_png: str | Path = "four_basin_ts_strip.png", layout: str = "grid",
                             panel_height: float = 2.6, panel_width: float | None = None,
                             figsize: tuple[int, int] = (10, 6), legend_font: int = 9, axis_font: int = 9,
                             date_fmt: str = "%d‑%b‑%Y", hspace: float = 0.06, wspace: float = 0.04, dpi: int = 600,
                             main_title: str | None = None, main_title_font: int = 14, main_title_y: float = .99):
    layout = layout.lower()
    if layout not in ("grid", "vertical"):
        raise ValueError("layout must be 'grid' or 'vertical'")

    dfs = _load_trainval_predict_csvs(models_dir, resolution)
    if len(dfs) < 4:
        print("[WARN] fewer than 4 basins loaded – plot will have blanks.")

    order = ["Calpella", "Hopland", "Warm Springs", "Guerneville"]

    _bak = plt.rcParams.copy()
    plt.rcParams.update({"axes.labelsize": axis_font, "xtick.labelsize": axis_font - 2,
                         "ytick.labelsize": axis_font - 2})

    if layout == "grid":
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        axes_map = dict(zip(order, [axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]]))

    else:
        if panel_width is None:
            panel_width = panel_height * 1.6
        fig, axes = plt.subplots(4, 1, figsize=(panel_width, panel_height * 4), dpi=dpi, sharex=True)
        axes_map = dict(zip(order, axes))

    if main_title:
        fig.suptitle(main_title, fontsize=main_title_font, y=main_title_y)

    def _ts_panel(ax, df: pd.DataFrame, title: str):
        ax.plot(df["Date"], df["Observed"], lw=1.8, label="Observed")
        if "HMS_Predicted" in df:
            ax.plot(df["Date"], df["HMS_Predicted"], lw=1.4, label="HMS")
        if "LSTM_Predicted" in df:
            ax.plot(df["Date"], df["LSTM_Predicted"], lw=1.4, label="LSTM")
        if "PLSTM_Predicted" in df:
            ax.plot(df["Date"], df["PLSTM_Predicted"], lw=1.4, label="PI‑LSTM")

        ax.set_title(title, fontsize=axis_font + 1, pad=6)
        ax.grid(alpha=.3)
        ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        ax.tick_params(labelsize=axis_font - 2)

    for basin in order:
        ax = axes_map[basin]
        if basin in dfs:
            _ts_panel(ax, dfs[basin], f"{basin} – {period_label}")
            if basin == "Calpella":
                ax.legend(fontsize=legend_font, fancybox=True, framealpha=.9, borderpad=.4)
        else:
            ax.set_visible(False)

    if layout == "grid":
        axes_map["Guerneville"].set_xlabel("Date", fontsize=axis_font)
        axes_map["Warm Springs"].set_xlabel("Date", fontsize=axis_font)
        fig.subplots_adjust(hspace=hspace, wspace=wspace, left=.06, right=.995, top=.92, bottom=.06)
    else:
        axes_map["Warm Springs"].set_xlabel("Date", fontsize=axis_font)
        fig.subplots_adjust(hspace=hspace, left=.08, right=.995, top=.92, bottom=.06)

    out_png = prepare_out_path(out_png, kind="plot_triptych")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(_bak)

    return out_png


def four_basin_trainval_strip(models_dir: Path | str, *, resolution: str = "hourly",
                              out_png: Path | str = "four_basin_trainval_ts_strip.png",
                              individual_xaxis: bool = False, figsize_per_basin: tuple[int, int] = (12, 5),
                              legend_font: int = 20, axis_font: int = 18, date_fmt: str = "%d‑%b‑%Y", dpi: int = 600,
                              point_alpha: float = .75, linewidth: float = 2.5, main_title: str | None = None,
                              main_title_font: int = 20, main_title_y: float = .975):

    from neuralhydrology.evaluation.metrics import calculate_all_metrics

    models_dir = Path(models_dir).resolve()
    resolution = resolution.lower().strip()
    is_mts = resolution.startswith("mts_")
    freq = resolution.split('_', 1)[1] if is_mts else resolution
    track = {"mts_daily": "1D", "mts_hourly": "1H"}.get(resolution, "")

    title_mode = "MTS‑LSTM" if is_mts else ""
    lstm_lbl = "MTS‑LSTM" if is_mts else "LSTM"
    pil_lbl = "MTS‑PILSTM" if is_mts else "PILSTM"

    dfs: dict[str, pd.DataFrame] = {}

    for basin in _BASINS:
        pats: list[str] = []
        if is_mts:
            pats += [f"{basin}_mts*_{freq}_trainval_ts_predict.csv",
                     f"{basin}_mts*_{freq}_train_val_ts_predict.csv"]
            if track:
                pats += [f"{basin}_mts*_{track}_trainval_ts_predict.csv",
                         f"{basin}_mts*_{track}_train_val_ts_predict.csv"]
        else:
            pats += [f"{basin}_{resolution}_trainval_ts_predict.csv",
                     f"{basin}_{resolution}_train_val_ts_predict.csv"]

        hits: list[Path] = []
        for pat in pats:
            hits = list(models_dir.glob(pat)) or list(models_dir.rglob(pat))
            if hits:
                break

        if not hits:
            print(f"[WARN]  missing CSV for {basin} (looked for patterns: {', '.join(pats)})")
            continue

        df = pd.read_csv(hits[0])
        if df["Date"].dtype != "datetime64[ns]":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        dfs[basin.replace('_', ' ').title()] = df

    if not dfs:
        raise FileNotFoundError(f"No *trainval* or *train_val* CSV files found inside {models_dir}")

    n = len(_ORDER_FIXED)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_basin[0], figsize_per_basin[1] * n), dpi=dpi,
                             sharex=not individual_xaxis)

    if main_title:
        fig.suptitle(main_title, fontsize=main_title_font, y=main_title_y)

    for idx, (ax, basin) in enumerate(zip(axes, _ORDER_FIXED)):
        if basin not in dfs:
            ax.set_visible(False)
            continue
        df = dfs[basin]

        to_da = lambda s: xr.DataArray(s.values, dims=["date"], coords={"date": df["Date"]})
        obs_da, hms_da, lst_da, pil_da = map(to_da, (
            df["Observed"], df.get("HMS_Predicted", pd.Series(index=df.index, dtype=float)),
            df["LSTM_Predicted"], df["PLSTM_Predicted"]))

        # Safer NSE: fall back to NaN if no valid overlap
        def _safe_nse(o: xr.DataArray, s: xr.DataArray) -> float:
            o_np = pd.to_numeric(pd.Series(o.values), errors="coerce").to_numpy()
            s_np = pd.to_numeric(pd.Series(s.values), errors="coerce").to_numpy()
            keep = np.isfinite(o_np) & np.isfinite(s_np)
            return _nse(o_np[keep], s_np[keep]) if keep.any() else np.nan

        metrics = {
            "HMS": dict(NSE=_safe_nse(obs_da, hms_da), PBIAS=_pbias(obs_da.values, hms_da.values)),
            lstm_lbl: dict(NSE=_safe_nse(obs_da, lst_da), PBIAS=_pbias(obs_da.values, lst_da.values)),
            pil_lbl: dict(NSE=_safe_nse(obs_da, pil_da), PBIAS=_pbias(obs_da.values, pil_da.values)),
        }

        _fmt = lambda d: ", ".join(f"{k}={d[k]:.3f}" for k in d if np.isfinite(d[k]))

        ax.plot(df["Date"], df["Observed"], lw=linewidth, label="Observed")
        if "HMS_Predicted" in df:
            ax.plot(df["Date"], df["HMS_Predicted"], lw=linewidth, alpha=point_alpha,
                    label=f"HMS {_fmt(metrics['HMS'])}")
        ax.plot(df["Date"], df["LSTM_Predicted"], lw=linewidth, alpha=point_alpha,
                label=f"{lstm_lbl} {_fmt(metrics[lstm_lbl])}")
        ax.plot(df["Date"], df["PLSTM_Predicted"], lw=linewidth, alpha=point_alpha,
                label=f"{pil_lbl} {_fmt(metrics[pil_lbl])}")

        nice_freq = freq.capitalize()
        row_title = f"{basin} {title_mode} {nice_freq}".replace("  ", " ").strip()
        ax.set_title(row_title, fontsize=axis_font + 2, pad=8)

        ax.set_ylabel("Inflow (cfs)", fontsize=axis_font)
        ax.grid(alpha=.4)
        ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        ax.tick_params(axis="both", labelsize=axis_font - 2)
        ax.legend(fontsize=legend_font, fancybox=True, framealpha=.9, borderpad=1.2)

        if individual_xaxis or idx == n - 1:
            ax.set_xlabel("Date", fontsize=axis_font)

    if not individual_xaxis:
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = prepare_out_path(out_png, kind="plot_triptych")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png
