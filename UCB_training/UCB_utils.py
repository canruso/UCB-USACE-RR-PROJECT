from __future__ import annotations
from typing import Union, List
import json
import math
import os
import re
import shutil
import subprocess
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

_BASINS = ("calpella", "guerneville", "hopland", "warm_springs")
_ORDER_FIXED = ("Calpella", "Hopland", "Warm Springs", "Guerneville")
_TS_DIR = "timeseries"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_A = _REPO_ROOT / "data" / "russian_river_data"
_DATA_B = _REPO_ROOT / "russian_river_data"
_PERIOD_RE = re.compile(r"(validation|valid|val|test|train(_?val)?)", re.I)
_CTX: dict[str, str | None] = {"basin": None, "res": None}
_RUN_STAMP: str | None = None
_RUN_TAG: str | None = None
_APPEND_STAMP_TO_FILENAMES: bool = False


def _resolve_outputs_root(repo_root: Path) -> Path:
    cand_top = repo_root / "outputs"
    cand_pkg = repo_root / "UCB_training" / "outputs"
    if cand_top.exists():
        return cand_top
    if cand_pkg.exists():
        return cand_pkg
    return cand_top


_OUTPUTS = _resolve_outputs_root(_REPO_ROOT)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw HMS/HEC-style table into a tidy DataFrame indexed by datetime (index name 'date').
    The main objective is to replace rows with 24:00:00 to 00:00:00.

    Expected columns (case-sensitive as exported by HEC):
        - 'Date' (or 'Day' after renaming)
        - 'Time' (or 'time' before renaming)
        - other value columns (drop 'Ordinate' if present)
    """
    df = df.copy()
    df.columns = df.iloc[0]
    df = df[3:]

    df.columns = df.columns.astype(str).str.strip()
    if "Ordinate" in df.columns:
        df = df.drop(columns=["Ordinate"])

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "Day"})
    if "Time" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "Time"})

    # Handle 24:00:00 by rolling to next day at 00:00:00
    mask = df["Time"] == "24:00:00"
    if mask.any():
        df.loc[mask, "Day"] = ((pd.to_datetime(df.loc[mask, "Day"], format="%d-%b-%y") + pd.Timedelta(days=1))
                               .dt.strftime("%d-%b-%y"))
        df["Time"] = df["Time"].replace("24:00:00", "00:00:00")

    df["date"] = pd.to_datetime(df["Day"], format="%d-%b-%y") + pd.to_timedelta(df["Time"])
    df.dropna(subset=["date"], inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep=False)]

    df.set_index("date", inplace=True)
    for col in ("Day", "Time"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def open_tensorboard(logdir: str, port: int = 6006):
    """
    Launch TensorBoard for the given logdir on the specified port and open the browser.
    Returns the Popen process handle.
    """
    logdir_path = Path(logdir)
    if not logdir_path.exists():
        raise FileNotFoundError(f"Log directory {logdir} does not exist.")

    event_files = list(logdir_path.rglob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in log directory {logdir}.")

    cmd = f"tensorboard --logdir={logdir} --port={port} --host=0.0.0.0"
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)  # give it a moment to start
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        print(f"TensorBoard started at {url} with logs from {logdir}")
    except Exception as e:
        raise RuntimeError(f"Failed to start TensorBoard: {e}")
    return process


def fractional_multi_lr(epochs: int, fractions: list[float], lrs: list[float], round_up: bool = True) -> dict[
    int, float]:
    """
    Build a dictionary for a flexible piecewise learning-rate schedule.

      - 'fractions': N fractional breakpoints summing to <= 1.0
      - 'lrs': N+1 learning rates
      - Generates N+1 segments in total (including the initial segment)

    Example:
      epochs=16, fractions=[0.4, 0.3], lrs=[0.01, 0.005, 0.001]
      first 40% of epochs: 0.01, next 30%: 0.005, last 30%: 0.001
      schedule like {0: 0.01, 7: 0.005, 11: 0.001} if round_up=True
    """
    if len(lrs) != len(fractions) + 1:
        raise ValueError("Number of learning rates must be len(fractions) + 1. "
                         f"Got {len(lrs)} LRs and {len(fractions)} fractions.")
    if any(f < 0 for f in fractions):
        raise ValueError("Fractions must be non-negative.")
    if sum(fractions) > 1.0 + 1e-9:
        raise ValueError(f"The sum of fractions exceeds 1.0 => {sum(fractions)}")

    schedule: dict[int, float] = {0: float(lrs[0])}
    cumulative = 0.0
    for i, frac in enumerate(fractions, start=1):
        cumulative += frac
        boundary_float = cumulative * epochs
        boundary_index = math.ceil(boundary_float) if round_up else int(boundary_float)
        boundary_index = min(max(boundary_index, 0), max(epochs - 1, 0))
        schedule[boundary_index] = float(lrs[i])

    return schedule


def write_paths(tag: str, trainer, *, filename: str = "stored_runs.json") -> None:
    if trainer is None:
        print(f"trainer for tag '{tag}' is None – nothing written.")
        return

    run_dirs = trainer._model if isinstance(trainer._model, list) else [trainer._model]
    run_dirs = [Path(p).resolve() for p in run_dirs]

    def _infer_runs_parent(p: Path) -> Path:
        parts = p.parts
        try:
            i = parts.index("runs")
        except ValueError:
            return p.parent
        if i + 1 < len(parts):
            return Path(*parts[: i + 2])
        return Path(*parts[: i + 1])

    runs_parent = _infer_runs_parent(run_dirs[0])
    names = [p.relative_to(runs_parent).as_posix() for p in run_dirs]

    reg_file = Path(filename)
    data = json.loads(reg_file.read_text()) if reg_file.exists() else {}
    data[tag] = names
    reg_file.parent.mkdir(parents=True, exist_ok=True)
    reg_file.write_text(json.dumps(data, indent=2))

    print(f"stored experiments for '{tag}' in {reg_file.name}: {names}")


def to_path_or_list(seq) -> Union[Path, List[Path]]:
    """Helper to convert an iterable of path-like strings to a Path object if length==1, else a list of Path's."""
    seq = [Path(p) for p in seq]
    return seq[0] if len(seq) == 1 else seq


def make_run_stamp() -> str:
    """Helper to create a run stamp to be used in the naming of files/folders."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_root() -> Path:
    """Return the root folder of the repository."""
    return _REPO_ROOT


def data_dir() -> Path:
    """Return the data folder, preferring 'data/russian_river_data' but falling back if needed."""
    return _DATA_A if _DATA_A.exists() else _DATA_B


def set_active_context(basin: str, resolution: str, *, run_stamp: str | None = None,
                       append_stamp_to_filenames: bool = False, run_tag: str | None = None) -> None:
    """
    Register the active basin + resolution so prepare_out_path() can route relative outputs.
    """
    _CTX["basin"] = basin
    _CTX["res"] = resolution
    global _RUN_STAMP, _APPEND_STAMP_TO_FILENAMES, _RUN_TAG
    _RUN_STAMP = run_stamp
    _APPEND_STAMP_TO_FILENAMES = append_stamp_to_filenames
    _RUN_TAG = run_tag.strip() if run_tag else None


def _compose_run_folder_name() -> str | None:
    """Compose the run folder name <RUN_TAG>_<RUN_STAMP> or just <RUN_STAMP>, or None if no stamp."""
    if not _RUN_STAMP:
        return None
    prefix = f"{_RUN_TAG}_" if _RUN_TAG else ""
    return f"{prefix}{_RUN_STAMP}"


def get_basin_root(basin: str) -> Path:
    """Return outputs/<basin>"""
    return _OUTPUTS / basin


def get_output_dir(basin: str, resolution: str) -> Path:
    """
    Resolution in {'mts_daily','mts_hourly','daily','hourly'}
    Returns: outputs/<basin>/<resolution>[/<RUN_TAG_>STAMP]
    """
    base = get_basin_root(basin) / resolution
    run_folder = _compose_run_folder_name()
    return base / run_folder if run_folder else base


def ensure_output_tree(basin: str, resolution: str) -> Path:
    """
    Create the expected folder layout under outputs/<basin>/<resolution>/...

    Rules:
    - 'mts'            → NO-OP (logical alias only; nothing on disk)
    - 'mts_daily/_hourly' → create ONLY the resolution root; do NOT pre-create
                            timeseries/metrics/plots subtrees
    - 'daily'/'hourly' → pre-create the full tree (legacy behavior)
    """
    res = resolution.strip().lower()
    root = get_output_dir(basin, resolution)

    # Synthetic alias: never create a tree for 'mts'
    if res == "mts":
        return root

    # For MTS resolutions, only ensure the root exists (no subtrees).
    if res in {"mts_daily", "mts_hourly"}:
        root.mkdir(parents=True, exist_ok=True)
        return root

    # Legacy single-resolution modes: keep the full layout.
    subdirs = [
        f"{_TS_DIR}/train_val",
        f"{_TS_DIR}/validation",
        f"{_TS_DIR}/test",
        "metrics/validation",
        "metrics/train_val",
        "metrics/test",
        "plots/timeseries/validation",
        "plots/timeseries/test",
        "plots/timeseries/train_val",
        "plots/gridded_scatterplots",
        "plots/gridded_timeseries",
        "hyperparams",
        "logs",
    ]
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def get_yaml_path(name_or_basin: str) -> Path:
    """
    Return a YAML path inside UCB_training/configs. If you pass 'warm_springs_mtslstm2',
    it uses that exact filename; if you pass 'warm_springs', it looks for 'warm_springs*.yaml'
    and picks the first match.

    Note: Feel free to make this function more restrictive if you want!
    """

    cfg_dir = _REPO_ROOT / "UCB_training" / "configs"
    p = cfg_dir / (name_or_basin + ("" if name_or_basin.endswith(".yaml") else ".yaml"))
    if p.exists():
        return p
    hits = sorted(cfg_dir.glob(f"{name_or_basin}*.yaml"))
    if not hits:
        raise FileNotFoundError(f"No YAML file for '{name_or_basin}' in {cfg_dir}")
    return hits[0]


def _infer_period_from_text(txt: str | None) -> str | None:
    """Try to infer a period label ("validation" / "test" / "train_val") from a filename stem.
    Why we need this: if the user provides a relative path like "my_results.csv", we want to
    route it to the appropriate subfolder under timeseries/ or metrics/ based on whether
    it contains "val", "test", or "train" in the name.
    """
    if not txt:
        return None
    m = _PERIOD_RE.search(txt)
    if not m:
        return None
    w = m.group(1).lower()
    if w.startswith(("val", "valid")):
        return "validation"
    if w.startswith("test"):
        return "test"
    if w.startswith("train"):
        return "train_val"

    return None


def _maybe_append_stamp(p: Path) -> Path:
    """Optionally append the active run-stamp to the filename."""
    if not _APPEND_STAMP_TO_FILENAMES or not _RUN_STAMP:
        return p
    return p.with_name(f"{p.stem}_{_RUN_STAMP}{p.suffix}")


def _need_routing(p: Path) -> bool:
    """
    Route any *relative* path (bare name OR subpath). Absolute paths are left untouched.
    This ensures that user-provided subfolders like "warm_springs_mts_hourly_scatter/..."
    are placed under the correct plots/ subtree instead of next to it.
    """
    return not p.is_absolute()


def prepare_out_path(name_or_path, *, kind: str, period: str | None = None) -> Path:
    """
    Given a relative path or bare name, return a routed output path under the active context.
    'kind' in {'timeseries','metrics','plot_timeseries','plot_scatter','plot_triptych','csv','ts'}
    'period' in {'validation','test','train_val'} or None to infer from filename
    If no active context is set, returns the path as-is (relative to cwd).
    If active context has a run stamp and append_stamp_to_filenames=True, appends stamp to the filename before suffix

    """
    p = Path(name_or_path)
    if not _need_routing(p):
        return p

    basin, res = _CTX["basin"], _CTX["res"]
    if not basin or not res:
        return p

    base = get_output_dir(basin, res)
    per = period or _infer_period_from_text(p.stem)

    if kind in {"timeseries", "csv", "ts"}:
        sub = f"{_TS_DIR}/{per or 'validation'}"
    elif kind == "metrics":
        sub = f"metrics/{per or 'validation'}"
    elif kind == "plot_timeseries":
        sub = f"plots/timeseries/{per or 'validation'}"
    elif kind == "plot_scatter":
        sub = "plots/gridded_scatterplots"
    elif kind == "plot_triptych":
        sub = "plots/gridded_timeseries"
    else:
        sub = ""

    rel = p if p.parent != Path(".") else Path(p.name)
    out = base / sub / rel
    return _maybe_append_stamp(out)


def switch_ctx_explicit(basin: str, resolution: str, *, run_stamp: str | None = None, run_label: str | None = None,
                        append_stamp_to_filenames: bool = False) -> Path:
    """
    Set active basin+resolution context, ensure output tree, chdir into it, and return the output dir.
    """
    set_active_context(basin, resolution, run_stamp=run_stamp, run_tag=run_label,
                       append_stamp_to_filenames=append_stamp_to_filenames)

    ensure_output_tree(basin, resolution)
    out_dir = get_output_dir(basin, resolution)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    return out_dir


def ctx_for(basin: str, *, run_stamp: str | None = None, run_label: str | None = None,
            append_stamp_to_filenames: bool = False):
    """
    Needed for the MTS notebook; Routes between 1H and 1D resolutions
    """

    def _switch(resolution: str) -> Path:
        return switch_ctx_explicit(basin, resolution, run_stamp=run_stamp, run_label=run_label,
                                   append_stamp_to_filenames=append_stamp_to_filenames)

    return _switch


def get_shared_dir(basin: str, mode: str) -> Path:
    """
    mode in {'mts','daily','hourly'} (not 'mts_daily'); returns e.g. outputs/<basin>/mts_shared
    """
    mode = mode.strip().lower()
    if mode not in {"mts", "daily", "hourly"}:
        raise ValueError(f"mode must be 'mts', 'daily', or 'hourly', got {mode}")
    return get_basin_root(basin) / f"{mode}_shared"


def ensure_shared_tree(basin: str, mode: str) -> Path:
    root = get_shared_dir(basin, mode)
    for sub in ["hyperparams", "hyperparams/archive", "runs", "runs/archive", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _artifact_root(basin: str, mode: str) -> Path:
    root = get_shared_dir(basin, mode)
    (root / "hyperparams").mkdir(parents=True, exist_ok=True)
    (root / "hyperparams" / "archive").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "archive").mkdir(parents=True, exist_ok=True)
    return root


def _midnight_stamp_utc() -> str:
    """Return a simple, stable stamp 'YYYYMMDDT000000Z' (midnight UTC today)."""
    today = datetime.now(timezone.utc).date()
    return f"{today.strftime('%Y%m%d')}T000000Z"


def hparams_exists(basin: str, mode: str, label: str = "BASELINE", stamp: str | None = None) -> bool:
    """ Check if the best-params CSV exists in LATEST for the given basin+mode+label.
    If 'stamp' is provided, check the archive file with that stamp. Otherwise, check the LATEST file.
    """
    root = _artifact_root(basin, mode)
    hp_dir = root / "hyperparams"
    arch = hp_dir / "archive"
    prefix = f"{basin}_{mode}_{label}"

    if stamp:
        return (arch / f"{prefix}_hyperparams_{stamp}.csv").exists()
    return (hp_dir / f"{prefix}_hyperparams.csv").exists()


def load_hparams(basin: str, mode: str, label: str = "BASELINE", stamp: str | None = None,
                 strict_archive: bool = True) -> pd.DataFrame:
    """
    Load best-params CSV.

    Behavior:
    - If 'stamp' is provided: load from archive/<prefix>_hyperparams_{stamp}.csv.
      If missing:
        - strict_archive=True  -> raise FileNotFoundError
        - strict_archive=False -> fall back to LATEST, then to newest archive.
    - If 'stamp' is None: load LATEST. If missing, fall back to newest archive.
    """
    root = _artifact_root(basin, mode)
    hp_dir = root / "hyperparams"
    arch = hp_dir / "archive"
    prefix = f"{basin}_{mode}_{label}"

    if stamp:
        stamped = arch / f"{prefix}_hyperparams_{stamp}.csv"
        if stamped.exists():
            print(stamped)
            return pd.read_csv(stamped)
        if strict_archive:
            raise FileNotFoundError(
                f"Expected stamped hyperparams not found: {stamped}\n"
                f"(You can set strict_archive=False to fall back to latest.)")

    latest = hp_dir / f"{prefix}_hyperparams.csv"
    if latest.exists():
        print(latest)
        return pd.read_csv(latest)

    matches = sorted(arch.glob(f"{prefix}_hyperparams_*.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        print(matches[0])
        return pd.read_csv(matches[0])

    raise FileNotFoundError(f"No best‑params CSV for {prefix} in {hp_dir} or {arch}")


def save_hparams(*, best_df: pd.DataFrame, basin: str, mode: str, label: str = "BASELINE", run_stamp: str | None = None,
                 df_no: pd.DataFrame | None = None, df_phys: pd.DataFrame | None = None) -> None:
    """
    Save best‑params + (optionally) full grid results to:
      LATEST:  outputs/<basin>/<mode>_shared/hyperparams/...
      ARCHIVE: outputs/<basin>/<mode>_shared/hyperparams/archive/..._{STAMP}.csv
    """
    if run_stamp is None:
        run_stamp = _midnight_stamp_utc()

    root = _artifact_root(basin, mode)
    hp_dir = root / "hyperparams"
    arch = hp_dir / "archive"
    prefix = f"{basin}_{mode}_{label}"

    best_latest = hp_dir / f"{prefix}_hyperparams.csv"
    best_archive = arch / f"{prefix}_hyperparams_{run_stamp}.csv"
    best_df.to_csv(best_latest, index=False)
    best_df.to_csv(best_archive, index=False)

    if df_no is not None:
        df_no.to_csv(hp_dir / f"{prefix}_no_physics_gridsearch.csv", index=False)
        df_no.to_csv(arch / f"{prefix}_no_physics_gridsearch_{run_stamp}.csv", index=False)
    if df_phys is not None:
        df_phys.to_csv(hp_dir / f"{prefix}_physics_gridsearch.csv", index=False)
        df_phys.to_csv(arch / f"{prefix}_physics_gridsearch_{run_stamp}.csv", index=False)


def runs_latest_path(basin: str, mode: str, label: str = "BASELINE") -> Path:
    """Location of the LATEST runs registry JSON for a basin+mode+label."""
    root = _artifact_root(basin, mode)
    return root / "runs" / f"{basin}_{mode}_{label}_stored_runs.json"


def archive_runs_json(
        latest_path: Path, basin: str, mode: str, label: str = "BASELINE", run_stamp: str | None = None) -> Path:
    """Copy the given runs JSON to the per-stamp archive folder, returning the archive path."""
    if run_stamp is None:
        run_stamp = _midnight_stamp_utc()
    arch = _artifact_root(basin, mode) / "runs" / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    out = arch / f"{basin}_{mode}_{label}_stored_runs_{run_stamp}.json"
    shutil.copy2(latest_path, out)
    return out


def read_csv_artifact(name: str, *, kind: str, period: str | None = None, index_col=None, stamp: str | None = None,
                      run_label: str | None = None) -> pd.DataFrame:
    """Read CSV's according to the new routing logic."""
    p = prepare_out_path(name, kind=kind, period=period)
    if p.exists():
        return pd.read_csv(p, index_col=index_col)

    if stamp and run_label:
        try:
            base = get_output_dir(_CTX["basin"], _CTX["res"])
            sub = p.parent.relative_to(base)
            q = base / f"{run_label}_{stamp}" / sub / p.name
            if q.exists():
                return pd.read_csv(q, index_col=index_col)
        except Exception:
            pass

    cands = sorted(p.parent.glob(p.stem + "*.csv"), key=lambda q: q.stat().st_mtime, reverse=True)

    if not cands:
        raise FileNotFoundError(f"Couldn't find {name} under {p.parent}")
    return pd.read_csv(cands[0], index_col=index_col)


def runs_parent_dir(basin: str, mode: str, label: str, stamp: str | None = None) -> Path:
    """
    Return the parent folder that holds experiment run folders for a (basin, mode, label[, stamp]).
    Layout: outputs/<basin>/<mode>_shared/runs/<label>[_<stamp>]
    """
    # get_shared_dir(basin, mode) - outputs/<basin>/<mode>_shared
    base = get_shared_dir(basin, mode) / "runs"
    parent = base / (f"{label}_{stamp}" if stamp else label)
    return parent


def resolve_run_dirs(filename: str | Path, *, basin: str, mode: str, label: str, stamp: str | None = None,
                     tags: tuple[str, ...] = ("no_physics", "physics")) -> dict[str, list[Path]]:
    p = Path(filename)
    data = json.loads(p.read_text()) if p.exists() else {}

    parent = runs_parent_dir(basin, mode, label, stamp)
    out: dict[str, list[Path]] = {}
    for tag in tags:
        names = data.get(tag, [])
        out[tag] = [parent / Path(name) for name in names]
    return out


def resolve_basin_file(basin: str, *, must_exist: bool = True) -> Path:
    """
    Return the absolute path to the basin list file for NeuralHydrology.
      resolve_basin_file('calpella') -> .../notebooks/basins/calpella/calpella
    """
    base = (repo_root() / "notebooks" / "basins" / basin.lower()).resolve()
    name = "warm springs" if basin.lower() == "warm_springs" else basin.lower()
    candidates = [base / name, base / f"{name}.txt"]

    for p in candidates:
        if p.exists():
            return p.resolve()

    if must_exist:
        existing = ", ".join(sorted(p.name for p in base.iterdir())) if base.exists() else "<<folder missing>>"
        tried = ", ".join(c.name for c in candidates)
        raise FileNotFoundError(
            f"Could not find basin list for '{basin}'. Looked for [{tried}] in {base}. Existing: {existing}")

    return (candidates[0]).resolve()


def ensure_absolute_basin_files(trainer_or_cfg, basin: str,
                                keys: tuple[str, ...] = (
                                        "train_basin_file", "validation_basin_file", "test_basin_file")) -> Path:
    """
    Ensure the given NH config (or UCB_trainer instance) has ABSOLUTE paths for the basin list entries.
    If any is missing or relative, set all to the resolved absolute basin list path for the given basin.
    """

    cfg = getattr(trainer_or_cfg, "_config", trainer_or_cfg)
    needs_patch = False
    for k in keys:
        val = getattr(cfg, k, None)
        if val is None:
            needs_patch = True
            break
        try:
            if not Path(val).is_absolute():
                needs_patch = True
                break
        except TypeError:
            needs_patch = True
            break

    if not needs_patch:
        for k in keys:
            v = getattr(cfg, k, None)
            if v is not None:
                try:
                    return Path(v).resolve()
                except Exception:
                    break
    p = resolve_basin_file(basin)
    cfg.update_config({k: p for k in keys}, dev_mode=True)
    return p

def basin_prefix(basin: str) -> str:
    if basin.lower() == "warm_springs":
        return "WarmSprings_Inflow"
    return basin.replace("_", " ").title().replace(" ", "")


def read_hec_table(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    has_date = raw.apply(lambda r: r.astype(str).str.contains(r"\bdate\b", case=False, na=False).any(), axis=1)
    if not bool(has_date.any()):
        raise RuntimeError(f"'Date' header row not found in {path}")
    hdr = int(has_date.idxmax())
    header = raw.iloc[hdr].tolist()
    units  = raw.iloc[hdr + 1].tolist()
    types  = raw.iloc[hdr + 2].tolist()
    colnum = raw.iloc[0].tolist() if raw.iloc[0].astype(str).str.contains("col num", case=False, na=False).any() else None
    data_start = hdr + 3
    df = raw.iloc[data_start:].copy()
    df.columns = header
    meta = {"header": header, "units": units, "types": types, "colnum": colnum, "data_start": data_start}
    return df, meta

def write_hec_table(df: pd.DataFrame, meta: dict, out_path: Path) -> None:
    cols = list(df.columns)
    def _row(vals: list[str]) -> pd.DataFrame:
        v = list(vals)[:len(cols)]
        if len(v) < len(cols):
            v += [""] * (len(cols) - len(v))
        return pd.DataFrame([v], columns=cols)
    parts = []
    if meta.get("colnum") is not None:
        parts.append(_row(meta["colnum"]))
    parts += [_row(meta["header"]), _row(meta["units"]), _row(meta["types"]), df[cols].astype(str)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(out_path, index=False, header=False)

def find_label(df: pd.DataFrame, name: str) -> str:
    want = name.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == want:
            return c
    raise KeyError(f"Column not found: {name}")
