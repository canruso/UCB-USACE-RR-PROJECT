"""
Utility to generate loss curve plots from NeuralHydrology run directories.
This module reads TensorBoard event files and validation metrics to create
comprehensive training/validation loss visualizations.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_metrics_from_csvs(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract training and validation metrics from CSV files in the run directory.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing validation/model_epoch* folders

    Returns
    -------
    train_df : pd.DataFrame
        DataFrame with columns [epoch, metric_name, value] for training
    valid_df : pd.DataFrame
        DataFrame with columns [epoch, metric_name, value] for validation
    """
    validation_dir = run_dir / "validation"
    train_dir = run_dir / "train"

    valid_data = []
    train_data = []

    # Extract validation metrics
    if validation_dir.exists():
        for epoch_dir in sorted(validation_dir.glob("model_epoch*")):
            epoch_num = int(epoch_dir.name.replace("model_epoch", ""))
            metrics_file = epoch_dir / "validation_metrics.csv"

            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                # Skip basin column, get all metric columns
                for col in df.columns:
                    if col.lower() != 'basin':
                        valid_data.append({
                            'epoch': epoch_num,
                            'metric': col,
                            'value': df[col].values[0]
                        })

    # Extract training metrics (if available)
    if train_dir.exists():
        for epoch_dir in sorted(train_dir.glob("model_epoch*")):
            epoch_num = int(epoch_dir.name.replace("model_epoch", ""))
            metrics_file = epoch_dir / "train_metrics.csv"

            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                for col in df.columns:
                    if col.lower() != 'basin':
                        train_data.append({
                            'epoch': epoch_num,
                            'metric': col,
                            'value': df[col].values[0]
                        })

    train_df = pd.DataFrame(train_data) if train_data else pd.DataFrame(columns=['epoch', 'metric', 'value'])
    valid_df = pd.DataFrame(valid_data) if valid_data else pd.DataFrame(columns=['epoch', 'metric', 'value'])

    return train_df, valid_df


def extract_losses_from_tensorboard(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract training and validation losses from TensorBoard event files.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing TensorBoard event files

    Returns
    -------
    losses : Dict[str, List[Tuple[int, float]]]
        Dictionary with keys 'train_loss' and 'valid_loss', values are lists of (epoch, loss) tuples
    """
    event_files = list(run_dir.glob("events.out.tfevents.*"))

    if not event_files:
        return {'train_loss': [], 'valid_loss': []}

    # Use the most recent event file
    event_file = sorted(event_files, key=lambda x: x.stat().st_mtime)[-1]

    try:
        ea = EventAccumulator(str(event_file))
        ea.Reload()

        losses = {}

        # Extract training loss
        if 'train/avg_total_loss' in ea.Tags()['scalars']:
            train_loss = ea.Scalars('train/avg_total_loss')
            losses['train_loss'] = [(int(s.step), s.value) for s in train_loss]
        else:
            losses['train_loss'] = []

        # Extract validation loss
        if 'valid/avg_total_loss' in ea.Tags()['scalars']:
            valid_loss = ea.Scalars('valid/avg_total_loss')
            losses['valid_loss'] = [(int(s.step), s.value) for s in valid_loss]
        else:
            losses['valid_loss'] = []

        return losses
    except Exception as e:
        print(f"Warning: Could not read TensorBoard file {event_file}: {e}")
        return {'train_loss': [], 'valid_loss': []}


def plot_loss_curves(run_dir: Path, save_path: Optional[Path] = None,
                     show_metrics: bool = False, figsize: Tuple[int, int] = (12, 6)) -> Path:
    """
    Generate and save a simplified loss curve plot for a NeuralHydrology run.

    Creates a single plot showing training and validation loss over epochs,
    with the best validation epoch clearly marked.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory
    save_path : Path, optional
        Where to save the plot. If None, saves to run_dir/loss_curves.png
    show_metrics : bool, optional
        Whether to include validation metrics (default: False for simplicity)
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches

    Returns
    -------
    Path
        Path where the plot was saved
    """
    run_dir = Path(run_dir)

    if save_path is None:
        save_path = run_dir / "loss_curves.png"

    # Extract data from TensorBoard
    tb_losses = extract_losses_from_tensorboard(run_dir)

    if not tb_losses['train_loss'] and not tb_losses['valid_loss']:
        # No TensorBoard data, try to extract from CSVs
        print(f"Warning: No TensorBoard data found for {run_dir.name}, attempting CSV extraction")
        train_df, valid_df = extract_metrics_from_csvs(run_dir)

        if valid_df.empty:
            print(f"Error: No training data found for {run_dir.name}")
            return None

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot training loss
    if tb_losses['train_loss']:
        train_epochs, train_losses = zip(*tb_losses['train_loss'])
        ax.plot(train_epochs, train_losses, 'b-', label='Training Loss',
                linewidth=2.5, alpha=0.8)

    # Plot validation loss
    if tb_losses['valid_loss']:
        valid_epochs, valid_losses = zip(*tb_losses['valid_loss'])
        ax.plot(valid_epochs, valid_losses, 'r-', label='Validation Loss',
                linewidth=2.5, alpha=0.8)

        # Mark the best validation loss
        best_idx = np.argmin(valid_losses)
        best_epoch = valid_epochs[best_idx]
        best_loss = valid_losses[best_idx]

        # Vertical line at best epoch
        ax.axvline(x=best_epoch, color='darkgreen', linestyle='--',
                   linewidth=2, alpha=0.6, label=f'Best Epoch ({best_epoch})')

        # Star marker at best point
        # Format loss value appropriately (use scientific notation if < 0.01)
        if abs(best_loss) < 0.01:
            loss_label = f'Best Val Loss: {best_loss:.2e}'
        else:
            loss_label = f'Best Val Loss: {best_loss:.4f}'

        ax.plot(best_epoch, best_loss, 'g*', markersize=20, markeredgecolor='darkgreen',
                markeredgewidth=1.5, label=loss_label)

    # Styling
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Progress: Loss Over Epochs', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss curve saved to: {save_path}")
    return save_path


def plot_all_runs_in_directory(parent_dir: Path, recursive: bool = True):
    """
    Generate loss curve plots for all run directories in a parent directory.

    Parameters
    ----------
    parent_dir : Path
        Parent directory containing run folders
    recursive : bool, optional
        Whether to search recursively for run directories
    """
    parent_dir = Path(parent_dir)

    # Find all directories with config.yml (indicating a run directory)
    if recursive:
        run_dirs = [p.parent for p in parent_dir.rglob("config.yml")]
    else:
        run_dirs = [p.parent for p in parent_dir.glob("*/config.yml")]

    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        try:
            plot_loss_curves(run_dir)
            print(f"  ✓ Processed: {run_dir.relative_to(parent_dir)}")
        except Exception as e:
            print(f"  ✗ Error processing {run_dir.relative_to(parent_dir)}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_loss_curves.py <run_directory_or_parent>")
        print("\nExamples:")
        print("  python plot_loss_curves.py outputs/calpella/daily_shared/runs/BASELINE_20250815T000000Z/testing_run_2906_213838")
        print("  python plot_loss_curves.py outputs  # Process all runs recursively")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    # Check if it's a run directory (has config.yml) or a parent directory
    if (path / "config.yml").exists():
        plot_loss_curves(path)
    else:
        plot_all_runs_in_directory(path)
