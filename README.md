# UCB-USACE Russian River LSTM Forecasting

Deep learning streamflow forecasting for the Russian River basin (Northern California), developed at UC Berkeley in collaboration with the US Army Corps of Engineers (USACE) Hydrologic Engineering Center (HEC).

## Project Overview

This project evaluates LSTM and multi-timescale LSTM (MTS-LSTM) architectures for operational streamflow forecasting at four Russian River basin gauges, comparing pure data-driven models against physics-informed variants that incorporate HEC-HMS simulation outputs.

### Study Basins

| Basin | USGS Gauge | Upstream Management |
|-------|-----------|---------------------|
| **Guerneville** | Russian River at Guerneville | Most managed (Lake Mendocino + Potter Valley + Lake Sonoma) |
| **Hopland** | Russian River near Hopland | Moderate (Lake Mendocino + Potter Valley) |
| **Calpella** | East Fork Russian River near Calpella | Moderate (Lake Mendocino) |
| **Warm Springs** | Dry Creek near Geyserville | Least managed (Lake Sonoma only) |

### Model Configurations

- **LSTM**: Single-timescale daily or hourly models (`gage_nlayer` configs)
- **MTS-LSTM**: Multi-timescale models processing daily and hourly inputs simultaneously (`mtslstm2` configs)
- **PILSTM**: Physics-informed LSTM - standard LSTM with HEC-HMS outputs appended as additional input features (not true physics constraints)

### Key Experiments

- **CROSS_VAL_V5**: Production cross-validated hyperparameter search with plateau early stopping
- **EXTREME_SEQ_A**: Extreme year holdout - trains on 8 water years, validates on 2 (wet + dry), tests on 4 extreme years
- **BASELINE_NOBC**: Boundary condition ablation - tests whether LSTMs implicitly learn upstream management signals

## Repository Structure

```
UCB_training/           # Training framework (trainer, grid search, utilities, plotting)
  configs/              # YAML experiment configurations (4 basins x multiple modes)
neuralhydrology/        # Modified NeuralHydrology framework (custom datasets, MTS-LSTM)
  datasetzoo/           # Includes custom RussianRiver and SyntheticRussianRiver datasets
notebooks/
  basins/               # Per-basin experiment notebooks (grid search, training, evaluation)
  analysis/             # Cross-basin analysis and figures
run_hyperparam_search.py  # Standalone hyperparameter search orchestrator (grid/Bayesian, CV/no-CV)
```

## Setup

### Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- NeuralHydrology (included as modified fork)
- See `setup.cfg` for full dependency list

### Installation

```bash
git clone https://github.com/canruso/UCB-USACE-RR-PROJECT.git
cd UCB-USACE-RR-PROJECT
pip install -e .
```

### Data

Russian River forcing and streamflow data must be placed in the `data/` directory. Data is not included in this repository.

## Usage

### Notebook-based (interactive)

Each basin has a primary notebook (e.g., `notebooks/basins/calpella/calpella_mts-lstm2.ipynb`) that handles:
1. Hyperparameter grid search (parallelized via multiprocessing)
2. Best model selection (CV or fixed validation)
3. Final training and evaluation on test period

### Script-based (headless)

```bash
python run_hyperparam_search.py
```

Configure `BASIN`, `MODE`, `RUN_LABEL`, `USE_CV`, and `HYPERPARAM_SPACE` at the top of the script.

## Acknowledgments

This project is built on the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework by Kratzert et al. The framework has been modified to support custom Russian River datasets, multi-timescale configurations, and physics-informed input features.

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology - A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  year = {2022},
  volume = {7},
  number = {71},
  pages = {4050},
  doi = {10.21105/joss.04050},
  url = {https://doi.org/10.21105/joss.04050},
}
```

## License

This project extends NeuralHydrology, which is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
