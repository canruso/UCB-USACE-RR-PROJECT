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
