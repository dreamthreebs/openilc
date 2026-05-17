# OPENILC

Internal Linear Combination tools for CMB data analysis.

OPENILC is split into three layers:

- `openilc/`: the installable core package, currently exposing `NILC` and `HILC`.
- External CSV configuration files under `configs/`.
- Tutorial helpers and scripts under `tutorials/`.

This keeps the algorithm reusable while leaving experiment settings easy to edit.

## Installation

Install the core package in editable mode:

```bash
pip install -e .
```

For the tutorial simulations that generate CMB spectra with CAMB and foregrounds
with PySM3, install the optional tutorial dependencies:

```bash
pip install -e ".[tutorial]"
```

## Quick Start

```python
import numpy as np
from openilc import NILC

maps = np.load("./test_data/sim_cfn.npy")

nilc = NILC.from_csv(
    "configs/bands.csv",
    "configs/needlets_default.csv",
    weights_name="./nilc_weight/w_map.npz",
    Sm_maps=maps,
    lmax=500,
    n_iter=1,
    weight_in_alm=False,
)

clean_map = nilc.run_nilc()
```

For beam-aware runs, use the beam configuration:

```python
nilc = NILC.from_csv(
    "configs/bands_beam.csv",
    "configs/needlets_beam.csv",
    Sm_maps=maps,
    lmax=1000,
    n_iter=1,
)
```

## Configuration

Configuration is intentionally outside the Python package. CSV is the
recommended format here because the configuration is naturally tabular and easy
to compare by eye.

- `configs/bands.csv`: basic frequency-channel configuration.
- `configs/bands_beam.csv`: beam-aware frequency-channel configuration.
- `configs/needlets_default.csv`: default needlet bins.
- `configs/needlets_beam.csv`: beam-aware needlet bins.

Important rules:

- The last needlet `lmax` should match the `lmax` passed to `NILC`.
- Each needlet row should connect smoothly with neighboring rows through
  `lmin`, `lpeak`, and `lmax`.
- `nside` should be large enough for the requested `lmax`; in practice, keep
  `lmax < 2 * nside`.
- `lmax_alm` controls which channels are usable at each needlet scale. In
  `NILC.calc_beta_for_scale`, channels with `lmax_alm < beta_lmax` are dropped.

Recommended CSV usage:

```python
nilc = NILC.from_csv(
    "configs/bands_beam.csv",
    "configs/needlets_beam.csv",
    Sm_maps=maps,
    weights_name="./nilc_weight/w_alm.npz",
    lmax=1000,
    n_iter=1,
)
```

## Simulated Data

Large binary data is not stored in this repository. Tutorial simulations are
generated at runtime by `tutorials/sim_data.py`:

- CMB spectra: CAMB with Planck 2018-like parameters.
- Foregrounds: PySM3 dust and synchrotron presets.
- Noise: Gaussian pixel noise using the configured `nstd`.

Useful helpers:

```python
from tutorials.sim_data import (
    estimate_lmax_from_beam,
    get_band_table,
    get_cmb_cls,
    get_foreground,
)
```

`estimate_lmax_from_beam(beam_arcmin, lmax, bl_floor=1e-4)` is kept for teaching:
it shows where a Gaussian beam transfer function becomes small enough that
deconvolution is numerically risky. The actual beam tutorial still uses the
configured `lmax_alm` from `configs/bands_beam.csv`.

## Tutorials

The tutorial scripts are examples, not pytest tests:

- `tutorials/tutorial_nilc.py`: basic NILC example using the default CSV configs.
- `tutorials/tutorial_nilc_beam.py`: beam-aware NILC example using the beam CSV configs.
- `tutorials/tutorial_ilc_bias.py`: ILC bias tutorial.
- `tutorials/tutorial_cpr_ilc.py`: CPR/HILC experiments.

Typical workflow:

```bash
cd tutorials
python tutorial_nilc.py
python tutorial_nilc_beam.py
```

These scripts write generated arrays into ignored local directories such as
`test_data/`, `test_data_beam/`, and `nilc_weight/`.

## Tests

Automated tests live under `tests/`:

```bash
python -m pytest tests
```

The tutorial scripts are named `tutorial_*.py` so pytest does not collect them as
unit tests.

## Import

Use the package import path:

```python
from openilc import NILC, HILC
```

## Known Issues

- NILC currently has large memory usage.
- Calculating the beta covariance matrix can be slow.

## High Performance Computing

Spherical harmonic transforms can be accelerated by building `healpy` from
optimized native libraries. See the healpy installation notes, then install
`cfitsio`, `healpix`, and rebuild `healpy` from source if needed.
