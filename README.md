# OPENILC

Internal Linear Combination tools for CMB data analysis.

OPENILC is split into three layers:

- `openilc/`: the installable core package, currently exposing `NILC` and `HILC`.
- External YAML configuration files under `configs/`.
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

nilc = NILC.from_config(
    "configs/default.yaml",
    weights_name="./nilc_weight/w_map.npz",
    Sm_maps=maps,
    n_iter=1,
    weight_in_alm=False,
)

clean_map = nilc.run_nilc()
```

For beam-aware runs, use the beam configuration:

```python
nilc = NILC.from_config(
    "configs/beam.yaml",
    Sm_maps=maps,
    n_iter=1,
)
```

## Configuration

Configuration is intentionally outside the Python package. YAML is the
recommended format because it is readable, editable, and can include comments.

- `configs/default.yaml`: basic NILC tutorial configuration.
- `configs/beam.yaml`: beam-aware NILC tutorial configuration.

Important rules:

- The last needlet `lmax` should match the `lmax` passed to `NILC`.
- Each needlet row should connect smoothly with neighboring rows through
  `lmin`, `lpeak`, and `lmax`.
- `nside` should be large enough for the requested `lmax`; in practice, keep
  `lmax < 2 * nside`.
- `lmax_alm` controls which channels are usable at each needlet scale. In
  `NILC.calc_beta_for_scale`, channels with `lmax_alm < beta_lmax` are dropped.

Recommended YAML usage:

```python
nilc = NILC.from_config(
    "configs/beam.yaml",
    Sm_maps=maps,
    weights_name="./nilc_weight/w_alm.npz",
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
configured `lmax_alm` from `configs/beam.yaml`.

## Tutorials

The tutorial scripts are examples, not pytest tests:

- `tutorials/tutorial_nilc.py`: basic NILC example using `configs/default.yaml`.
- `tutorials/tutorial_nilc_beam.py`: beam-aware NILC example using `configs/beam.yaml`.
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
