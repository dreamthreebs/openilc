from pathlib import Path

import numpy as np

from openilc import HILC, NILC, load_yaml_config
from tutorials.sim_data import estimate_lmax_from_beam, get_band_table, get_cmb_cls


def test_public_imports():
    assert NILC is not None
    assert HILC is not None


def test_external_config_paths_exist():
    assert Path("configs/default.yaml").exists()
    assert Path("configs/beam.yaml").exists()


def test_sim_data_shapes_and_beam_lmax_estimate():
    assert len(get_band_table()) == 5
    assert len(get_band_table(beam_version=True)) == 5
    assert get_cmb_cls(32).shape == (33, 4)
    assert np.isfinite(get_cmb_cls(32)).all()
    assert estimate_lmax_from_beam(63, 1000, bl_floor=1e-4) == 551


def test_yaml_config_loader():
    config = load_yaml_config("configs/beam.yaml")
    assert config["nilc"]["lmax"] == 1000
    assert config["bands"][0]["lmax_alm"] == 551
    assert config["needlets"][-1]["lmax"] == 1000
