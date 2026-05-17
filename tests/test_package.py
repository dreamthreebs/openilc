from pathlib import Path

import numpy as np
import pytest
import healpy as hp

from openilc import HILC, NILC, get_sht_backend, load_csv_table
from tutorials.sim_data import estimate_lmax_from_beam, get_band_table, get_cmb_cls


def test_public_imports():
    assert NILC is not None
    assert HILC is not None


def test_external_config_paths_exist():
    assert Path("configs/bands.csv").exists()
    assert Path("configs/bands_beam.csv").exists()
    assert Path("configs/needlets_default.csv").exists()
    assert Path("configs/needlets_beam.csv").exists()


def test_sim_data_shapes_and_beam_lmax_estimate():
    assert len(get_band_table()) == 5
    assert len(get_band_table(beam_version=True)) == 5
    assert get_cmb_cls(32).shape == (33, 4)
    assert np.isfinite(get_cmb_cls(32)).all()
    assert estimate_lmax_from_beam(63, 1000, bl_floor=1e-4) == 551


def test_csv_config_loader():
    bands = load_csv_table("configs/bands_beam.csv")
    needlets = load_csv_table("configs/needlets_beam.csv")
    assert bands.at[0, "lmax_alm"] == 551
    assert needlets.at[len(needlets) - 1, "lmax"] == 1000


def test_nilc_validates_lmax_matches_needlets():
    bands = [{"lmax_alm": 9}, {"lmax_alm": 9}]
    needlets = [{"lmin": 0, "lpeak": 0, "lmax": 8, "nside": 8}]
    alms = np.zeros((2, hp.Alm.getsize(9)), dtype=complex)

    with pytest.raises(ValueError, match="last needlet lmax"):
        NILC(bands, needlets, Sm_alms=alms, lmax=9, nside=8)


def test_nilc_validates_channel_count_matches_bandinfo():
    bands = [{"lmax_alm": 8}]
    needlets = [{"lmin": 0, "lpeak": 0, "lmax": 8, "nside": 8}]
    alms = np.zeros((2, hp.Alm.getsize(8)), dtype=complex)

    with pytest.raises(ValueError, match="bandinfo rows"):
        NILC(bands, needlets, Sm_alms=alms, lmax=8, nside=8)


def test_nilc_validates_mask_shape():
    bands = [{"lmax_alm": 8}, {"lmax_alm": 8}]
    needlets = [{"lmin": 0, "lpeak": 0, "lmax": 8, "nside": 8}]
    maps = np.zeros((2, hp.nside2npix(8)))
    mask = np.ones(hp.nside2npix(4))

    with pytest.raises(ValueError, match="mask must have shape"):
        NILC(bands, needlets, Sm_maps=maps, mask=mask, lmax=8)


def test_nilc_validates_active_channels_per_needlet_scale():
    bands = [{"lmax_alm": 4}, {"lmax_alm": 8}]
    needlets = [{"lmin": 0, "lpeak": 0, "lmax": 8, "nside": 8}]
    alms = np.zeros((2, hp.Alm.getsize(8)), dtype=complex)

    with pytest.raises(ValueError, match="at least 2 channels"):
        NILC(bands, needlets, Sm_alms=alms, lmax=8, nside=8)


def test_ducc0_sht_backend_matches_healpy_iter0():
    ducc0 = pytest.importorskip("ducc0")
    assert ducc0 is not None

    rng = np.random.default_rng(0)
    nside = 16
    lmax = 32
    alm = (
        rng.normal(size=hp.Alm.getsize(lmax))
        + 1j * rng.normal(size=hp.Alm.getsize(lmax))
    )

    healpy_sht = get_sht_backend("healpy")
    ducc_sht = get_sht_backend("ducc0")

    m_healpy = healpy_sht.alm2map(alm, nside)
    m_ducc = ducc_sht.alm2map(alm, nside)
    np.testing.assert_allclose(m_ducc, m_healpy, rtol=0, atol=1e-10)

    alm_healpy = healpy_sht.map2alm(m_healpy, lmax=lmax, n_iter=0)
    alm_ducc = ducc_sht.map2alm(m_healpy, lmax=lmax, n_iter=0)
    np.testing.assert_allclose(alm_ducc, alm_healpy, rtol=1e-10, atol=1e-10)


def test_ducc0_pseudo_sht_backend_recovers_synthetic_alm():
    ducc0 = pytest.importorskip("ducc0")
    assert ducc0 is not None

    nside = 16
    lmax = 16
    cl = np.ones(lmax + 1)
    cl[0] = 0
    np.random.seed(1)
    alm = hp.synalm(cl, lmax=lmax, new=True)

    ducc_sht = get_sht_backend("ducc0")
    ducc_pseudo_sht = get_sht_backend("ducc0_pseudo")

    m = ducc_sht.alm2map(alm, nside)
    alm_adjoint = ducc_sht.map2alm(m, lmax=lmax, n_iter=0)
    alm_pseudo = ducc_pseudo_sht.map2alm(m, lmax=lmax, n_iter=5)

    adjoint_error = np.linalg.norm(alm_adjoint - alm) / np.linalg.norm(alm)
    pseudo_error = np.linalg.norm(alm_pseudo - alm) / np.linalg.norm(alm)
    assert pseudo_error < adjoint_error
    assert pseudo_error < 1e-6
