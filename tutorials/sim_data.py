from functools import lru_cache

import camb
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u


BANDS = (
    {"freq": 40, "mapdepth": 10.6, "beam": 63, "nstd": 1.5427, "lmax_alm": 2000},
    {"freq": 95, "mapdepth": 1.35, "beam": 30, "nstd": 0.1965, "lmax_alm": 2000},
    {"freq": 155, "mapdepth": 1.24, "beam": 17, "nstd": 0.1804, "lmax_alm": 2000},
    {"freq": 215, "mapdepth": 1.9, "beam": 11, "nstd": 0.2765, "lmax_alm": 2000},
    {"freq": 270, "mapdepth": 1.64, "beam": 9, "nstd": 0.2387, "lmax_alm": 2000},
)

BEAM_BANDS = (
    {"freq": 40, "mapdepth": 10.6, "beam": 63, "nstd": 1.5427, "lmax_alm": 551},
    {"freq": 95, "mapdepth": 1.35, "beam": 30, "nstd": 0.1965, "lmax_alm": 1158},
    {"freq": 155, "mapdepth": 1.24, "beam": 17, "nstd": 0.1804, "lmax_alm": 2044},
    {"freq": 215, "mapdepth": 1.9, "beam": 11, "nstd": 0.2765, "lmax_alm": 3159},
    {"freq": 270, "mapdepth": 1.64, "beam": 9, "nstd": 0.2387, "lmax_alm": 3860},
)


def get_band_table(beam_version=False):
    return BEAM_BANDS if beam_version else BANDS


def estimate_lmax_from_beam(beam_arcmin, lmax, bl_floor=1e-4):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_arcmin) / 60, lmax=lmax)
    return np.argmax(bl < bl_floor) if np.any(bl < bl_floor) else lmax


@lru_cache(maxsize=None)
def get_cmb_cls(lmax=8000):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544)
    pars.InitPower.set_params(As=np.exp(3.044) / 1e10, ns=0.9649)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    spectra = results.get_cmb_power_spectra(
        pars, CMB_unit="muK", raw_cl=True
    )["total"]
    return np.asarray(spectra[: lmax + 1])


@lru_cache(maxsize=None)
def _sky(nside):
    return pysm3.Sky(
        nside=nside,
        preset_strings=["d1", "s1"],
        output_unit="uK_CMB",
    )


def get_foreground(freq_ghz, nside):
    return _sky(nside).get_emission(freq_ghz * u.GHz)[0].value
