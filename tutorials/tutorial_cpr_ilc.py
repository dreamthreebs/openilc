import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time
import argparse

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from openilc import HILC, NILC
from tutorials.sim_data import get_band_table, get_cmb_cls, get_foreground

nside = 512
npix = hp.nside2npix(nside)
lmax = 500
R_tol = 1 / 100
TEST_DATA = Path("./test_data")
NILC_WEIGHT = Path("./nilc_weight")
MAP_RTOLS = (1 / 100, 1 / 500, 1 / 1000)
BIAS_RTOLS = (1 / 100, 1 / 1000)


def _rtol_name(rtol):
    return str(rtol)


def _map_result_paths(rtol):
    name = _rtol_name(rtol)
    paths = [
        TEST_DATA / f"cln_cmb_w_map_{name}.npy",
        TEST_DATA / f"fg_res_w_map_{name}.npy",
        TEST_DATA / f"n_res_w_map_{name}.npy",
    ]
    if rtol in BIAS_RTOLS:
        paths.append(TEST_DATA / f"c_res_w_map_{name}.npy")
    return paths


def _missing(paths):
    return [path for path in paths if not path.exists()]


def gen_sim():
    # generate cmb + foreground + noise simulation, cmb is T mode and no beam on different frequency
    sim_list = []
    fg_list = []
    n_list = []
    bands = get_band_table()
    cl_cmb = get_cmb_cls(lmax=8000).T[0]
    print(f"{cl_cmb.shape=}")

    np.random.seed(seed=0)
    cmb = hp.synfast(cls=cl_cmb, nside=nside)

    for band in bands:
        freq = band["freq"]
        beam = band["beam"]
        print(f"{freq=}, {beam=}")

        fg = get_foreground(freq, nside)
        nstd = band["nstd"]
        noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))

        sim = cmb + fg + noise
        sim_list.append(sim)
        fg_list.append(fg)
        n_list.append(noise)

    TEST_DATA.mkdir(exist_ok=True, parents=True)
    np.save("./test_data/sim_cfn.npy", sim_list)
    np.save("./test_data/sim_c.npy", cmb)
    np.save("./test_data/sim_f.npy", fg_list)
    np.save("./test_data/sim_n.npy", n_list)


def gen_sim_alm():
    # generate cmb + foreground + noise simulation, cmb is T mode and no beam on different frequency
    sim_list = []
    fg_list = []
    n_list = []
    bands = get_band_table()
    cl_cmb = get_cmb_cls(lmax=8000).T[0]
    print(f"{cl_cmb.shape=}")

    np.random.seed(seed=0)
    cmb = hp.synfast(cls=cl_cmb, nside=nside)
    cmb_alm = hp.map2alm(cmb, lmax=lmax)

    for band in bands:
        freq = band["freq"]
        beam = band["beam"]
        print(f"{freq=}, {beam=}")

        fg = get_foreground(freq, nside)
        fg_alm = hp.map2alm(fg, lmax=lmax)
        nstd = band["nstd"]
        noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))
        noise_alm = hp.map2alm(noise, lmax=lmax)

        sim = cmb + fg + noise
        sim_alm = hp.map2alm(sim, lmax=lmax)
        sim_list.append(sim_alm)
        fg_list.append(fg_alm)
        n_list.append(noise_alm)

    TEST_DATA.mkdir(exist_ok=True, parents=True)
    np.save("./test_data/sim_cfn_alm.npy", sim_list)
    np.save("./test_data/sim_c_alm.npy", cmb_alm)
    np.save("./test_data/sim_f_alm.npy", fg_list)
    np.save("./test_data/sim_n_alm.npy", n_list)


def run_hilc():
    # do nilc and save the weights in map
    time0 = time.time()
    sim = np.load("./test_data/sim_cfn_alm.npy")
    obj_hilc = HILC(
        deltal=60, alms=sim, lmax=lmax, freqs=[30, 95, 155, 215, 270], fsky=1
    )
    alm_cln, weight = obj_hilc.do_CILC()
    cl_cln = hp.alm2cl(alm_cln, lmax=lmax)

    cmb = np.asarray([np.load("./test_data/sim_c_alm.npy")] * 5)
    obj_cmb = HILC(
        deltal=120,
        alms=cmb,
        lmax=lmax,
        freqs=[30, 95, 155, 215, 270],
        fsky=1,
        weight=weight,
    )
    alm_cmb_res, _ = obj_cmb.do_CILC()

    fg = np.load(f"./test_data/sim_f_alm.npy")
    obj_fg = HILC(
        deltal=120,
        alms=fg,
        lmax=lmax,
        freqs=[30, 95, 155, 215, 270],
        fsky=1,
        weight=weight,
    )
    alm_fg_res, _ = obj_fg.do_CILC()
    cl_fg_res = hp.alm2cl(alm_fg_res, lmax=lmax)

    noise = np.load(f"./test_data/sim_n_alm.npy")
    obj_n = HILC(
        deltal=120,
        alms=noise,
        lmax=lmax,
        freqs=[30, 95, 155, 215, 270],
        fsky=1,
        weight=weight,
    )
    alm_n_res, _ = obj_n.do_CILC()
    cl_n_res = hp.alm2cl(alm_n_res, lmax=lmax)

    cl_ilc_bias = 2 * hp.alm2cl(alm_cmb_res, alm_fg_res + alm_n_res, lmax=lmax)
    print(f"{time.time()-time0=}")

    return cl_cln, cl_fg_res, cl_n_res, cl_ilc_bias


def run_nilc_w_alm():
    # do nilc and save the weights in alm
    time0 = time.time()
    sim = np.load("./test_data/sim_cfn.npy")
    obj_nilc = NILC.from_csv("configs/bands.csv", "configs/needlets_default.csv",
        weights_name="./nilc_weight/w_alm.npz",
        Sm_maps=sim,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        Rtol=R_tol,
    )
    clean_map = obj_nilc.run_nilc()
    np.save("./test_data/cln_cmb_w_alm.npy", clean_map)
    print(f"{time.time()-time0=}")


def get_fg_res_w_alm():
    # this function tells you how to debias other component by using the weights in alm
    fg = np.load("./test_data/sim_f.npy")
    obj_nilc = NILC.from_csv("configs/bands.csv", "configs/needlets_default.csv",
        weights_config="./nilc_weight/w_alm.npz",
        Sm_maps=fg,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
    )
    fg_res = obj_nilc.run_nilc()
    np.save("./test_data/fg_res_w_alm.npy", fg_res)


def get_n_res_w_alm():
    # calc noise bias
    noise = np.load("./test_data/sim_n.npy")
    obj_nilc = NILC.from_csv("configs/bands.csv", "configs/needlets_default.csv",
        weights_config="./nilc_weight/w_alm.npz",
        Sm_maps=noise,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
    )
    fg_res = obj_nilc.run_nilc()
    np.save("./test_data/n_res_w_alm.npy", fg_res)


def run_nilc_w_map(R_tol=1 / 1000):
    # do nilc and save the weights in map
    time0 = time.time()
    sim = np.load("./test_data/sim_cfn.npy")
    obj_nilc = NILC.from_csv("configs/bands.csv", "configs/needlets_default.csv",
        weights_name="./nilc_weight/w_map.npz",
        Sm_maps=sim,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        weight_in_alm=False,
        Rtol=R_tol,
    )
    clean_map = obj_nilc.run_nilc()
    np.save(f"./test_data/cln_cmb_w_map_{R_tol}.npy", clean_map)
    print(f"{time.time()-time0=}")


def get_fg_res_w_map(R_tol=1 / 1000):
    # this function tells you how to debias other component by using the weights in map
    fg = np.load("./test_data/sim_f.npy")
    obj_nilc = NILC.from_csv("configs/bands.csv", "configs/needlets_default.csv",
        weights_config="./nilc_weight/w_map.npz",
        Sm_maps=fg,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        weight_in_alm=False,
    )
    fg_res = obj_nilc.run_nilc()
    np.save(f"./test_data/fg_res_w_map_{R_tol}.npy", fg_res)


def get_n_res_w_map(R_tol=1 / 1000):
    # calc noise bias
    noise = np.load("./test_data/sim_n.npy")
    obj_nilc = NILC.from_csv("configs/bands.csv", "configs/needlets_default.csv",
        weights_config="./nilc_weight/w_map.npz",
        Sm_maps=noise,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        weight_in_alm=False,
    )
    fg_res = obj_nilc.run_nilc()
    np.save(f"./test_data/n_res_w_map_{R_tol}.npy", fg_res)


def get_cmb_res_w_map(R_tol=1 / 1000):
    # calc noise bias
    cmb = np.asarray([np.load("./test_data/sim_c.npy")] * 5)
    obj_nilc = NILC.from_csv("configs/bands.csv", "configs/needlets_default.csv",
        weights_config="./nilc_weight/w_map.npz",
        Sm_maps=cmb,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        weight_in_alm=False,
    )
    cmb_res = obj_nilc.run_nilc()
    np.save(f"./test_data/c_res_w_map_{R_tol}.npy", cmb_res)


def prepare_check_inputs(force=False):
    map_inputs = [
        TEST_DATA / "sim_cfn.npy",
        TEST_DATA / "sim_c.npy",
        TEST_DATA / "sim_f.npy",
        TEST_DATA / "sim_n.npy",
    ]
    alm_inputs = [
        TEST_DATA / "sim_cfn_alm.npy",
        TEST_DATA / "sim_c_alm.npy",
        TEST_DATA / "sim_f_alm.npy",
        TEST_DATA / "sim_n_alm.npy",
    ]

    if force or _missing(map_inputs):
        print("Generating map-space simulations...")
        gen_sim()
    if force or _missing(alm_inputs):
        print("Generating harmonic-space simulations...")
        gen_sim_alm()

    NILC_WEIGHT.mkdir(exist_ok=True, parents=True)
    for rtol in MAP_RTOLS:
        if not force and not _missing(_map_result_paths(rtol)):
            continue

        print(f"Generating NILC map-weight products for R_tol={rtol}...")
        run_nilc_w_map(R_tol=rtol)
        get_fg_res_w_map(R_tol=rtol)
        get_n_res_w_map(R_tol=rtol)
        if rtol in BIAS_RTOLS:
            get_cmb_res_w_map(R_tol=rtol)


def validate_check_inputs():
    required = [
        TEST_DATA / "sim_c.npy",
        TEST_DATA / "sim_cfn_alm.npy",
        TEST_DATA / "sim_c_alm.npy",
        TEST_DATA / "sim_f_alm.npy",
        TEST_DATA / "sim_n_alm.npy",
    ]
    for rtol in MAP_RTOLS:
        required.extend(_map_result_paths(rtol))

    missing = _missing(required)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "missing CPR tutorial input files:\n"
            f"{formatted}\n"
            "Run `python tutorials/tutorial_cpr_ilc.py --prepare-only` first, "
            "or run without `--check-only` to generate them automatically."
        )


def check_res():
    # check result compared with input and the foreground residual and noise bias
    # save weights in alm or map should not affect the final results so there lines in plots are overlapped
    validate_check_inputs()

    l = np.arange(lmax + 1)

    cmb = np.load("./test_data/sim_c.npy")
    cl_cmb = hp.anafast(cmb, lmax=lmax)

    plt.loglog(l * (l + 1) * cl_cmb / (2 * np.pi), label="input cmb")

    R_tol = 1 / 100
    cln_cmb_map = np.load(f"./test_data/cln_cmb_w_map_{R_tol}.npy")
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    # hp.gnomview(cln_cmb_map, rot=[0, 90, 0], title="cln cmb map")
    plt.loglog(
        l * (l + 1) * cl_cln_map / (2 * np.pi),
        label=f"after nilc map weight map {R_tol=}",
    )

    fg_res_map = np.load(f"./test_data/fg_res_w_map_{R_tol}.npy")
    cl_fgres_map = hp.anafast(fg_res_map, lmax=lmax)
    plt.loglog(
        l * (l + 1) * cl_fgres_map / (2 * np.pi),
        label=f"foreground residual weight map {R_tol=}",
    )

    # hp.gnomview(fg_res_map, rot=[0, 90, 0], title="fg res map")

    n_res_map = np.load(f"./test_data/n_res_w_map_{R_tol}.npy")
    cl_nres_map = hp.anafast(n_res_map, lmax=lmax)
    plt.loglog(
        l * (l + 1) * cl_nres_map / (2 * np.pi),
        label=f"noise residual weight map {R_tol=}",
    )
    cmb_res_map = np.load(f"./test_data/c_res_w_map_{R_tol}.npy")
    # cl_c_res_map = hp.anafast(cmb_res_map, lmax=lmax)
    cl_nilc_bias = 2 * hp.anafast(cmb_res_map, fg_res_map + n_res_map, lmax=lmax)

    plt.loglog(
        l * (l + 1) * np.abs(cl_nilc_bias) / (2 * np.pi),
        label=f"nilc bias {R_tol=}",
    )

    R_tol = 1 / 500
    cln_cmb_map = np.load(f"./test_data/cln_cmb_w_map_{R_tol}.npy")
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    # hp.gnomview(cln_cmb_map, rot=[0, 90, 0], title="cln cmb map")
    plt.loglog(
        l * (l + 1) * cl_cln_map / (2 * np.pi),
        label=f"after nilc map weight map {R_tol=}",
    )

    fg_res_map = np.load(f"./test_data/fg_res_w_map_{R_tol}.npy")
    cl_fgres_map = hp.anafast(fg_res_map, lmax=lmax)
    plt.loglog(
        l * (l + 1) * cl_fgres_map / (2 * np.pi),
        label=f"foreground residual weight map {R_tol=}",
    )

    # hp.gnomview(fg_res_map, rot=[0, 90, 0], title="fg res map")

    n_res_map = np.load(f"./test_data/n_res_w_map_{R_tol}.npy")
    cl_nres_map = hp.anafast(n_res_map, lmax=lmax)
    plt.loglog(
        l * (l + 1) * cl_nres_map / (2 * np.pi),
        label=f"noise residual weight map {R_tol=}",
    )

    R_tol = 1 / 1000
    cln_cmb_map = np.load(f"./test_data/cln_cmb_w_map_{R_tol}.npy")
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    # hp.gnomview(cln_cmb_map, rot=[0, 90, 0], title="cln cmb map")
    plt.loglog(
        l * (l + 1) * cl_cln_map / (2 * np.pi),
        label=f"after nilc map weight map {R_tol=}",
    )

    fg_res_map = np.load(f"./test_data/fg_res_w_map_{R_tol}.npy")
    cl_fgres_map = hp.anafast(fg_res_map, lmax=lmax)
    plt.loglog(
        l * (l + 1) * cl_fgres_map / (2 * np.pi),
        label=f"foreground residual weight map {R_tol=}",
    )

    # hp.gnomview(fg_res_map, rot=[0, 90, 0], title="fg res map")

    n_res_map = np.load(f"./test_data/n_res_w_map_{R_tol}.npy")
    cl_nres_map = hp.anafast(n_res_map, lmax=lmax)
    plt.loglog(
        l * (l + 1) * cl_nres_map / (2 * np.pi),
        label=f"noise residual weight map {R_tol=}",
    )

    cmb_res_map = np.load(f"./test_data/c_res_w_map_{R_tol}.npy")
    # cl_c_res_map = hp.anafast(cmb_res_map, lmax=lmax)
    cl_nilc_bias = 2 * hp.anafast(cmb_res_map, fg_res_map + n_res_map, lmax=lmax)

    plt.loglog(
        l * (l + 1) * np.abs(cl_nilc_bias) / (2 * np.pi),
        label=f"nilc bias {R_tol=}",
    )

    plt.loglog(
        np.abs(cl_nilc_bias) / cl_cmb,
        label=f"fractional nilc bias {R_tol=}",
    )

    print(f"{np.mean(np.abs(cl_nilc_bias[50:400]) / cl_cmb[50:400])=}")

    cl_hilc, cl_hilc_fg_res, cl_hilc_n_res, cl_hilc_bias = run_hilc()
    plt.loglog(
        l * (l + 1) * cl_hilc / (2 * np.pi),
        label="clean hilc",
    )

    plt.loglog(
        l * (l + 1) * cl_hilc_fg_res / (2 * np.pi),
        label="hilc fgres",
    )

    plt.loglog(
        l * (l + 1) * cl_hilc_n_res / (2 * np.pi),
        label="hilc noise res",
    )

    plt.loglog(
        l * (l + 1) * np.abs(cl_hilc_bias) / (2 * np.pi),
        label="hilc bias",
    )

    plt.loglog(
        np.abs(cl_hilc_bias) / cl_cmb,
        label="fractional hilc bias",
    )

    print(f"{np.mean(np.abs(cl_hilc_bias[50:400]) / cl_cmb[50:400])=}")
    # hp.gnomview(n_res_map, rot=[0, 90, 0], title="noise res map")
    # plt.show()

    plt.xlabel("$\\ell$")
    plt.ylabel("$D_\\ell [\\mu K^2]$")

    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run the CPR/HILC tutorial and generate missing intermediate files."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="generate missing simulation and NILC residual files, then exit",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only read existing files and plot/check results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate intermediate files even if they already exist",
    )
    args = parser.parse_args()

    if not args.check_only:
        prepare_check_inputs(force=args.force)
    if not args.prepare_only:
        check_res()


if __name__ == "__main__":
    main()
