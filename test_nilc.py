import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import time

from pathlib import Path
from nilc import NILC

# ==== NEW: PySM imports ====
import pysm3
import pysm3.units as u
# ===========================

nside = 512
npix = hp.nside2npix(nside)
lmax = 500
R_tol = 1 / 100


def gen_sim():
    """
    generate cmb + foreground(d0+s0) + noise simulations

    - CMB: scalar T field from input Cl, no beam
    - FG: PySM3 Sky with dust d0 + synchrotron s0, unit uK_CMB, no beam
    - Noise: Gaussian pixel noise with nstd from FreqBand5
    """
    sim_list = []
    fg_list = []
    n_list = []

    df = pd.read_csv("./data/FreqBand5")
    cl_cmb = np.load("./data/cmb/cmbcl_8k.npy").T[0]
    print(f"{cl_cmb.shape=}")

    # --- CMB simulation (T only) ---
    np.random.seed(seed=0)
    cmb = hp.synfast(cls=cl_cmb, nside=nside)

    # --- NEW: build PySM sky with d0+s0 ---
    # Output unit chosen to match CMB map units (uK_CMB)
    sky = pysm3.Sky(
        nside=nside,
        preset_strings=["d1", "s1"],
        output_unit="uK_CMB",
    )

    for freq_idx in range(len(df)):
        freq = df.at[freq_idx, "freq"]  # in GHz
        beam = df.at[freq_idx, "beam"]
        print(f"{freq=}, {beam=}")

        # --- NEW: foreground from PySM d0+s0 at this frequency ---
        # sky.get_emission returns (I,Q,U) in uK_CMB as an astropy Quantity
        fg_maps = sky.get_emission(freq * u.GHz)  # shape (3, npix)
        fg = fg_maps[0].value  # use I only, drop units

        # --- Noise map ---
        nstd = df.at[freq_idx, "nstd"]
        noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))

        # Combined channel map
        sim = cmb + fg + noise

        sim_list.append(sim)
        fg_list.append(fg)
        n_list.append(noise)

    Path("./test_data").mkdir(exist_ok=True, parents=True)
    np.save("./test_data/sim_cfn.npy", np.array(sim_list))
    np.save("./test_data/sim_c.npy", cmb)
    np.save("./test_data/sim_f.npy", np.array(fg_list))
    np.save("./test_data/sim_n.npy", np.array(n_list))


def test_nilc_w_alm():
    # do nilc and save the weights in alm
    time0 = time.time()
    sim = np.load("./test_data/sim_cfn.npy")
    obj_nilc = NILC(
        needlet_config="./needlets/default.csv",
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
    obj_nilc = NILC(
        needlet_config="./needlets/default.csv",
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
    obj_nilc = NILC(
        needlet_config="./needlets/default.csv",
        weights_config="./nilc_weight/w_alm.npz",
        Sm_maps=noise,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
    )
    fg_res = obj_nilc.run_nilc()
    np.save("./test_data/n_res_w_alm.npy", fg_res)


def test_nilc_w_map():
    # do nilc and save the weights in map
    time0 = time.time()
    sim = np.load("./test_data/sim_cfn.npy")
    obj_nilc = NILC(
        needlet_config="./needlets/default.csv",
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
    np.save("./test_data/cln_cmb_w_map.npy", clean_map)
    print(f"{time.time()-time0=}")


def get_fg_res_w_map():
    # this function tells you how to debias other component by using the weights in map
    fg = np.load("./test_data/sim_f.npy")
    obj_nilc = NILC(
        needlet_config="./needlets/default.csv",
        weights_config="./nilc_weight/w_map.npz",
        Sm_maps=fg,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        weight_in_alm=False,
    )
    fg_res = obj_nilc.run_nilc()
    np.save("./test_data/fg_res_w_map.npy", fg_res)


def get_n_res_w_map():
    # calc noise bias
    noise = np.load("./test_data/sim_n.npy")
    obj_nilc = NILC(
        needlet_config="./needlets/default.csv",
        weights_config="./nilc_weight/w_map.npz",
        Sm_maps=noise,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        weight_in_alm=False,
    )
    fg_res = obj_nilc.run_nilc()
    np.save("./test_data/n_res_w_map.npy", fg_res)


def get_cmb_res_w_map():
    # calc noise bias
    cmb = np.asarray([np.load("./test_data/sim_c.npy")] * 5)
    obj_nilc = NILC(
        needlet_config="./needlets/default.csv",
        weights_config="./nilc_weight/w_map.npz",
        Sm_maps=cmb,
        mask=None,
        lmax=lmax,
        nside=nside,
        n_iter=1,
        weight_in_alm=False,
    )
    cmb_res = obj_nilc.run_nilc()
    np.save("./test_data/c_res_w_map.npy", cmb_res)


def check_res():
    # check result compared with input and the foreground residual and noise bias

    l = np.arange(lmax + 1)

    cmb = np.load("./test_data/sim_c.npy")
    cl_cmb = hp.anafast(cmb, lmax=lmax)

    cln_cmb = np.load("./test_data/cln_cmb_w_alm.npy")
    cl_cln = hp.anafast(cln_cmb, lmax=lmax)

    fg_res = np.load("./test_data/fg_res_w_alm.npy")
    cl_fgres = hp.anafast(fg_res, lmax=lmax)

    n_res = np.load("./test_data/n_res_w_alm.npy")
    cl_nres = hp.anafast(n_res, lmax=lmax)

    cln_cmb_map = np.load("./test_data/cln_cmb_w_map.npy")
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    hp.gnomview(cln_cmb_map, rot=[0, 90, 0], title="cln cmb map")

    fg_res_map = np.load("./test_data/fg_res_w_map.npy")
    cl_fgres_map = hp.anafast(fg_res_map, lmax=lmax)
    hp.gnomview(fg_res_map, rot=[0, 90, 0], title="fg res map")

    n_res_map = np.load("./test_data/n_res_w_map.npy")
    cl_nres_map = hp.anafast(n_res_map, lmax=lmax)
    hp.gnomview(n_res_map, rot=[0, 90, 0], title="noise res map")
    plt.show()

    plt.loglog(l * (l + 1) * cl_cmb / (2 * np.pi), label="input cmb")
    plt.loglog(l * (l + 1) * cl_cln / (2 * np.pi), label="after nilc weight alm")
    plt.loglog(
        l * (l + 1) * cl_fgres / (2 * np.pi), label="foreground residual weight alm"
    )
    plt.loglog(l * (l + 1) * cl_nres / (2 * np.pi), label="noise residual weight alm")

    plt.loglog(
        l * (l + 1) * cl_cln_map / (2 * np.pi), label="after nilc map weight map"
    )
    plt.loglog(
        l * (l + 1) * cl_fgres_map / (2 * np.pi), label="foreground residual weight map"
    )
    plt.loglog(
        l * (l + 1) * cl_nres_map / (2 * np.pi), label="noise residual weight map"
    )
    plt.loglog(
        l * (l + 1) * (cl_cln_map - cl_nres_map) / (2 * np.pi),
        label="debiased weight map",
    )

    plt.xlabel("$\\ell$")
    plt.ylabel("$D_\\ell [\\mu K^2]$")

    plt.legend()
    plt.show()


def main():
    gen_sim()

    test_nilc_w_alm()
    get_fg_res_w_alm()
    get_n_res_w_alm()
    test_nilc_w_map()
    get_fg_res_w_map()
    get_n_res_w_map()

    check_res()


if __name__ == "__main__":
    main()
