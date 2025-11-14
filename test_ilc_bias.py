import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import time

from pathlib import Path
from nilc import NILC
# from memory_profiler import profile

nside = 512
npix = hp.nside2npix(nside)
lmax = 500
R_tol = 1 / 1000


def gen_sim():
    # generate cmb + foreground + noise simulation, cmb is T mode and no beam on different frequency
    sim_list = []
    n_list = []
    # df = pd.read_csv('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/FreqBand5') # for ali users
    # cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T[0] # for ali users
    df = pd.read_csv("./data/FreqBand5")
    cl_cmb = np.load("./data/cmb/cmbcl_8k.npy").T[0]
    print(f"{cl_cmb.shape=}")

    np.random.seed(seed=0)
    cmb = hp.synfast(cls=cl_cmb, nside=nside)

    for freq_idx in range(len(df)):
        freq = df.at[freq_idx, "freq"]
        beam = df.at[freq_idx, "beam"]
        print(f"{freq=}, {beam=}")

        # fg = np.load(f'/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/FG5/{freq}.npy')[0] # for ali users
        # nstd = np.load(f'/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/NSTDNORTH5/{freq}.npy')[0] # for ali users
        nstd = df.at[freq_idx, "nstd"]
        noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))

        sim = cmb + noise
        sim_list.append(sim)
        n_list.append(noise)

    Path("./test_data").mkdir(exist_ok=True, parents=True)
    np.save("./test_data/sim_cn.npy", sim_list)
    np.save("./test_data/sim_c.npy", cmb)
    np.save("./test_data/sim_n.npy", n_list)


def test_nilc_w_map():
    # do nilc and save the weights in map
    time0 = time.time()
    sim = np.load("./test_data/sim_cn.npy")
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
    # save weights in alm or map should not affect the final results so there lines in plots are overlapped

    l = np.arange(lmax + 1)

    cmb = np.load("./test_data/sim_c.npy")
    cl_cmb = hp.anafast(cmb, lmax=lmax)

    cln_cmb_map = np.load("./test_data/cln_cmb_w_map.npy")
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    hp.gnomview(cln_cmb_map, rot=[0, 90, 0], title="cln cmb map")

    n_res_map = np.load("./test_data/n_res_w_map.npy")
    cl_nres_map = hp.anafast(n_res_map, lmax=lmax)
    hp.gnomview(n_res_map, rot=[0, 90, 0], title="noise res map")
    plt.show()

    plt.loglog(l * (l + 1) * cl_cmb / (2 * np.pi), label="input cmb")
    plt.loglog(
        l * (l + 1) * cl_cln_map / (2 * np.pi), label="after nilc map weight map"
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


def check_ilc_bias():
    # check ilc bias, see paper arxiv: http://arxiv.org/abs/0807.0773

    l = np.arange(lmax + 1)

    cmb = np.load("./test_data/sim_c.npy")
    cl_cmb = hp.anafast(cmb, lmax=lmax)

    cln_cmb_map = np.load("./test_data/cln_cmb_w_map.npy")
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    hp.mollview(cln_cmb_map, rot=[0, 90, 0], title="cln cmb map")

    cmb_res_map = np.load("./test_data/c_res_w_map.npy")
    cl_cmb_map = hp.anafast(cmb_res_map, lmax=lmax)
    hp.mollview(cmb_res_map, rot=[0, 90, 0], title="cmb res map")

    n_res_map = np.load("./test_data/n_res_w_map.npy")
    cl_nres_map = hp.anafast(n_res_map, lmax=lmax)
    hp.mollview(n_res_map, rot=[0, 90, 0], title="noise res map")
    plt.show()

    cl_ilc_bias = 2 * hp.anafast(cmb_res_map, n_res_map, lmax=lmax)

    plt.loglog(l * (l + 1) * cl_cmb / (2 * np.pi), label="input cmb")

    plt.loglog(
        l * (l + 1) * cl_cln_map / (2 * np.pi), label="after nilc map weight map"
    )
    plt.loglog(l * (l + 1) * cl_cmb_map / (2 * np.pi), label="cmb residual weight map")
    plt.loglog(
        l * (l + 1) * cl_nres_map / (2 * np.pi), label="noise residual weight map"
    )
    plt.loglog(l * (l + 1) * np.abs(cl_ilc_bias) / (2 * np.pi), label="ilc bias")
    plt.loglog(np.abs(cl_ilc_bias) / cl_cmb, label="fractional ilc bias")

    plt.xlabel("$\\ell$")
    plt.ylabel("$D_\\ell [\\mu K^2]$")

    plt.legend()
    plt.show()


def main():
    gen_sim()

    test_nilc_w_map()
    get_cmb_res_w_map()
    get_n_res_w_map()

    check_res()
    check_ilc_bias()


main()
