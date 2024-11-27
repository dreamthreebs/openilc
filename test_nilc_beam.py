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
lmax = 1000

def expand_alm_with_hp(alm, lmax_current, lmax_target):
    """
    Expands a healpy alm array from lmax_current to lmax_target by padding with zeros.

    Args:
    alm (ndarray): Input alm array with current lmax.
    lmax_current (int): The current lmax of the alm array.
    lmax_target (int): The target lmax to expand to.

    Returns:
    ndarray: The expanded alm array with lmax_target.
    """
    # Ensure the target lmax is not smaller than the current lmax
    if lmax_target <= lmax_current:
        return alm
    else:
        # Compute the total number of alm for the target lmax
        alm_target_size = hp.Alm.getsize(lmax_target)

        # Initialize the expanded alm array with zeros
        alm_expanded = np.zeros(alm_target_size, dtype=alm.dtype)

        # Fill the expanded array with existing alm values

        for l in range(lmax_current + 1):
            for m in range(l + 1): # m ranges from 0 to l
                idx_current = hp.Alm.getidx(lmax_current, l, m)
                idx_target = hp.Alm.getidx(lmax_target, l, m)
                alm_expanded[idx_target] = alm[idx_current]
                # print(f'{l=}, {m=}, {idx_current=}, {idx_target=}')
        return alm_expanded

def gen_sim():
    # generate cmb + foreground simulation, cmb is T mode and no beam on different frequency
    sim_list = []
    cmb_list = []
    fg_list = []
    noise_list = []
    df = pd.read_csv('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/FreqBand')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T[0]
    print(f'{cl_cmb.shape=}')

    l = np.arange(lmax+1)

    bl_floor = 1e-4
    for freq_idx in range(len(df)):
        freq = df.at[freq_idx, 'freq']
        beam = df.at[freq_idx, 'beam']
        print(f'{freq=}, {beam=}')
        bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

        lmax_alm = np.argmax(bl < bl_floor) if np.any(bl < bl_floor) else lmax
        print(f"The index where the value is smaller than {bl_floor} is: {lmax_alm}")

        np.random.seed(seed=0)
        cmb_with_beam = hp.synfast(cls=cl_cmb, nside=nside, fwhm=np.deg2rad(beam)/60)

        cmb = hp.alm2map(hp.almxfl(hp.map2alm(cmb_with_beam, lmax=lmax_alm), fl=1/bl), nside=nside)
        # cls_cmb = hp.anafast(cmb, lmax=lmax)

        # plt.loglog(l, l*(l+1)*cls_cmb/(2*np.pi), label=f'{beam=}')
        # plt.legend()
        # plt.show()

        fg = np.load(f'/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/FG/{freq}.npy')[0]
        nstd = np.load(f'/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/NSTDNORTH/{freq}.npy')[0]
        noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))

        sim = cmb + fg + noise
        sim_list.append(sim)
        cmb_list.append(cmb)
        fg_list.append(fg)
        noise_list.append(noise)


    Path('./test_data').mkdir(exist_ok=True, parents=True)
    np.save('./test_data/sim_cfn.npy', sim_list)
    np.save('./test_data/sim_c.npy', cmb_list)
    np.save('./test_data/sim_f.npy', fg_list)
    np.save('./test_data/sim_n.npy', noise_list)

def test_nilc_w_alm():
    # do nilc and save the weights in alm
    time0 = time.time()
    sim = np.load('./test_data/sim_cfn.npy')
    obj_nilc = NILC(needlet_config='./needlets/beam_version.csv', weights_name='./nilc_weight/w_alm.npz', Sm_maps=sim, mask=None, lmax=lmax, nside=nside, n_iter=1)
    clean_map = obj_nilc.run_nilc()
    np.save('./test_data/cln_cmb_w_alm.npy', clean_map)
    print(f'{time.time()-time0=}')

def get_fg_res_w_alm():
    # this function tells you how to debias other component by using the weights in alm
    fg = np.load('./test_data/sim_f.npy')
    obj_nilc = NILC(needlet_config='./needlets/beam_version.csv', weights_config='./nilc_weight/w_alm.npz', Sm_maps=fg, mask=None, lmax=lmax, nside=nside, n_iter=1)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data/fg_res_w_alm.npy', fg_res)

def get_n_res_w_alm():
    noise = np.load('./test_data/sim_n.npy')
    obj_nilc = NILC(needlet_config='./needlets/beam_version.csv', weights_config='./nilc_weight/w_alm.npz', Sm_maps=noise, mask=None, lmax=lmax, nside=nside, n_iter=1)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data/n_res_w_alm.npy', fg_res)

def test_nilc_w_map():
    # do nilc and save the weights in map
    time0 = time.time()
    sim = np.load('./test_data/sim_cfn.npy')
    obj_nilc = NILC(needlet_config='./needlets/beam_version.csv', weights_name='./nilc_weight/w_map.npz', Sm_maps=sim, mask=None, lmax=lmax, nside=nside, n_iter=1, weight_in_alm=False)
    clean_map = obj_nilc.run_nilc()
    np.save('./test_data/cln_cmb_w_map.npy', clean_map)
    print(f'{time.time()-time0=}')

def get_fg_res_w_map():
    # this function tells you how to debias other component by using the weights in map
    fg = np.load('./test_data/sim_f.npy')
    obj_nilc = NILC(needlet_config='./needlets/beam_version.csv', weights_config='./nilc_weight/w_map.npz', Sm_maps=fg, mask=None, lmax=lmax, nside=nside, n_iter=1, weight_in_alm=False)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data/fg_res_w_map.npy', fg_res)

def get_n_res_w_map():
    noise = np.load('./test_data/sim_n.npy')
    obj_nilc = NILC(needlet_config='./needlets/beam_version.csv', weights_config='./nilc_weight/w_map.npz', Sm_maps=noise, mask=None, lmax=lmax, nside=nside, n_iter=1, weight_in_alm=False)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data/n_res_w_map.npy', fg_res)



def check_res():
    # check result compared with input and the foreground residual
    l = np.arange(lmax+1)

    cmb = np.load('./test_data/sim_c.npy')
    cl_cmb = hp.anafast(cmb[6], lmax=lmax)

    cln_cmb = np.load('./test_data/cln_cmb_w_alm.npy')
    cl_cln = hp.anafast(cln_cmb, lmax=lmax)

    fg_res = np.load('./test_data/fg_res_w_alm.npy')
    cl_fgres = hp.anafast(fg_res, lmax=lmax)

    n_res = np.load('./test_data/n_res_w_alm.npy')
    cl_nres = hp.anafast(n_res, lmax=lmax)

    cln_cmb_map = np.load('./test_data/cln_cmb_w_map.npy')
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    hp.gnomview(cln_cmb_map, rot=[0,90,0], title='cln cmb map')

    fg_res_map = np.load('./test_data/fg_res_w_map.npy')
    cl_fgres_map = hp.anafast(fg_res_map, lmax=lmax)
    hp.gnomview(fg_res_map, rot=[0,90,0], title='fg res map')

    n_res_map = np.load('./test_data/n_res_w_map.npy')
    cl_nres_map = hp.anafast(n_res_map, lmax=lmax)
    hp.gnomview(n_res_map, rot=[0,90,0], title='noise res map')
    plt.show()

    plt.loglog(l*(l+1)*cl_cmb/(2*np.pi), label='input cmb')
    plt.loglog(l*(l+1)*cl_cln/(2*np.pi), label='after nilc weight alm')
    plt.loglog(l*(l+1)*cl_fgres/(2*np.pi), label='foreground residual weight alm')
    plt.loglog(l*(l+1)*cl_nres/(2*np.pi), label='noise residual weight alm')

    plt.loglog(l*(l+1)*cl_cln_map/(2*np.pi), label='after nilc map weight map')
    plt.loglog(l*(l+1)*cl_fgres_map/(2*np.pi), label='foreground residual weight map')
    plt.loglog(l*(l+1)*cl_nres_map/(2*np.pi), label='noise residual weight map')

    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell [\\mu K^2]$')

    plt.legend()
    plt.show()

def main():

    # gen_sim()
    # test_nilc_w_alm()
    # get_fg_res_w_alm()
    # get_n_res_w_alm()
    # test_nilc_w_map()
    # get_fg_res_w_map()
    # get_n_res_w_map()

    check_res()

main()
