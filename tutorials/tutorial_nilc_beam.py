import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from openilc import NILC
from tutorials.sim_data import estimate_lmax_from_beam, get_band_table, get_cmb_cls, get_foreground
# from memory_profiler import profile

nside = 512
npix = hp.nside2npix(nside)
lmax = 1000

def gen_sim():
    # generate cmb + foreground + noise simulation, cmb is T mode and no beam on different frequency
    sim_list = []
    cmb_list = []
    fg_list = []
    noise_list = []
    bands = get_band_table(beam_version=True)
    cl_cmb = get_cmb_cls(lmax=8000).T[0]
    print(f'{cl_cmb.shape=}')

    l = np.arange(lmax+1)

    for band in bands:
        freq = band['freq']
        beam = band['beam']
        config_lmax_alm = band['lmax_alm']
        print(f'{freq=}, {beam=}')
        bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

        bl_floor = 1e-4
        floor_lmax_alm = estimate_lmax_from_beam(beam, lmax, bl_floor=bl_floor)
        lmax_alm = min(config_lmax_alm, lmax)
        deconv_filter = 1 / bl[: lmax_alm + 1]
        print(f'{bl_floor=}, {floor_lmax_alm=}, {config_lmax_alm=}, using {lmax_alm=}')

        np.random.seed(seed=0)
        cmb = hp.synfast(cls=cl_cmb, nside=nside, fwhm=np.deg2rad(beam)/60)

        # cls_cmb = hp.anafast(cmb, lmax=lmax)

        # plt.loglog(l, l*(l+1)*cls_cmb/(2*np.pi), label=f'{beam=}')
        # plt.legend()
        # plt.show()

        fg = get_foreground(freq, nside)
        nstd = band['nstd']
        noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))

        sim_with_beam = cmb + fg + noise

        sim = hp.alm2map(hp.almxfl(hp.map2alm(sim_with_beam, lmax=lmax_alm), fl=deconv_filter), nside=nside)
        cmb = hp.alm2map(hp.almxfl(hp.map2alm(cmb, lmax=lmax_alm), fl=deconv_filter), nside=nside)
        fg = hp.alm2map(hp.almxfl(hp.map2alm(fg, lmax=lmax_alm), fl=deconv_filter), nside=nside)
        noise = hp.alm2map(hp.almxfl(hp.map2alm(noise, lmax=lmax_alm), fl=deconv_filter), nside=nside)

        sim_list.append(sim)
        cmb_list.append(cmb)
        fg_list.append(fg)
        noise_list.append(noise)


    Path('./test_data_beam').mkdir(exist_ok=True, parents=True)
    np.save('./test_data_beam/sim_cfn.npy', sim_list)
    np.save('./test_data_beam/sim_c.npy', cmb_list)
    np.save('./test_data_beam/sim_f.npy', fg_list)
    np.save('./test_data_beam/sim_n.npy', noise_list)

def run_nilc_w_alm():
    # do nilc and save the weights in alm
    time0 = time.time()
    sim = np.load('./test_data_beam/sim_cfn.npy')
    obj_nilc = NILC.from_config("configs/beam.yaml", weights_name='./nilc_weight/w_alm.npz', Sm_maps=sim, mask=None, lmax=lmax, nside=nside, n_iter=1)
    clean_map = obj_nilc.run_nilc()
    np.save('./test_data_beam/cln_cmb_w_alm.npy', clean_map)
    print(f'{time.time()-time0=}')

def get_fg_res_w_alm():
    # this function tells you how to debias other component by using the weights in alm
    fg = np.load('./test_data_beam/sim_f.npy')
    obj_nilc = NILC.from_config("configs/beam.yaml", weights_config='./nilc_weight/w_alm.npz', Sm_maps=fg, mask=None, lmax=lmax, nside=nside, n_iter=1)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data_beam/fg_res_w_alm.npy', fg_res)

def get_n_res_w_alm():
    # calc noise bias term
    noise = np.load('./test_data_beam/sim_n.npy')
    obj_nilc = NILC.from_config("configs/beam.yaml", weights_config='./nilc_weight/w_alm.npz', Sm_maps=noise, mask=None, lmax=lmax, nside=nside, n_iter=1)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data_beam/n_res_w_alm.npy', fg_res)

def run_nilc_w_map():
    # do nilc and save the weights in map
    time0 = time.time()
    sim = np.load('./test_data_beam/sim_cfn.npy')
    obj_nilc = NILC.from_config("configs/beam.yaml", weights_name='./nilc_weight/w_map.npz', Sm_maps=sim, mask=None, lmax=lmax, nside=nside, n_iter=1, weight_in_alm=False)
    clean_map = obj_nilc.run_nilc()
    np.save('./test_data_beam/cln_cmb_w_map.npy', clean_map)
    print(f'{time.time()-time0=}')

def get_fg_res_w_map():
    # this function tells you how to debias other component by using the weights in map
    fg = np.load('./test_data_beam/sim_f.npy')
    obj_nilc = NILC.from_config("configs/beam.yaml", weights_config='./nilc_weight/w_map.npz', Sm_maps=fg, mask=None, lmax=lmax, nside=nside, n_iter=1, weight_in_alm=False)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data_beam/fg_res_w_map.npy', fg_res)

def get_n_res_w_map():
    # noise bias term
    noise = np.load('./test_data_beam/sim_n.npy')
    obj_nilc = NILC.from_config("configs/beam.yaml", weights_config='./nilc_weight/w_map.npz', Sm_maps=noise, mask=None, lmax=lmax, nside=nside, n_iter=1, weight_in_alm=False)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data_beam/n_res_w_map.npy', fg_res)



def check_res():
    # check result compared with input and the foreground residual and noise bias
    # save weights in alm or map should not affect the final results so there lines in plots are overlapped

    l = np.arange(lmax+1)

    cmb = np.load('./test_data_beam/sim_c.npy')
    cl_cmb = hp.anafast(cmb[4], lmax=lmax)

    cln_cmb = np.load('./test_data_beam/cln_cmb_w_alm.npy')
    cl_cln = hp.anafast(cln_cmb, lmax=lmax)

    fg_res = np.load('./test_data_beam/fg_res_w_alm.npy')
    cl_fgres = hp.anafast(fg_res, lmax=lmax)

    n_res = np.load('./test_data_beam/n_res_w_alm.npy')
    cl_nres = hp.anafast(n_res, lmax=lmax)

    cln_cmb_map = np.load('./test_data_beam/cln_cmb_w_map.npy')
    cl_cln_map = hp.anafast(cln_cmb_map, lmax=lmax)
    hp.gnomview(cln_cmb_map, rot=[0,90,0], title='cln cmb map')

    fg_res_map = np.load('./test_data_beam/fg_res_w_map.npy')
    cl_fgres_map = hp.anafast(fg_res_map, lmax=lmax)
    hp.gnomview(fg_res_map, rot=[0,90,0], title='fg res map')

    n_res_map = np.load('./test_data_beam/n_res_w_map.npy')
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

    gen_sim()

    run_nilc_w_alm()
    get_fg_res_w_alm()
    get_n_res_w_alm()
    run_nilc_w_map()
    get_fg_res_w_map()
    get_n_res_w_map()

    check_res()

main()
