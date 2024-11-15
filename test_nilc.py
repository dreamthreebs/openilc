import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import time

from pathlib import Path
from nilc import NILC
from memory_profiler import profile

nside = 512
lmax = 500

def gen_sim():
    # generate cmb + foreground simulation, cmb is T mode and no beam on different frequency
    sim_list = []
    fg_list = []
    df = pd.read_csv('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/FreqBand')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T[0]
    print(f'{cl_cmb.shape=}')

    np.random.seed(seed=0)
    cmb = hp.synfast(cls=cl_cmb, nside=nside)

    for freq_idx in range(len(df)):
        freq = df.at[freq_idx, 'freq']
        beam = df.at[freq_idx, 'beam']
        print(f'{freq=}, {beam=}')

        fg = np.load(f'/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/FGSim/FG/{freq}.npy')[0]

        sim = cmb + fg
        sim_list.append(sim)
        fg_list.append(fg)

    Path('./test_data').mkdir(exist_ok=True, parents=True)
    np.save('./test_data/sim_cf.npy', sim_list)
    np.save('./test_data/sim_c.npy', cmb)
    np.save('./test_data/sim_f.npy', fg_list)

def test_nilc():
    # do nilc and save the weights
    time0 = time.time()
    sim = np.load('./test_data/sim_cf.npy')
    obj_nilc = NILC(weights_name='./nilc_weight/test.npz', Sm_maps=sim, mask=None, lmax=lmax, nside=nside, n_iter=1)
    clean_map = obj_nilc.run_nilc()
    np.save('./test_data/cln_cmb.npy', clean_map)
    print(f'{time.time()-time0=}')

def get_fg_res():
    # this function tells you how to debias other component by using the weights
    fg = np.load('./test_data/sim_f.npy')
    obj_nilc = NILC(weights_config='./nilc_weight/test.npz', Sm_maps=fg, mask=None, lmax=lmax, nside=nside, n_iter=1)
    fg_res = obj_nilc.run_nilc()
    np.save('./test_data/fg_res.npy', fg_res)

def check_res():
    # check result compared with input and the foreground residual
    l = np.arange(lmax+1)

    cmb = np.load('./test_data/sim_c.npy')
    cl_cmb = hp.anafast(cmb, lmax=lmax)

    cln_cmb = np.load('./test_data/cln_cmb.npy')
    cl_cln = hp.anafast(cln_cmb, lmax=lmax)

    fg_res = np.load('./test_data/fg_res.npy')
    cl_fgres = hp.anafast(fg_res, lmax=lmax)

    plt.loglog(l*(l+1)*cl_cmb/(2*np.pi), label='input cmb')
    plt.loglog(l*(l+1)*cl_cln/(2*np.pi), label='after nilc')
    plt.loglog(l*(l+1)*cl_fgres/(2*np.pi), label='foreground residual')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell [\mu K^2]$')

    plt.legend()
    plt.show()

def main():
    # gen_sim()
    test_nilc()
    # get_fg_res()
    # check_res()

main()
