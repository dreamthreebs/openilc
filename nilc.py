import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import gc

from pathlib import Path

class NILC:
    def __init__(self, bandinfo='./bandinfo.csv', needlet_config='./needlets/beam_version.csv', weights_name=None, weights_config=None, Sm_alms=None, Sm_maps=None, mask=None, lmax=1000, nside=1024, Rtol=1/1000, n_iter=3, weight_in_alm=True):

        """
        Needlets internal linear combination

        input Sm_maps should be dimention: (n_freq, n_pixel) or Sm_alms with dimention:(n_freq, n_alm)

        """

        self.bandinfo = pd.read_csv(bandinfo) # load band info
        self.needlet = pd.read_csv(needlet_config) # load cosine needlets config
        self.n_needlet = len(self.needlet) # number of needlets bin
        self.weight_in_alm = weight_in_alm # save weight to alm or maps

        if weights_name is not None:
            if Path(weights_name).suffix != '.npz':
                raise ValueError('the weights should be saved as .npz file')
            self.weights_name = Path(weights_name) # where to save your weights
            self.weights_name.parent.mkdir(parents=True, exist_ok=True) # you don't need to make a new dir for weights
        else:
            self.weights_name = weights_name

        if weights_config is not None:
            self.weights_config = Path(weights_config) # where to find your weights
        else:
            self.weights_config = weights_config

        self.Rtol = Rtol # theoretical percentage of ilc bias (will change your degree of freedom when calc R covariance matrix)
        self.lmax = lmax # maximum lmax when calculating alm, should be set as the same as needlets last bin's lmax
        self.n_iter = n_iter # iteration number when calculating alm

        if (weights_config is not None) and (weights_name is not None):
            raise ValueError('weights should not be given and calculated at the same time!')

        if (Sm_maps is None) and (Sm_alms is None):
            raise ValueError('no input!')

        if Sm_maps is not None:
            self.nmaps = Sm_maps.shape[0]
            self.npix = Sm_maps.shape[-1]
            self.nside = hp.npix2nside(self.npix)
            self.maps = Sm_maps
            if mask is not None:
                self.maps = Sm_maps * mask
            self.mask = mask

            Sm_alms_list = []
            for i in range(self.nmaps):
                Sm_alm = hp.map2alm(self.maps[i], lmax=lmax, iter=self.n_iter)
                Sm_alms_list.append(Sm_alm)
            self.alms = np.asarray(Sm_alms_list)

            del self.maps, Sm_maps
            gc.collect()

        if Sm_alms is not None:
            self.alms = Sm_alms
            self.nmaps = Sm_alms.shape[0]
            self.npix = hp.nside2npix(nside)
            self.nside = nside

        print(f'{weights_config=}, {weights_name=}, {needlet_config=}')
        print(f'{Rtol=}, {lmax=}, nside={self.nside}')

    def calc_hl(self):
        hl = np.zeros((self.n_needlet, self.lmax+1))
        l_range = np.arange(self.lmax+1)
        for j in range(self.n_needlet):
            nlmax = self.needlet.at[j,'lmax']
            nlmin = self.needlet.at[j,'lmin']
            nlpeak = self.needlet.at[j,'lpeak']

            condition1 = (l_range < nlmin) | (l_range > nlmax)
            condition2 = l_range < nlpeak
            condition3 = l_range > nlpeak
            eps=1e-15
            hl[j] = np.where(condition1, 0,
                       np.where(condition2, np.cos(((nlpeak-l_range)/(nlpeak-nlmin+eps)) * np.pi/2),
                       np.where(condition3, np.cos(((l_range-nlpeak)/(nlmax-nlpeak+eps)) * np.pi/2), 1)))
        self.hl = hl

    def calc_FWHM(self):
        Neff = (self.nmaps - 1) / self.Rtol
        FWHM = np.zeros(self.n_needlet)
        for j in range(self.n_needlet):
            dof = np.sum(self.hl[j]**2 * (2*np.arange(self.lmax+1)+1))
            # dof = ((self.needlet.at[j, 'lmax']+1)**2 - self.needlet.at[j, 'lmin']**2)
            print(f'{dof = }')
            fsky = Neff / dof
            print(f'initial {fsky = }')
            if fsky > 1:
                fsky = 1
            print(f'final {fsky = }')
            dof_eff = fsky * dof
            print(f'{dof_eff = }')
            n_pix = hp.nside2npix(self.needlet.at[j, 'nside'])
            actual_pix = fsky * n_pix
            print(f'the pixel used in {j} scale is:{actual_pix}')
            pixarea = actual_pix * hp.nside2pixarea(self.needlet.at[j, 'nside']) # spherical cap area A=2*pi(1-cos(theta))
            theta = np.arccos(1 - pixarea / (2 * np.pi)) * 180 / np.pi
            FWHM[j] = np.sqrt(8 * np.log(2)) * theta
        self.FWHM = FWHM

    def calc_beta_for_scale(self, j):
        print(f'calculate beta for scale {j}...')
        hl = self.hl[j]
        beta_nside = self.needlet.at[j, 'nside']
        beta_lmax = self.needlet.at[j, 'lmax']
        beta_npix = hp.nside2npix(beta_nside)

        idx_to_remove = self.bandinfo[self.bandinfo['lmax_alm'] < beta_lmax].index
        alms = np.delete(self.alms, idx_to_remove, axis=0)
        nmaps = np.size(alms, axis=0)
        print(f'{idx_to_remove=}, {alms.shape=}, {nmaps=}')

        beta = np.zeros((nmaps, beta_npix))

        for i in range(nmaps):
            beta_alm_ori = hp.almxfl(alms[i], self.hl[j])
            beta[i] = hp.alm2map(beta_alm_ori, beta_nside)

        print(f'{beta.shape = }')

        return beta

    def calc_w_for_scale(self, j, beta):
        w_list = []
        print(f"calc_weights at number:{j}")

        nmaps = np.size(beta, axis=0)
        oneVec = np.ones(nmaps)

        R_nside = self.needlet.at[j, 'nside']
        R_lmax = self.needlet.at[j, 'lmax']
        R = np.zeros((hp.nside2npix(R_nside), nmaps, nmaps))
        for c1 in range(nmaps):
            for c2 in range(c1,nmaps):
                prodMap = beta[c1] * beta[c2]
                # hp.mollview(prodMap, norm='hist', title = f"{j = }, {c1 = }, {c2 = }")
                # plt.show()
                RMap = hp.smoothing(prodMap, np.deg2rad(self.FWHM[j]),iter=0)
                # hp.mollview(np.abs(RMap), norm='log',title = f"{c1 = }, {c2 = }")
                # plt.show()
                if c1 != c2:
                    R[:,c1,c2] = RMap
                    R[:,c2,c1] = RMap
                else:
                    # eps = 0.1 * np.min(np.abs(RMap))
                    # R[:,c1,c2] = RMap + eps # for no noise testing
                    # print(f"{eps = }")
                    # R[:,c1,c2] = RMap + np.mean(RMap) # for no noise testing
                    R[:,c1,c2] = RMap
        invR = np.linalg.inv(R)
        if self.weight_in_alm:
            w_map = (invR@oneVec).T/(oneVec@invR@oneVec + 1e-15)
            w = np.asarray([hp.map2alm(w_map[i], lmax=R_lmax) for i in range(nmaps)])
        else:
            w = (invR@oneVec).T/(oneVec@invR@oneVec + 1e-15)
        return w

    def calc_map(self):
        resMap = 0

        if self.weights_config is None:
            weight_list = []
        else:
            print('weight are given...')
            weights = np.load(self.weights_config)

        for j in range(self.n_needlet):
            print(f'begin calculation at scale {j}')
            print(f'calc beta...')
            beta = self.calc_beta_for_scale(j)

            R_nside = self.needlet.at[j, 'nside']

            if self.weight_in_alm:
                if self.weights_config is None:
                    print(f'calc weight...')
                    ilc_w_alm = self.calc_w_for_scale(j, beta)
                else:
                    ilc_w_alm = weights[f'arr_{j}']
                print(f'{ilc_w_alm.shape=}')
                nmaps = np.size(beta, axis=0)
                ilc_w = np.asarray([hp.alm2map(ilc_w_alm[i], nside=R_nside) for i in range(nmaps)])
            else:
                if self.weights_config is None:
                    print(f'calc weight...')
                    ilc_w = self.calc_w_for_scale(j, beta)
                else:
                    ilc_w = weights[f'arr_{j}']

            print(f'{ilc_w.shape=}')

            res  = np.sum(beta * ilc_w, axis=0)
            print(f'calc ilc beta...')
            res_alm = hp.map2alm(res, iter=self.n_iter)
            print(f'{res_alm.shape = }')
            res_alm = hp.almxfl(res_alm, self.hl[j])
            print(f'after resxflalm shape = {res_alm.shape}')
            ilced_Map = hp.alm2map(res_alm, self.nside)
            resMap = resMap + ilced_Map

            if self.weights_config is None:
                if self.weight_in_alm:
                    weight_list.append(ilc_w_alm)
                else:
                    weight_list.append(ilc_w)
                self.weights = weight_list
        return resMap

    def run_nilc(self):
        print('calc_hl...')
        self.calc_hl()
        print('calc_FWHM...')
        self.calc_FWHM()

        res_map = self.calc_map()

        if self.weights_config is None:
            np.savez(self.weights_name, *self.weights)

        print('Calculation completed!')

        return res_map


