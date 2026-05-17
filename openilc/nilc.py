import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import gc

from pathlib import Path
from .configs import load_csv_table, load_table
from .sht import get_sht_backend


_BAND_COLUMNS = ("lmax_alm",)
_NEEDLET_COLUMNS = ("lmin", "lpeak", "lmax", "nside")


class NILC:
    def __init__(self, bandinfo, needlet_config, weights_name=None, weights_config=None, Sm_alms=None, Sm_maps=None, mask=None, lmax=1000, nside=1024, Rtol=1/1000, n_iter=3, weight_in_alm=True, sht_backend="healpy", sht_nthreads=None):

        """
        Needlets internal linear combination

        Parameters:
            bandinfo: list[dict] or ConfigTable
            Band information, including frequency, beam, lmax_alm (the lmax when map2alm)
            needlet_config: list[dict] or ConfigTable
            Needlets configuration, including lmin, lpeak and lmax of each needlets scale
            weights_name: str
            Path where to save the weights at each needlets scale, if weight_in_alm is True, save weights in alm, else save in maps
            weights_config: str
            Path where to load the weights at each needlets scale, if weight_in_alm is True, load weights in alm, else load in maps
            Sm_alms: np.ndarray
            Input smoothed alms, all your maps should be in the same resolution to satisfy the ILC condition. The shape of this parameter should be (n_freq, n_alm)
            Sm_maps: np.ndarray
            Input smoothed maps, all your maps should be in the same resolution to satisfy the ILC condition. The shape of this parameter should be (n_freq, n_pixel)
            mask: np.ndarray
            Input mask, shape should be (n_pixel,)
            lmax: int
            Maximum ell, should be the same in the latest lmax in needlet_config
            nside: int
            The nside of output
            Rtol: float
            Theoretical percentage of ilc bias (will change your degree of freedom when calc R covariance matrix)
            n_iter: int
            Iteration number when calculating alm
            weight_in_alm: bool
            Whether to save weight in alm
            sht_backend: str
            Spherical harmonic backend: "healpy", "ducc0", or "ducc0_pseudo"
            sht_nthreads: int
            Number of threads for the SHT backend. For ducc0, None maps to 0,
            which lets ducc0 use all available hardware threads.
        """

        self.bandinfo = load_table(bandinfo) # load band info
        self.needlet = load_table(needlet_config) # load cosine needlets config
        self.n_needlet = len(self.needlet) # number of needlets bin
        self.weight_in_alm = weight_in_alm
        self.sht_backend = sht_backend
        self.sht = get_sht_backend(sht_backend, nthreads=sht_nthreads)

        if weights_name is not None:
            if Path(weights_name).suffix != '.npz':
                raise ValueError('the weights should be saved as .npz file')
            self.weights_name = Path(weights_name)
            self.weights_name.parent.mkdir(parents=True, exist_ok=True) # you don't need to make a new dir for weights
        else:
            self.weights_name = weights_name

        if weights_config is not None:
            self.weights_config = Path(weights_config)
        else:
            self.weights_config = weights_config

        self.Rtol = Rtol
        self.lmax = lmax
        self.n_iter = n_iter

        self._validate_basic_inputs(
            weights_name=weights_name,
            weights_config=weights_config,
            Sm_maps=Sm_maps,
            Sm_alms=Sm_alms,
            mask=mask,
            nside=nside,
            lmax=lmax,
            Rtol=Rtol,
        )

        if sht_backend == "ducc0" and n_iter not in (0, None):
            raise ValueError("ducc0 backend requires n_iter=0 or n_iter=None")

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
                lmax_alm = self.bandinfo.at[i, 'lmax_alm']
                if lmax_alm >= lmax:
                    Sm_alm = self.sht.map2alm(self.maps[i], lmax=lmax, n_iter=self.n_iter)
                else:
                    Sm_alm = self.sht.map2alm(self.maps[i], lmax=lmax_alm, n_iter=self.n_iter)
                    Sm_alm = hp.resize_alm(alm=Sm_alm, lmax=lmax_alm, mmax=lmax_alm, lmax_out=lmax, mmax_out=lmax)
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
        print(f'{Rtol=}, {lmax=}, nside={self.nside}, sht_backend={self.sht_backend}')

    def _validate_basic_inputs(
        self,
        *,
        weights_name,
        weights_config,
        Sm_maps,
        Sm_alms,
        mask,
        nside,
        lmax,
        Rtol,
    ):
        if weights_config is not None and weights_name is not None:
            raise ValueError("weights_config and weights_name cannot both be given")

        if Sm_maps is None and Sm_alms is None:
            raise ValueError("either Sm_maps or Sm_alms must be provided")

        if not isinstance(lmax, (int, np.integer)) or lmax < 0:
            raise ValueError("lmax must be a non-negative integer")

        if Rtol <= 0:
            raise ValueError("Rtol must be positive")

        self._validate_config_tables(lmax)

        if Sm_maps is not None:
            Sm_maps = np.asarray(Sm_maps)
            if Sm_maps.ndim != 2:
                raise ValueError("Sm_maps must have shape (n_freq, n_pixel)")
            if not hp.isnpixok(Sm_maps.shape[1]):
                raise ValueError(
                    "Sm_maps has an invalid HEALPix pixel count: "
                    f"{Sm_maps.shape[1]}"
                )
            if mask is not None and np.asarray(mask).shape != (Sm_maps.shape[1],):
                raise ValueError(
                    "mask must have shape (n_pixel,), matching Sm_maps; "
                    f"got {np.asarray(mask).shape}"
                )
            self._validate_channel_count(Sm_maps.shape[0])

        if Sm_alms is not None:
            Sm_alms = np.asarray(Sm_alms)
            if Sm_alms.ndim != 2:
                raise ValueError("Sm_alms must have shape (n_freq, n_alm)")
            if not hp.isnsideok(nside):
                raise ValueError(f"nside must be a valid HEALPix nside; got {nside}")
            alm_lmax = hp.Alm.getlmax(Sm_alms.shape[1])
            if alm_lmax < lmax:
                raise ValueError(
                    "Sm_alms does not contain enough harmonic modes for lmax="
                    f"{lmax}; inferred alm lmax is {alm_lmax}"
                )
            self._validate_channel_count(Sm_alms.shape[0])

        if Sm_maps is not None and Sm_alms is not None:
            if np.asarray(Sm_maps).shape[0] != np.asarray(Sm_alms).shape[0]:
                raise ValueError("Sm_maps and Sm_alms must have the same n_freq")

    def _validate_config_tables(self, lmax):
        if len(self.bandinfo) == 0:
            raise ValueError("bandinfo must contain at least one row")
        if len(self.needlet) == 0:
            raise ValueError("needlet_config must contain at least one row")

        self._require_columns(self.bandinfo, _BAND_COLUMNS, "bandinfo")
        self._require_columns(self.needlet, _NEEDLET_COLUMNS, "needlet_config")

        previous_lmax = None
        for j, row in enumerate(self.needlet.rows):
            n_lmin = row["lmin"]
            n_lpeak = row["lpeak"]
            n_lmax = row["lmax"]
            nside = row["nside"]

            if not all(isinstance(value, (int, np.integer)) for value in (n_lmin, n_lpeak, n_lmax, nside)):
                raise ValueError(
                    "needlet_config rows must use integer lmin, lpeak, lmax, "
                    f"and nside values; row {j} is {row}"
                )
            if n_lmin < 0 or n_lpeak < 0 or n_lmax < 0:
                raise ValueError(f"needlet_config row {j} has negative ell values")
            if not (n_lmin <= n_lpeak <= n_lmax):
                raise ValueError(
                    "needlet_config row "
                    f"{j} must satisfy lmin <= lpeak <= lmax"
                )
            if n_lmax > lmax:
                raise ValueError(
                    f"needlet_config row {j} has lmax={n_lmax}, exceeding "
                    f"NILC lmax={lmax}"
                )
            if not hp.isnsideok(nside):
                raise ValueError(
                    f"needlet_config row {j} has invalid HEALPix nside={nside}"
                )
            if previous_lmax is not None and n_lmin > previous_lmax:
                raise ValueError(
                    f"needlet_config row {j} leaves a gap after lmax="
                    f"{previous_lmax}"
                )
            previous_lmax = n_lmax

        final_lmax = self.needlet.at[len(self.needlet) - 1, "lmax"]
        if final_lmax != lmax:
            raise ValueError(
                "NILC lmax must match the last needlet lmax; "
                f"got lmax={lmax}, last needlet lmax={final_lmax}"
            )

        for i, row in enumerate(self.bandinfo.rows):
            lmax_alm = row["lmax_alm"]
            if not isinstance(lmax_alm, (int, np.integer)) or lmax_alm < 0:
                raise ValueError(
                    f"bandinfo row {i} must have a non-negative integer lmax_alm"
                )

    @staticmethod
    def _require_columns(table, columns, table_name):
        for column in columns:
            missing_rows = [
                str(i) for i, row in enumerate(table.rows) if column not in row
            ]
            if missing_rows:
                raise ValueError(
                    f"{table_name} is missing required column '{column}' "
                    f"in row(s): {', '.join(missing_rows)}"
                )

    def _validate_channel_count(self, n_freq):
        if n_freq != len(self.bandinfo):
            raise ValueError(
                "number of input channels must match bandinfo rows; "
                f"got n_freq={n_freq}, bandinfo rows={len(self.bandinfo)}"
            )

        for j, row in enumerate(self.needlet.rows):
            active_channels = sum(
                band["lmax_alm"] >= row["lmax"] for band in self.bandinfo.rows
            )
            if active_channels < 2:
                raise ValueError(
                    f"needlet scale {j} keeps only {active_channels} channel(s) "
                    f"after applying lmax_alm >= {row['lmax']}; at least 2 "
                    "channels are required"
                )

    @classmethod
    def from_csv(cls, bands, needlets, **kwargs):
        return cls(
            bandinfo=load_csv_table(bands),
            needlet_config=load_csv_table(needlets),
            **kwargs,
        )

    def calc_hl(self):
        # generate cosine filter in alm at each needlets scale
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
        # get gaussian smoothing sigma from Rtol
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

        idx_to_remove = self.bandinfo.indices_where(
            'lmax_alm', lambda value: value < beta_lmax
        )
        alms = np.delete(self.alms, idx_to_remove, axis=0)
        nmaps = np.size(alms, axis=0)
        print(f'{idx_to_remove=}, {alms.shape=}, {nmaps=}')

        beta = np.zeros((nmaps, beta_npix))

        for i in range(nmaps):
            beta_alm_ori = hp.almxfl(alms[i], self.hl[j])
            beta[i] = self.sht.alm2map(beta_alm_ori, beta_nside)

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
            w = np.asarray([self.sht.map2alm(w_map[i], lmax=R_lmax, n_iter=None) for i in range(nmaps)])
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
                ilc_w = np.asarray([self.sht.alm2map(ilc_w_alm[i], nside=R_nside) for i in range(nmaps)])
            else:
                if self.weights_config is None:
                    print(f'calc weight...')
                    ilc_w = self.calc_w_for_scale(j, beta)
                else:
                    ilc_w = weights[f'arr_{j}']

            print(f'{ilc_w.shape=}')

            res  = np.sum(beta * ilc_w, axis=0)
            print(f'calc ilc beta...')
            res_alm = self.sht.map2alm(res, n_iter=self.n_iter)
            print(f'{res_alm.shape = }')
            res_alm = hp.almxfl(res_alm, self.hl[j])
            print(f'after resxflalm shape = {res_alm.shape}')
            ilced_Map = self.sht.alm2map(res_alm, self.nside)
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

        if self.weights_config is None and self.weights_name is not None:
            np.savez(self.weights_name, *self.weights)

        print('Calculation completed!')

        return res_map
