from functools import lru_cache

import numpy as np
import healpy as hp


@lru_cache(maxsize=None)
def _ducc_healpix_geometry(nside):
    import ducc0

    return ducc0.healpix.Healpix_Base(nside, "RING").sht_info()


class HealpySHT:
    def __init__(self, nthreads=1):
        self.nthreads = nthreads

    def map2alm(self, m, lmax=None, n_iter=0):
        if n_iter is None:
            return hp.map2alm(m, lmax=lmax)
        return hp.map2alm(m, lmax=lmax, iter=n_iter)

    def alm2map(self, alm, nside):
        return hp.alm2map(alm, nside=nside)


class DuccAdjointSHT:
    def __init__(self, nthreads=0):
        import ducc0

        self.ducc0 = ducc0
        self.nthreads = nthreads

    def map2alm(self, m, lmax=None, n_iter=0):
        if n_iter not in (0, None):
            raise ValueError("ducc0 backend does not support iterative map2alm")
        if lmax is None:
            lmax = 3 * hp.npix2nside(len(m)) - 1
        nside = hp.npix2nside(len(m))
        alm = self.ducc0.sht.adjoint_synthesis(
            map=np.asarray(m).reshape(1, -1),
            lmax=lmax,
            spin=0,
            nthreads=self.nthreads,
            **_ducc_healpix_geometry(nside),
        ).reshape(-1)
        return alm * (4 * np.pi / len(m))

    def alm2map(self, alm, nside):
        return self.ducc0.sht.synthesis(
            alm=np.asarray(alm).reshape(1, -1),
            lmax=hp.Alm.getlmax(len(alm)),
            spin=0,
            nthreads=self.nthreads,
            **_ducc_healpix_geometry(nside),
        ).reshape(-1)


class DuccPseudoSHT(DuccAdjointSHT):
    def __init__(self, nthreads=0, epsilon=1e-10):
        super().__init__(nthreads=nthreads)
        self.epsilon = epsilon

    def map2alm(self, m, lmax=None, n_iter=3):
        if lmax is None:
            lmax = 3 * hp.npix2nside(len(m)) - 1
        maxiter = 3 if n_iter is None else n_iter
        if maxiter < 0:
            raise ValueError("ducc0_pseudo backend requires n_iter >= 0")
        nside = hp.npix2nside(len(m))
        alm, _, _, _, _ = self.ducc0.sht.pseudo_analysis(
            map=np.asarray(m).reshape(1, -1),
            lmax=lmax,
            spin=0,
            nthreads=self.nthreads,
            maxiter=maxiter,
            epsilon=self.epsilon,
            **_ducc_healpix_geometry(nside),
        )
        return alm.reshape(-1)


DuccSHT = DuccAdjointSHT


def get_sht_backend(name="healpy", nthreads=None):
    if name in (None, "healpy"):
        return HealpySHT(nthreads=1 if nthreads is None else nthreads)
    if name == "ducc0":
        return DuccAdjointSHT(nthreads=0 if nthreads is None else nthreads)
    if name == "ducc0_pseudo":
        return DuccPseudoSHT(nthreads=0 if nthreads is None else nthreads)
    raise ValueError(f"unknown SHT backend: {name}")
