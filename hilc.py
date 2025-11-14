import healpy as hp
import numpy as np


class HILC:
    def __init__(self, deltal, alms, lmax, freqs, fsky, weight=None):
        self.deltal = deltal
        # self.spectra = spectra
        self.alms = alms
        self.ellmax = lmax
        self.ells = np.arange(lmax + 1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.ellmax + 1))
        self.epsilon = 1e-10
        self.w = weight

    def get_Cl(self):
        # Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = (
                    hp.alm2cl(self.alms[i], self.alms[j], lmax=self.ellmax) / self.fsky
                )
                for l in range(self.ellmax + 1):
                    self.Clij[i][j][l] = pre_Cl[l]

        return self.Clij

    def get_Rlij(self):
        self.get_Cl()
        # Construct Rlij covariance matrix
        prefactor = (2 * self.ells + 1) / (4 * np.pi)
        Rlij_no_binning = np.einsum("l,ijl->ijl", prefactor, self.Clij)
        self.Rlij = np.zeros((len(self.freqs), len(self.freqs), self.ellmax + 1))
        # deltal_array = np.abs(self.freqs-1-self.Ndeproj)/2/self.Rtol/self.fsky/(self.ells+1)
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = (
                    np.convolve(Rlij_no_binning[i][j], np.ones(2 * self.deltal + 1))
                )[self.deltal : self.ellmax + 1 + self.deltal]
        return self.Rlij

    def get_Rlij_inv(self):
        # Get inverse of R_{\ell}^{ij '}
        # self.Rlij_inv = np.array([np.linalg.pinv(self.Rlij[:,:,l]) for l in range(self.ellmax+1)])
        self.Rlij_inv = np.array(
            [
                np.linalg.inv(
                    self.Rlij[:, :, l]
                    + self.epsilon * np.eye(self.Rlij[:, :, l].shape[0])
                )
                for l in range(self.ellmax + 1)
            ]
        )
        return self.Rlij_inv  # index as Rlij_inv[l][i][j]

    def get_ab(self):
        # get spectral response vectors
        self.a = np.ones(len(self.freqs))  # index as a[i]
        # self.b = self.spectra.tsz_spectral_response(self.freqs) #index as b[i]
        return self.a

    def weights(self):
        # get weights
        numerator = np.einsum("lij,j->il", self.Rlij_inv, self.a)
        denominator = np.einsum("lkm,k,m->l", self.Rlij_inv, self.a, self.a)
        self.w = numerator / denominator  # index as w[i][l]
        print(print("constraint:", np.dot(self.w[:, 100], self.a)))
        return self.w

    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_ab()
        if self.w is not None:
            weight = self.w
        else:
            self.weights()
        weight = (
            self.w / np.sum(self.w, axis=0)
        )  # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        alm_clean = np.zeros(len(self.alms[0]), dtype=complex)
        # for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            alm_clean += hp.almxfl(
                self.alms[i],
                weight[i],
            )

        return alm_clean, weight
