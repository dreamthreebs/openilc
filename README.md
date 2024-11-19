# OPENILC
needlets internal combination for cmb data analysis

## STRUCTURE AND USAGE
* nilc.py is the source module to do nilc
* ./needlets is the cosine needlets configuration place, and one raw's lmin and lpeak should be the same as previous raw's lpeak and lmax! The last lmax should be the same as your lmax in NILC's lmax. Remember to set your nside to be bigger than lmax/2, because different needlets bin can have different resolution and lmax are well calculated when lmax<2*nside.
* you should see test_nilc, it is a tutorial(though not complete now).

## KNOWN ISSUE
* large memory usage in NILC
* slow rate when calculating beta covariance matrix in NILC

## FUTURE PLAN
* use pytest
* add other ILC

## HIGH PERFORMANCE COMPUTATION
SHT can be replace by other package or use healpy build from sources:
* see https://github.com/healpy/healpy/blob/main/INSTALL.rst#generating-native-binaries first
* download `cfitsio` version 4.5.0, then run ./configure --disable-curl;make;make install
* download `healpix` version 3.8.3, configure with your `cfitsio` fitsio.h and cfitsio lib to build healpix cxx
* add PKG_CONFIG_PATH for `cfitsio` and `healpix`
* pip install --no-binary healpy healpy
