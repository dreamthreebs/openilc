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
