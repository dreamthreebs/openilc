# OPENILC
Internal linear combination for cmb data analysis

## **NILC (Needlets Internal Linear Combination)**

This repository provides tools for performing Needlets Internal Linear Combination (NILC). Below is an overview of the key components and guidelines for using them effectively.

### **Components**

#### **1. Source Code**
- **`nilc.py`**: The main module to perform NILC.

#### **2. Configuration Files**
- **`./needlets`**:  
  - Defines the cosine needlets configuration.  
  - **Important Rules**:  
    - Each row's `lmin` and `lpeak` must match the previous row's `lpeak` and `lmax`.
    - The last `lmax` should match `lmax` in your NILC settings.  
  - **Resolution Note**:  
    Ensure your `nside` is greater than `lmax / 2` because different needlet bins may have varying resolutions. The calculations for `lmax` are valid only when `lmax < 2 * nside`.

- **`./bandinfo.csv`**:  
  - Experimental configuration file that includes parameters like frequency and beam.  
  - Ensure `lmax_alm` is **greater than** your NILC `lmax`.  
  - **Usage**: This file is primarily for testing the NILC effect **without beam corrections**.

- **`./bandinfo_beam.csv`**:  
  - Experimental configuration file for cases considering beam corrections.  
  - **Important Rules**:  
    - Set `lmax_alm` carefully to avoid numerical errors at large `ell` values (e.g., for `beam=60`, `lmax_alm ~ 500`).  
    - Ensure to use the bigger band-limited lmax (when you do deconvolve and convolve your map) than `lmax_alm`

#### **3. Tutorials**
- **`test_nilc.py`**:  
  A basic (though currently incomplete) tutorial on using NILC.  

- **`test_nilc_beam.py`**:  
  A tutorial demonstrating how to perform NILC with beam considerations.

### **Usage Guidelines**
1. **Set up `./needlets` configuration**:  
   Ensure consistency between `lmin`, `lpeak`, and `lmax` across rows and align with your NILC parameters.

2. **Verify `nside`**:  
   It should be at least `lmax / 2` for accurate results.

3. **Check beam parameters**:  
   When working with `./bandinfo_beam.csv`, account for potential numerical errors at high `ell`.

4. **Run Tutorials**:  
   Use the `test_nilc.py` and `test_nilc_beam.py` scripts to understand the process and verify your setup.


# KNOWN ISSUE
* large memory usage in NILC
* slow rate when calculating beta covariance matrix in NILC

# FUTURE PLAN
* use pytest
* add other ILC

# HIGH PERFORMANCE COMPUTATION
SHT can be replace by other package or use healpy build from sources:
* see https://github.com/healpy/healpy/blob/main/INSTALL.rst#generating-native-binaries first
* download `cfitsio` version 4.5.0, then run ./configure --disable-curl;make;make install
* download `healpix` version 3.8.3, configure with your `cfitsio` fitsio.h and cfitsio lib to build healpix cxx
* add PKG_CONFIG_PATH for `cfitsio` and `healpix`
* pip install --no-binary healpy healpy
