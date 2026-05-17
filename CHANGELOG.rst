Unreleased
----------

Release 1.6, 17th May 2026:
* package core algorithms under ``openilc`` and use ``from openilc import NILC, HILC``
* add editable-install packaging with ``pyproject.toml``
* move CSV configuration files into ``configs/`` with clearer names for bands and needlets
* add ``NILC.from_csv(...)`` as the recommended configuration entry point
* move tutorials and simulation helpers into ``tutorials/``
* generate tutorial CMB and foreground inputs at runtime with CAMB and PySM3 instead of storing ``data/`` in the repository
* remove tracked ``data/`` files and purge ``data/`` from local git history
* add English and Chinese README usage documentation
* add lightweight package/config tests under ``tests/``
* add optional ``ducc0`` and ``ducc0_pseudo`` spherical harmonic transform backends for NILC

Release 1.5, 13th January 2025:

* now your input map could not be the same as `lmax_alm` in `bandinfo.csv`

Release 1.4, 27th November 2024:

* remove redundant code when calculating beta
* added `bandinfo.csv` and `bandinfo` parameter in `nilc.py`. Now NILC calculate to your largest lmax of your bands, not the lowest (lmax represent where numerical error occur)
* add test for situation with beam

Release 1.3, 16th November 2024:

* support save weight in alm

Release 1.2, 15th November 2024:

* NILC code refactor: Now calculates an ILC map at each NILC scale first, rather than computing beta and weights for all scales upfront. This release improves memory usage control.

Release 1.1, 14th November 2024:

* add n_iter parameter in NILC algorithm, can fast the calculation on `hp.map2alm`
* reduce some memory usage in NILC

Release 1.0, 30th October 2024:

* needlets internal linear combination
