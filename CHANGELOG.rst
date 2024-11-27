Unreleased

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
