Unreleased

Release 1.2, 15th November 2024:

* NILC code refactor: Now calculates an ILC map at each NILC scale first, rather than computing beta and weights for all scales upfront. This release improves memory usage control.
* the previous code are in nilc_v1_1.py for backup, you can also use it.

Release 1.1, 14th November 2024:

* add n_iter parameter in NILC algorithm, can fast the calculation on `hp.map2alm`
* reduce some memory usage in NILC

Release 1.0, 30th October 2024:

* needlets internal linear combination
