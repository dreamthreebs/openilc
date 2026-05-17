import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from openilc import NILC


"""
Compare NILC spherical-harmonic-transform backends.

The comparison has two separate baselines:

- healpy with n_iter=0 vs ducc0, which uses the non-iterative adjoint transform.
- healpy with n_iter=N vs ducc0_pseudo, which passes N to
  ducc0.sht.pseudo_analysis(maxiter=N).

The pseudo backend is a different iterative solver from healpy's iter parameter,
so the tutorial reports map-level differences rather than assuming equivalence.
"""

R_TOL = 1 / 100
FREQS = np.asarray([40, 95, 155, 215, 270])


def make_needlets(lmax, nside):
    edges = np.linspace(0, lmax, 7, dtype=int)
    rows = []
    for index in range(6):
        lmin = 0 if index == 0 else edges[index - 1]
        lpeak = edges[index]
        row_lmax = edges[index + 1]
        beta_nside = min(nside, max(16, 2 ** int(np.ceil(np.log2(row_lmax / 2 + 1)))))
        rows.append(
            {
                "index": index + 1,
                "lmin": int(lmin),
                "lpeak": int(lpeak),
                "lmax": int(row_lmax),
                "nside": int(beta_nside),
            }
        )
    rows[0]["lpeak"] = 0
    rows[-1]["lmax"] = int(lmax)
    rows[-1]["nside"] = int(nside)
    return rows


def make_bands(lmax):
    return [
        {"band": index + 1, "freq": int(freq), "beam": 0, "lmax_alm": int(lmax)}
        for index, freq in enumerate(FREQS)
    ]


def make_simulated_maps(nside, lmax, seed=0):
    rng = np.random.default_rng(seed)
    ell = np.arange(lmax + 1)
    cl_cmb = np.zeros(lmax + 1)
    cl_cmb[2:] = 1e3 / (ell[2:] * (ell[2:] + 1))
    cmb = hp.synfast(cl_cmb, nside=nside, lmax=lmax, new=True)

    fg_alm = rng.normal(size=hp.Alm.getsize(lmax)) + 1j * rng.normal(
        size=hp.Alm.getsize(lmax)
    )
    fg_alm = hp.almxfl(fg_alm, 1 / np.maximum(ell + 1, 1) ** 1.2)
    fg_template = hp.alm2map(fg_alm, nside=nside, lmax=lmax)
    fg_template = fg_template / np.std(fg_template)

    maps = []
    for freq in FREQS:
        fg_scale = 20 * (freq / 95) ** -2.8 + 10 * (freq / 155) ** 1.6
        noise = rng.normal(scale=0.5 + 40 / freq, size=hp.nside2npix(nside))
        maps.append(cmb + fg_scale * fg_template + noise)
    return np.asarray(maps)


def run_nilc(maps, nside, lmax, backend, n_iter, nthreads, output_dir, verbose):
    weights_name = output_dir / f"weights_{backend}_iter{n_iter}_nside{nside}.npz"
    if weights_name.exists():
        weights_name.unlink()

    start = time.perf_counter()
    if verbose:
        nilc = NILC(
            bandinfo=make_bands(lmax),
            needlet_config=make_needlets(lmax, nside),
            weights_name=weights_name,
            Sm_maps=maps,
            lmax=lmax,
            nside=nside,
            n_iter=n_iter,
            Rtol=R_TOL,
            weight_in_alm=False,
            sht_backend=backend,
            sht_nthreads=nthreads,
        )
        clean_map = nilc.run_nilc()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            nilc = NILC(
                bandinfo=make_bands(lmax),
                needlet_config=make_needlets(lmax, nside),
                weights_name=weights_name,
                Sm_maps=maps,
                lmax=lmax,
                nside=nside,
                n_iter=n_iter,
                Rtol=R_TOL,
                weight_in_alm=False,
                sht_backend=backend,
                sht_nthreads=nthreads,
            )
            clean_map = nilc.run_nilc()
    seconds = time.perf_counter() - start
    return clean_map, seconds


def benchmark(nsides, nthreads, pseudo_iter, verbose):
    output_dir = ROOT / "tutorial_outputs" / "sht_backend"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for nside in nsides:
        lmax = min(3 * nside - 1, 500)
        print(f"\nBenchmarking nside={nside}, lmax={lmax}")
        maps = make_simulated_maps(nside, lmax, seed=nside)

        healpy0_map, healpy0_seconds = run_nilc(
            maps, nside, lmax, "healpy", 0, nthreads, output_dir, verbose
        )
        ducc0_map, ducc0_seconds = run_nilc(
            maps, nside, lmax, "ducc0", 0, nthreads, output_dir, verbose
        )
        healpy_iter_map, healpy_iter_seconds = run_nilc(
            maps, nside, lmax, "healpy", pseudo_iter, nthreads, output_dir, verbose
        )
        pseudo_map, pseudo_seconds = run_nilc(
            maps,
            nside,
            lmax,
            "ducc0_pseudo",
            pseudo_iter,
            nthreads,
            output_dir,
            verbose,
        )

        diff = ducc0_map - healpy0_map
        pseudo_diff = pseudo_map - healpy_iter_map
        rms_healpy = np.sqrt(np.mean(healpy0_map**2))
        rms_healpy_iter = np.sqrt(np.mean(healpy_iter_map**2))
        rms_diff = np.sqrt(np.mean(diff**2))
        pseudo_rms_diff = np.sqrt(np.mean(pseudo_diff**2))
        row = {
            "nside": nside,
            "lmax": lmax,
            "healpy_seconds": healpy0_seconds,
            "healpy_iter_seconds": healpy_iter_seconds,
            "ducc0_seconds": ducc0_seconds,
            "ducc0_pseudo_seconds": pseudo_seconds,
            "speedup": healpy0_seconds / ducc0_seconds,
            "pseudo_speedup": healpy_iter_seconds / pseudo_seconds,
            "rms_healpy": rms_healpy,
            "rms_healpy_iter": rms_healpy_iter,
            "rms_diff": rms_diff,
            "relative_rms_diff": rms_diff / rms_healpy,
            "pseudo_relative_rms_diff": pseudo_rms_diff / rms_healpy_iter,
            "max_abs_diff": np.max(np.abs(diff)),
            "pseudo_max_abs_diff": np.max(np.abs(pseudo_diff)),
        }
        rows.append(row)
        print(
            "  healpy_iter0={healpy_seconds:.3f}s, ducc0={ducc0_seconds:.3f}s, "
            "healpy_iterN={healpy_iter_seconds:.3f}s, "
            "ducc0_pseudo={ducc0_pseudo_seconds:.3f}s".format(**row)
        )
        print(
            "  ducc0 speedup={speedup:.2f}x, relative RMS diff={relative_rms_diff:.3e}".format(
                **row
            )
        )
        print(
            "  pseudo speedup={pseudo_speedup:.2f}x vs healpy_iterN, "
            "relative RMS diff={pseudo_relative_rms_diff:.3e}".format(
                **row
            )
        )

    return rows, output_dir


def plot_results(rows, output_dir):
    nsides = [row["nside"] for row in rows]
    healpy_times = [row["healpy_seconds"] for row in rows]
    healpy_iter_times = [row["healpy_iter_seconds"] for row in rows]
    ducc0_times = [row["ducc0_seconds"] for row in rows]
    pseudo_times = [row["ducc0_pseudo_seconds"] for row in rows]
    rel_rms = [row["relative_rms_diff"] for row in rows]
    pseudo_rel_rms = [row["pseudo_relative_rms_diff"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(nsides, healpy_times, "o-", label="healpy")
    axes[0].plot(nsides, ducc0_times, "o-", label="ducc0")
    axes[0].plot(nsides, healpy_iter_times, "o-", label="healpy_iterN")
    axes[0].plot(nsides, pseudo_times, "o-", label="ducc0_pseudo")
    axes[0].set_xlabel("nside")
    axes[0].set_ylabel("NILC runtime [s]")
    axes[0].set_title("Runtime")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(nsides, rel_rms, "o-", color="tab:red", label="ducc0 vs healpy iter0")
    axes[1].semilogy(
        nsides,
        pseudo_rel_rms,
        "o-",
        color="tab:purple",
        label="ducc0_pseudo vs healpy iterN",
    )
    axes[1].set_xlabel("nside")
    axes[1].set_ylabel("relative map RMS difference")
    axes[1].set_title("Map-level difference")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "sht_backend_benchmark.png", dpi=160)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare NILC with healpy, ducc0 adjoint, and ducc0 pseudo-analysis "
            "SHT backends."
        )
    )
    parser.add_argument(
        "--nsides",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="HEALPix nsides to benchmark. Add 512 for a larger run.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=0,
        help="ducc0 SHT threads. The ducc0 default 0 uses all hardware threads.",
    )
    parser.add_argument(
        "--pseudo-iter",
        type=int,
        default=3,
        help="maxiter for the ducc0_pseudo pseudo_analysis backend.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows, output_dir = benchmark(args.nsides, args.nthreads, args.pseudo_iter, args.verbose)
    plot_results(rows, output_dir)

    print(f"\nFigure written to {output_dir / 'sht_backend_benchmark.png'}")
    print("ducc0 is compared with healpy n_iter=0.")
    print("ducc0_pseudo is compared with healpy n_iter=--pseudo-iter.")
    print("ducc0_pseudo maps n_iter to ducc0.sht.pseudo_analysis(maxiter).")


if __name__ == "__main__":
    main()
