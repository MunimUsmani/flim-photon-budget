"""
FLIM photon-budget simulation
-----------------------------
Question: with a fixed photon budget per pixel, how accurately can we estimate the
free-NADH fraction from a two-component fluorescence decay, and which estimator
(least squares, Poisson maximum likelihood, phasor) gets closest to the physical limit?

Model (all assumptions are explicit and simple on purpose):
  - Two species: free NADH (short lifetime) and protein-bound NADH (long lifetime).
  - Decay is periodic with the laser repetition period (80 MHz -> 12.5 ns).
  - Instrument response function (IRF) is Gaussian.
  - Photon counting noise is Poisson. No background, no pile-up, no detector jitter tails.
  - The two lifetimes are assumed KNOWN; only the fraction f is estimated.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- parameters ----------------
REP_RATE = 80e6                  # Hz
T = 1e9 / REP_RATE               # ns, repetition period (12.5)
NBINS = 256
t = (np.arange(NBINS) + 0.5) * T / NBINS
TAU_FREE, TAU_BOUND = 0.4, 2.5   # ns, typical literature range for NADH
IRF_FWHM = 0.25                  # ns
TRUE_F = 0.5                     # fraction of photons from free species
PHOTONS = [50, 100, 300, 1000, 3000, 10000]
TRIALS = 500
rng = np.random.default_rng(0)

# ---------------- forward model ----------------
def species_pdf(tau):
    """Periodic exponential decay convolved (circularly) with the Gaussian IRF, normalized."""
    decay = np.exp(-t / tau) / (1 - np.exp(-T / tau))
    dt = T / NBINS
    k = np.fft.fftfreq(NBINS, d=dt)
    sigma = IRF_FWHM / 2.355
    irf_ft = np.exp(-2 * (np.pi * k * sigma) ** 2)       # Fourier transform of Gaussian
    p = np.real(np.fft.ifft(np.fft.fft(decay) * irf_ft))
    p = np.clip(p, 1e-12, None)
    return p / p.sum()

P1, P2 = species_pdf(TAU_FREE), species_pdf(TAU_BOUND)

def simulate(f, n_photons):
    """Draw one Poisson-noisy decay histogram with n_photons expected photons."""
    mu = n_photons * (f * P1 + (1 - f) * P2)
    return rng.poisson(mu)

# ---------------- estimators ----------------
def est_lsq(y):
    """Unweighted least squares: y ~ A*P1 + B*P2, then f = A/(A+B)."""
    M = np.stack([P1, P2], axis=1)
    (A, B), *_ = np.linalg.lstsq(M, y.astype(float), rcond=None)
    s = A + B
    return np.clip(A / s, 0, 1) if s > 0 else np.nan

def est_mle(y):
    """Poisson maximum likelihood for f (total counts fixed to observed total)."""
    def nll(f):
        return -np.sum(y * np.log(f * P1 + (1 - f) * P2))
    return minimize_scalar(nll, bounds=(0, 1), method="bounded", options={"xatol": 1e-6}).x

OMEGA = 2 * np.pi / T
def phasor(h):
    h = np.asarray(h, float)
    return np.array([np.sum(h * np.cos(OMEGA * t)), np.sum(h * np.sin(OMEGA * t))]) / h.sum()

PH1, PH2 = phasor(P1), phasor(P2)   # calibrated pure-species phasors (IRF included)
def est_phasor(y):
    """Project the measured phasor onto the line between the two pure-species phasors."""
    d = PH1 - PH2
    f = np.dot(phasor(y) - PH2, d) / np.dot(d, d)
    return np.clip(f, 0, 1)

METHODS = {"Least squares": est_lsq, "Poisson MLE": est_mle, "Phasor": est_phasor}

# ---------------- theoretical limit ----------------
def cramer_rao_std(f, n):
    """Std dev of any unbiased estimator of f (Fisher information of a multinomial histogram)."""
    mu = f * P1 + (1 - f) * P2
    fisher = n * np.sum((P1 - P2) ** 2 / mu)
    return 1 / np.sqrt(fisher)

# ---------------- experiment ----------------
if __name__ == "__main__":
    results = {m: {"bias": [], "std": []} for m in METHODS}
    for n in PHOTONS:
        for name, fn in METHODS.items():
            est = np.array([fn(simulate(TRUE_F, n)) for _ in range(TRIALS)])
            est = est[~np.isnan(est)]
            results[name]["bias"].append(est.mean() - TRUE_F)
            results[name]["std"].append(est.std())
        print(f"N={n:6d} | " + " | ".join(
            f"{m}: bias {results[m]['bias'][-1]:+.3f}, std {results[m]['std'][-1]:.3f}" for m in METHODS)
            + f" | CRB {cramer_rao_std(TRUE_F, n):.3f}")

    crb = [cramer_rao_std(TRUE_F, n) for n in PHOTONS]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # (a) example decays
    for n, c in [(100, "tab:red"), (10000, "tab:blue")]:
        ax[0].semilogy(t, np.clip(simulate(TRUE_F, n), 0.5, None), ".", ms=3, color=c, label=f"{n} photons")
    ax[0].semilogy(t, 10000 * (TRUE_F * P1 + (1 - TRUE_F) * P2), "k-", lw=1, label="noise-free (10,000)")
    ax[0].set(xlabel="Time (ns)", ylabel="Counts per bin", title="(a) Simulated decays")
    ax[0].legend(fontsize=8)
    # (b) precision
    for name, mk in zip(METHODS, ["o-", "s-", "^-"]):
        ax[1].loglog(PHOTONS, results[name]["std"], mk, label=name)
    ax[1].loglog(PHOTONS, crb, "k--", label="Cramér–Rao bound")
    ax[1].set(xlabel="Photons per pixel", ylabel="Std of estimated free fraction",
              title="(b) Precision vs photon budget")
    ax[1].legend(fontsize=8)
    # (c) bias
    for name, mk in zip(METHODS, ["o-", "s-", "^-"]):
        ax[2].semilogx(PHOTONS, results[name]["bias"], mk, label=name)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="Photons per pixel", ylabel="Bias (estimate − truth)", title="(c) Bias vs photon budget")
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("flim_results.png", dpi=150)
    print("saved flim_results.png")
