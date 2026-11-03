# FLIM photon budget: how well can we estimate NADH free fraction?

A small simulation, inspired by label-free two-photon autofluorescence lifetime microscopy (e.g. Tan et al. 2026, IEEE T-MBMC). I have no optics background; I built this to understand the central instrumentation tradeoff: **photons per pixel vs. accuracy vs. speed**.

## Question
For a two-component decay (free vs. protein-bound NADH), how precisely can the free fraction be estimated at a given photon count, and which estimator comes closest to the theoretical limit?

## Setup
- Lifetimes 0.4 ns (free) and 2.5 ns (bound), 80 MHz repetition rate (12.5 ns window), 256 time bins.
- Gaussian IRF, 0.25 ns FWHM. Poisson photon noise. True free fraction 0.5.
- Estimators: unweighted least squares, Poisson maximum likelihood, phasor unmixing.
- 500 Monte Carlo trials per photon level (50 to 10,000 photons). Compared against the Cramér–Rao bound.

## Result
(see `flim_results.png`)
- All three estimators are essentially unbiased in this idealized setting.
- **Poisson MLE reaches the Cramér–Rao bound**; unweighted least squares needs roughly 2x the photons for the same precision at low counts, because its Gaussian-noise assumption is wrong for photon counting.
- The phasor method is close to the bound, is closed-form (no iterative fitting), and so is attractive for fast or real-time imaging.

## Limitations (what I don't know yet)
- Lifetimes are assumed known. Real data needs calibration and may have more than two components.
- No background, afterpulsing, detector jitter tails, pile-up, or autofluorescence from other species.
- IRF is an ideal Gaussian; real IRFs are measured and asymmetric.
- Next steps: add background and a measured IRF, estimate lifetimes jointly, and look at spatially regularized estimation on image data.

## Run

Windows (PowerShell):

    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python flim_sim.py

macOS / Linux:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python flim_sim.py

Output: `flim_results.png` and a table of results printed to the terminal.
