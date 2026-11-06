#!/usr/bin/env python3
"""Probabilistic Sharpe ratio (PSR) and deflated Sharpe ratio (DSR) for a selected backtest.

PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )
SR0      = sqrt(V) ( (1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e)) )
DSR      = PSR(SR0)

SR is per period (not annualized). V is the variance of per period Sharpe ratios across the N trials.
Exit codes: 0 ok, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import math
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

SKILL, VERSION = "validation", "1.0"


def _np(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)
EULER_GAMMA = 0.5772156649015329
N01 = NormalDist()


def fail(msg):
    print(f"DATA CONTRACT: {msg}", file=sys.stderr)
    sys.exit(2)


def moments(x):
    x = np.asarray(x, dtype=float)
    mu, sd = x.mean(), x.std(ddof=1)
    d = x - mu
    m2 = (d ** 2).mean()
    skew = float((d ** 3).mean() / m2 ** 1.5)
    kurt = float((d ** 4).mean() / m2 ** 2)              # non excess: 3 for a normal distribution
    return float(mu / sd), skew, kurt


def psr(sr, sr_star, T, skew, kurt):
    den = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if den <= 0:
        fail("non positive PSR denominator; returns are too skewed or heavy tailed for this approximation")
    return N01.cdf((sr - sr_star) * math.sqrt(T - 1) / math.sqrt(den)), den


def expected_max_sr(V, N):
    if N < 2:
        return 0.0
    return math.sqrt(V) * ((1 - EULER_GAMMA) * N01.inv_cdf(1 - 1 / N)
                           + EULER_GAMMA * N01.inv_cdf(1 - 1 / (N * math.e)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--returns", required=True, help="CSV of periodic returns of the selected strategy (date column first)")
    ap.add_argument("--column", required=True, help="column of the selected strategy")
    ap.add_argument("--trials", type=int, help="number of independent trials N behind the selection")
    ap.add_argument("--sr-variance", type=float, help="variance V of per period Sharpe ratios across trials")
    ap.add_argument("--trial-returns", help="CSV of returns of all trials (date column, one column per trial)")
    ap.add_argument("--trial-sharpes", help="CSV with a 'sharpe' column of per period Sharpe ratios of all trials")
    ap.add_argument("--sr-benchmark", type=float, default=0.0, help="per period benchmark Sharpe for PSR (default 0)")
    ap.add_argument("--confidence", type=float, default=0.95, help="DSR level required to pass (default 0.95)")
    ap.add_argument("--periods-per-year", type=float, default=252.0, help="for reporting annualized figures only")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args()

    df = pd.read_csv(a.returns, index_col=0)
    if a.column not in df.columns:
        fail(f"column '{a.column}' not in {list(df.columns)}")
    x = df[a.column].astype(float)
    if x.isna().any():
        fail("returns contain gaps; the selected strategy needs a complete series")
    T = len(x)
    if T < 30:
        fail(f"only {T} observations; need at least 30")
    sr, skew, kurt = moments(x.values)

    # variance of Sharpe ratios across trials
    trial_srs = None
    if a.trial_returns:
        tr = pd.read_csv(a.trial_returns, index_col=0).astype(float)
        trial_srs = np.array([moments(tr[c].dropna().values)[0] for c in tr.columns])
    elif a.trial_sharpes:
        ts = pd.read_csv(a.trial_sharpes)
        if "sharpe" not in ts.columns:
            fail("--trial-sharpes needs a 'sharpe' column")
        trial_srs = ts["sharpe"].astype(float).values
    if trial_srs is not None:
        if len(trial_srs) < 2:
            fail("need at least two trials to estimate the variance of Sharpe ratios")
        V = float(np.var(trial_srs, ddof=1))
        N = a.trials or len(trial_srs)
    else:
        if a.trials is None or a.sr_variance is None:
            fail("supply --trial-returns, --trial-sharpes, or both --trials and --sr-variance")
        V, N = a.sr_variance, a.trials
    if N < 1:
        fail("--trials must be at least 1")
    if V < 0:
        fail("--sr-variance must be non negative")

    psr0, den = psr(sr, a.sr_benchmark, T, skew, kurt)
    sr0 = expected_max_sr(V, N)
    dsr, _ = psr(sr, sr0, T, skew, kurt)
    z = N01.inv_cdf(a.confidence)
    min_trl = (1 + den * (z / (sr - sr0)) ** 2) if sr > sr0 else None
    ann = math.sqrt(a.periods_per_year)

    checks = [
        {"name": "enough_observations", "passed": T >= 60, "detail": f"T = {T}"},
        {"name": "trial_count_covers_supplied_trials",
         "passed": trial_srs is None or N >= len(trial_srs),
         "detail": f"N = {N}" + ("" if trial_srs is None else f", trials supplied = {len(trial_srs)}")},
        {"name": "positive_variance_term", "passed": den > 0, "detail": f"{den:.6g}"},
    ]
    out = {
        "skill": SKILL, "version": VERSION,
        "inputs": {"column": a.column, "T": T, "trials": N, "sr_variance": V,
                   "sr_benchmark": a.sr_benchmark, "confidence": a.confidence,
                   "periods_per_year": a.periods_per_year},
        "results": {
            "sharpe_per_period": sr, "sharpe_annualized": sr * ann,
            "skewness": skew, "kurtosis": kurt,
            "psr": psr0,
            "sr0_per_period": sr0, "sr0_annualized": sr0 * ann,
            "dsr": dsr,
            "passes": dsr >= a.confidence,
            "min_track_record_length": min_trl,
        },
        "checks": checks,
    }
    out["passed"] = all(c["passed"] for c in checks)
    text = json.dumps(out, indent=2, default=_np)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
