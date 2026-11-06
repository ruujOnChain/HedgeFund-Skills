#!/usr/bin/env python3
"""Position size from fractional Kelly, capped by a volatility target and a maximum weight.

  full Kelly        f* = mu / sigma^2
  fractional Kelly  f  = c f*
  volatility cap    w_vt = sigma_target / sigma
  weight            sign(mu) * min(|f|, w_vt, max_weight)

mu and sigma are annual: mu is expected return in excess of cash, sigma is volatility.
Exit codes: 0 ok, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

SKILL, VERSION = "sizing", "1.0"


def _np(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def fail(msg):
    print(f"DATA CONTRACT: {msg}", file=sys.stderr)
    sys.exit(2)


def growth(f, mu, sigma):
    """Long run growth rate of investing fraction f: g(f) = f mu - f^2 sigma^2 / 2."""
    return f * mu - 0.5 * f * f * sigma * sigma


def estimate_vol(path, column, ppy, halflife):
    df = pd.read_csv(path, index_col=0)
    if column not in df.columns:
        fail(f"column '{column}' not in {path}; columns are {list(df.columns)}")
    r = df[column].astype(float).dropna()
    if len(r) < 60:
        fail(f"need at least 60 returns to estimate volatility, have {len(r)}")
    if halflife:
        lam = 0.5 ** (1.0 / halflife)
        wts = lam ** np.arange(len(r))[::-1]
        wts /= wts.sum()
        m = float((wts * r.values).sum())
        var = float((wts * (r.values - m) ** 2).sum())
        method = f"exponentially weighted, halflife {halflife} periods"
    else:
        var = float(r.var(ddof=1))
        method = "equal weighted sample"
    return math.sqrt(var * ppy), len(r), method


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mu", type=float, required=True, help="annual expected excess return, decimal (0.06 = 6%%)")
    ap.add_argument("--sigma", type=float, help="annual volatility, decimal")
    ap.add_argument("--returns", help="CSV of periodic simple returns to estimate sigma (date column first)")
    ap.add_argument("--column", help="column of --returns to use")
    ap.add_argument("--periods-per-year", type=float, default=252.0)
    ap.add_argument("--halflife", type=float, default=None, help="exponential weighting halflife in periods")
    ap.add_argument("--kelly-fraction", type=float, default=0.5, help="fraction c of full Kelly (default 0.5)")
    ap.add_argument("--target-vol", type=float, default=None, help="annual volatility target for the position")
    ap.add_argument("--max-weight", type=float, default=1.0, help="cap on absolute weight (default 1.0)")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args()

    if a.sigma is None and not a.returns:
        fail("supply --sigma or --returns with --column")
    if a.sigma is not None and a.returns:
        fail("supply --sigma or --returns, not both")
    if a.returns:
        if not a.column:
            fail("--returns needs --column")
        sigma, n_obs, method = estimate_vol(a.returns, a.column, a.periods_per_year, a.halflife)
    else:
        sigma, n_obs, method = a.sigma, None, "supplied"
    mu, c = a.mu, a.kelly_fraction
    if not (math.isfinite(mu) and math.isfinite(sigma)) or sigma <= 0:
        fail("mu must be finite and sigma must be positive")
    if not 0 < c <= 1:
        fail("--kelly-fraction must be in (0, 1]")
    if a.max_weight <= 0:
        fail("--max-weight must be positive")
    if a.target_vol is not None and a.target_vol <= 0:
        fail("--target-vol must be positive")

    f_full = mu / sigma ** 2
    f_frac = c * f_full
    caps = {"kelly": abs(f_frac), "max_weight": a.max_weight}
    if a.target_vol is not None:
        caps["target_vol"] = a.target_vol / sigma
    binding = min(caps, key=caps.get)
    weight = math.copysign(caps[binding], mu) if mu != 0 else 0.0
    g_max = growth(f_full, mu, sigma)
    g_w = growth(weight, mu, sigma)

    checks = [
        {"name": "kelly_matches_closed_form", "passed": abs(mu - f_full * sigma ** 2) <= 1e-12 * max(1, abs(mu)),
         "detail": "first order condition mu - f* sigma^2 = 0"},
        {"name": "within_max_weight", "passed": abs(weight) <= a.max_weight + 1e-12,
         "detail": f"|w| {abs(weight):.6g} vs cap {a.max_weight}"},
    ]
    if a.target_vol is not None:
        checks.append({"name": "within_target_vol", "passed": abs(weight) * sigma <= a.target_vol + 1e-12,
                       "detail": f"position vol {abs(weight) * sigma:.6g} vs target {a.target_vol}"})

    out = {
        "skill": SKILL, "version": VERSION,
        "inputs": {"mu": mu, "sigma": sigma, "sigma_method": method, "n_obs": n_obs, "kelly_fraction": c,
                   "target_vol": a.target_vol, "max_weight": a.max_weight},
        "results": {
            "weight": weight, "binding_constraint": binding,
            "full_kelly": f_full, "fractional_kelly": f_frac,
            "vol_target_weight": caps.get("target_vol"),
            "position_volatility": abs(weight) * sigma,
            "expected_growth": g_w, "max_growth": g_max,
            "share_of_max_growth": (g_w / g_max) if g_max > 0 else None,
        },
        "checks": checks,
    }
    out["passed"] = all(ch["passed"] for ch in checks)
    text = json.dumps(out, indent=2, default=_np)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
