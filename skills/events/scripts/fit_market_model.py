#!/usr/bin/env python3
"""Fit the market model R_t = alpha + beta R_m,t + e_t on an estimation window that ends before the event.

Default window: 250 trading days ending 20 trading days before the event (days -270 to -21).
Writes a JSON fit that scripts/car.py consumes.
Exit codes: 0 ok, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import event_index, fail, load_returns, ols  # noqa: E402

SKILL, VERSION = "events", "1.0"


def _np(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", required=True, help="CSV: date column, then total return price columns")
    ap.add_argument("--asset", required=True, help="column of the security")
    ap.add_argument("--market", required=True, help="column of the market index")
    ap.add_argument("--event-date", required=True, help="YYYY-MM-DD; day 0 is the first session on or after it")
    ap.add_argument("--est-len", type=int, default=250, help="estimation window length in trading days")
    ap.add_argument("--gap", type=int, default=20, help="trading days between estimation end and the event")
    ap.add_argument("--out", default="fit.json", help="where to write the fit (default fit.json)")
    a = ap.parse_args()

    if a.est_len < 30:
        fail("estimation window must be at least 30 days")
    ret = load_returns(a.prices, a.asset, a.market)
    e = event_index(ret, a.event_date)
    start, end = e - a.gap - a.est_len, e - a.gap          # end is exclusive
    if start < 0:
        fail(f"need {a.est_len + a.gap} trading days of returns before the event, have {e}")
    est = ret.iloc[start:end]
    alpha, beta, sigma, r2 = ols(est[a.asset].values, est[a.market].values)

    checks = [
        {"name": "estimation_window_complete", "passed": len(est) == a.est_len,
         "detail": f"{len(est)} of {a.est_len} days"},
        {"name": "estimation_ends_before_event", "passed": end <= e,
         "detail": f"last estimation day {est.index[-1]:%Y-%m-%d}, day 0 {ret.index[e]:%Y-%m-%d}"},
        {"name": "residual_volatility_positive", "passed": bool(np.isfinite(sigma) and sigma > 0),
         "detail": f"sigma_AR {sigma:.6g}"},
    ]
    out = {
        "skill": SKILL, "version": VERSION, "step": "fit_market_model",
        "inputs": {"prices": a.prices, "asset": a.asset, "market": a.market, "event_date": a.event_date,
                   "est_len": a.est_len, "gap": a.gap},
        "results": {
            "day0": f"{ret.index[e]:%Y-%m-%d}",
            "estimation_start": f"{est.index[0]:%Y-%m-%d}", "estimation_end": f"{est.index[-1]:%Y-%m-%d}",
            "n": len(est), "alpha": alpha, "beta": beta, "sigma_ar": sigma, "r_squared": r2,
            "estimation_end_offset": -(a.gap + 1),
        },
        "checks": checks,
    }
    out["passed"] = all(c["passed"] for c in checks)
    text = json.dumps(out, indent=2, default=_np)
    with open(a.out, "w") as f:
        f.write(text + "\n")
    print(text)
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
