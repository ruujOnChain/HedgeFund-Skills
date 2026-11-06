#!/usr/bin/env python3
"""Abnormal returns, cumulative abnormal return (CAR), and its t statistic over an event window.

AR_t = R_t - (alpha + beta R_m,t)       CAR = sum of AR_t over the window
t    = CAR / (sigma_AR * sqrt(L))        L = window length in days
Reads the fit written by scripts/fit_market_model.py.
Exit codes: 0 ok, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import os
import sys
from statistics import NormalDist

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import event_index, fail, load_returns  # noqa: E402

SKILL, VERSION = "events", "1.0"


def _np(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default="fit.json", help="JSON from fit_market_model.py")
    ap.add_argument("--window", nargs=2, type=int, default=[-1, 1], metavar=("FIRST", "LAST"),
                    help="event window in trading days relative to day 0 (default -1 1)")
    ap.add_argument("--prices", help="override the price file recorded in the fit")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args()

    with open(a.fit) as f:
        fit = json.load(f)
    if not fit.get("passed", False):
        fail("the market model fit did not pass its checks; rerun fit_market_model.py")
    inp, res = fit["inputs"], fit["results"]
    first, last = a.window
    if last < first:
        fail("window LAST must be on or after FIRST")

    ret = load_returns(a.prices or inp["prices"], inp["asset"], inp["market"])
    e = event_index(ret, inp["event_date"])
    lo, hi = e + first, e + last
    if lo < 0 or hi >= len(ret):
        fail(f"event window {first}..{last} runs outside the data")
    win = ret.iloc[lo:hi + 1]
    ar = win[inp["asset"]] - (res["alpha"] + res["beta"] * win[inp["market"]])
    L = len(ar)
    car = float(ar.sum())
    t = car / (res["sigma_ar"] * np.sqrt(L))
    p = 2 * (1 - NormalDist().cdf(abs(t)))

    est_end_offset = res["estimation_end_offset"]
    checks = [
        {"name": "windows_do_not_overlap", "passed": est_end_offset < first,
         "detail": f"estimation ends at day {est_end_offset}, event window starts at day {first}"},
        {"name": "event_window_complete", "passed": L == last - first + 1, "detail": f"{L} days"},
        {"name": "same_day_zero_as_fit", "passed": f"{ret.index[e]:%Y-%m-%d}" == res["day0"],
         "detail": f"day 0 {ret.index[e]:%Y-%m-%d}"},
    ]
    out = {
        "skill": SKILL, "version": VERSION, "step": "car",
        "inputs": {"asset": inp["asset"], "market": inp["market"], "event_date": inp["event_date"],
                   "day0": res["day0"], "window": [first, last],
                   "estimation": [res["estimation_start"], res["estimation_end"]]},
        "results": {
            "car": car, "t_stat": float(t), "p_value_normal": float(p), "window_length": L,
            "alpha": res["alpha"], "beta": res["beta"], "sigma_ar": res["sigma_ar"],
            "daily": [{"date": f"{d:%Y-%m-%d}", "day": first + i, "return": float(win[inp['asset']].iloc[i]),
                       "market": float(win[inp['market']].iloc[i]), "abnormal": float(v)}
                      for i, (d, v) in enumerate(ar.items())],
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
