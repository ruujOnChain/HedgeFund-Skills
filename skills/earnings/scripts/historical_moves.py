#!/usr/bin/env python3
"""Realized moves on past earnings reaction days, and their comparison with the implied move.

Timing decides the reaction session:
  bmo (before the open): previous close to the close of the release date
  amc (after the close): close of the release date to the next close
Exit codes: 0 ok, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

SKILL, VERSION = "earnings", "1.0"


def _np(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def fail(msg):
    print(f"DATA CONTRACT: {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", required=True, help="CSV: date column, then daily closing price columns")
    ap.add_argument("--asset", required=True, help="column of the security")
    ap.add_argument("--events", required=True, help="CSV with columns date, timing (bmo or amc)")
    ap.add_argument("--implied", type=float, help="implied move as a fraction (from implied_move.py) to compare")
    ap.add_argument("--min-events", type=int, default=4, help="fewest past releases for a meaningful history")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args()

    px = pd.read_csv(a.prices, parse_dates=[0], index_col=0).sort_index()
    if a.asset not in px.columns:
        fail(f"'{a.asset}' not in {list(px.columns)}")
    s = px[a.asset].astype(float).dropna()
    ev = pd.read_csv(a.events, parse_dates=["date"])
    if not {"date", "timing"}.issubset(ev.columns):
        fail("events need columns date, timing")
    ev["timing"] = ev["timing"].astype(str).str.lower().str.strip()
    bad = ev.loc[~ev["timing"].isin(["bmo", "amc"]), "timing"].tolist()
    if bad:
        fail(f"timing must be bmo or amc, found {bad}")

    rows, skipped = [], []
    for d, timing in ev.sort_values("date")[["date", "timing"]].itertuples(index=False):
        if d not in s.index:
            skipped.append({"date": f"{d:%Y-%m-%d}", "reason": "release date is not a trading day in the data"})
            continue
        i = s.index.get_loc(d)
        a0, a1 = (i - 1, i) if timing == "bmo" else (i, i + 1)
        if a0 < 0 or a1 >= len(s):
            skipped.append({"date": f"{d:%Y-%m-%d}", "reason": "reaction session outside the data"})
            continue
        mv = s.iloc[a1] / s.iloc[a0] - 1
        rows.append({"date": f"{d:%Y-%m-%d}", "timing": timing, "from": f"{s.index[a0]:%Y-%m-%d}",
                     "to": f"{s.index[a1]:%Y-%m-%d}", "move": float(mv), "abs_move": float(abs(mv))})

    absm = np.array([r["abs_move"] for r in rows])
    res = {"moves": rows, "skipped": skipped, "count": len(rows)}
    if len(rows):
        res.update({"mean_abs_move": float(absm.mean()), "median_abs_move": float(np.median(absm)),
                    "max_abs_move": float(absm.max()),
                    "share_up": float(np.mean([r["move"] > 0 for r in rows]))})
        if a.implied is not None:
            res["implied_move_pct"] = a.implied
            res["implied_to_mean_ratio"] = a.implied / res["mean_abs_move"] if res["mean_abs_move"] > 0 else None
            res["share_exceeding_implied"] = float(np.mean(absm > a.implied))

    checks = [
        {"name": "enough_history", "passed": len(rows) >= a.min_events, "detail": f"{len(rows)} releases measured"},
        {"name": "all_releases_measured", "passed": not skipped, "detail": f"{len(skipped)} skipped"},
    ]
    out = {"skill": SKILL, "version": VERSION, "step": "historical_moves",
           "inputs": {"asset": a.asset, "events": len(ev)}, "results": res, "checks": checks}
    out["passed"] = all(c["passed"] for c in checks)
    text = json.dumps(out, indent=2, default=_np)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
