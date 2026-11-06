#!/usr/bin/env python3
"""Implied earnings move from the at the money straddle: E|S_T - S_0| ~ C_ATM + P_ATM.

Either pass the two option prices directly (--call, --put), or an option chain from which the script
picks the first expiry on or after the reaction session and the strike nearest to spot.
Exit codes: 0 ok, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import sys

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
    ap.add_argument("--spot", type=float, required=True, help="underlying price at the time of the quotes")
    ap.add_argument("--call", type=float, help="at the money call price (mid)")
    ap.add_argument("--put", type=float, help="at the money put price (mid)")
    ap.add_argument("--strike", type=float, help="strike of --call and --put (default: spot)")
    ap.add_argument("--chain", help="CSV with columns expiry, strike, type (C or P), bid, ask")
    ap.add_argument("--reaction-date", help="first session that trades on the release, YYYY-MM-DD (chain mode)")
    ap.add_argument("--max-days-after", type=int, default=7,
                    help="largest gap in calendar days between reaction date and expiry before a check fails")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args()

    S = a.spot
    if not S > 0:
        fail("--spot must be positive")
    checks, picked = [], {}

    if a.chain:
        if not a.reaction_date:
            fail("chain mode needs --reaction-date")
        ch = pd.read_csv(a.chain, parse_dates=["expiry"])
        need = {"expiry", "strike", "type", "bid", "ask"}
        if not need.issubset(ch.columns):
            fail(f"chain needs columns {sorted(need)}")
        ch["type"] = ch["type"].astype(str).str.upper().str[0]
        rd = pd.Timestamp(a.reaction_date)
        exps = sorted(e for e in ch["expiry"].unique() if e >= rd)
        if not exps:
            fail(f"no expiry on or after {a.reaction_date}")
        exp = pd.Timestamp(exps[0])
        sub = ch[ch["expiry"] == exp]
        both = sorted(set(sub.loc[sub["type"] == "C", "strike"]) & set(sub.loc[sub["type"] == "P", "strike"]))
        if not both:
            fail(f"no strike with both a call and a put for expiry {exp:%Y-%m-%d}")
        K = min(both, key=lambda k: (abs(k - S), k))
        c = sub[(sub["type"] == "C") & (sub["strike"] == K)].iloc[0]
        p = sub[(sub["type"] == "P") & (sub["strike"] == K)].iloc[0]
        for name, row in (("call", c), ("put", p)):
            checks.append({"name": f"{name}_quote_valid", "passed": bool(0 <= row["bid"] <= row["ask"] and row["ask"] > 0),
                           "detail": f"bid {row['bid']}, ask {row['ask']}"})
        C, P = (c["bid"] + c["ask"]) / 2, (p["bid"] + p["ask"]) / 2
        gap = (exp - rd).days
        checks.append({"name": "expiry_close_to_release", "passed": 0 <= gap <= a.max_days_after,
                       "detail": f"expiry {exp:%Y-%m-%d} is {gap} days after the reaction date"})
        picked = {"expiry": f"{exp:%Y-%m-%d}", "reaction_date": a.reaction_date,
                  "call_bid": float(c["bid"]), "call_ask": float(c["ask"]),
                  "put_bid": float(p["bid"]), "put_ask": float(p["ask"])}
    else:
        if a.call is None or a.put is None:
            fail("supply --call and --put, or --chain with --reaction-date")
        C, P, K = a.call, a.put, (a.strike if a.strike is not None else S)
        checks.append({"name": "prices_positive", "passed": C > 0 and P > 0, "detail": f"call {C}, put {P}"})

    straddle = float(C + P)
    checks.append({"name": "strike_near_spot", "passed": abs(K - S) / S <= 0.025,
                   "detail": f"strike {K} vs spot {S} ({(K - S) / S:+.2%})"})
    out = {
        "skill": SKILL, "version": VERSION, "step": "implied_move",
        "inputs": {"spot": S, "strike": float(K), "call": float(C), "put": float(P), **picked},
        "results": {
            "straddle": straddle,
            "implied_move": straddle,
            "implied_move_pct": straddle / S,
            "breakeven_low": float(K - straddle), "breakeven_high": float(K + straddle),
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
