#!/usr/bin/env python3
"""Regime probabilities from relative market features, using only information available at each date.

1. Features (all trailing): short to long volatility ratio, trend strength, drawdown,
   and average cross asset correlation when more than one series is supplied.
2. A Gaussian hidden Markov model is fitted by EM on a training period that ends at --train-end.
3. The forward filter gives pi_t(k) = P(S_t = k | F_t) for every date, using data up to t only.
4. With target exposures per regime, the blended exposure is w_t = sum_k pi_t(k) w_k.

Exit codes: 0 ok, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

SKILL, VERSION = "regimes", "1.0"


def _np(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)
LABELS = {2: ["low_vol", "high_vol"], 3: ["low_vol", "normal", "high_vol"]}
SHORT, LONG, TREND, CORR = 21, 252, 126, 63


def fail(msg):
    print(f"DATA CONTRACT: {msg}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- features
def features(px, asset, others):
    r = np.log(px).diff()
    ra = r[asset]
    f = pd.DataFrame(index=px.index)
    f["vol_ratio"] = ra.rolling(SHORT).std() / ra.rolling(LONG).std()
    f["trend"] = ra.rolling(TREND).sum() / (ra.rolling(TREND).std() * np.sqrt(TREND))
    f["drawdown"] = px[asset] / px[asset].rolling(LONG).max() - 1.0
    cols = [asset] + others
    if len(cols) > 1:
        pairs = [(i, j) for k, i in enumerate(cols) for j in cols[k + 1:]]
        f["avg_corr"] = sum(r[i].rolling(CORR).corr(r[j]) for i, j in pairs) / len(pairs)
    return f.dropna()


# ---------------------------------------------------------------- HMM
def log_emission(X, means, var):
    # diagonal Gaussian log density, shape (T, K)
    d = X[:, None, :] - means[None, :, :]
    return -0.5 * (np.log(2 * np.pi * var)[None] + d * d / var[None]).sum(axis=2)


def lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def forward(logB, logpi, logA):
    T, K = logB.shape
    la = np.empty((T, K))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logB[t] + lse(la[t - 1][:, None] + logA, axis=0)
    return la


def backward(logB, logA):
    T, K = logB.shape
    lb = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        lb[t] = lse(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    return lb


def fit_hmm(X, K, order_col, max_iter=500, tol=1e-6, var_floor=1e-3):
    T, d = X.shape
    groups = np.array_split(np.argsort(X[:, order_col], kind="stable"), K)
    means = np.array([X[g].mean(axis=0) for g in groups])
    var = np.array([X[g].var(axis=0) + var_floor for g in groups])
    A = np.full((K, K), 0.1 / (K - 1)) + np.eye(K) * (0.9 - 0.1 / (K - 1))
    pi = np.full(K, 1.0 / K)
    prev, converged, it = -np.inf, False, 0
    for it in range(1, max_iter + 1):
        logB = log_emission(X, means, var)
        la, lb = forward(logB, np.log(pi), np.log(A)), backward(logB, np.log(A))
        ll = float(lse(la[-1], axis=0))
        lg = la + lb - ll
        g = np.exp(lg)
        lxi = la[:-1, :, None] + np.log(A)[None] + (logB[1:] + lb[1:])[:, None, :] - ll
        xi = np.exp(lxi).sum(axis=0)
        pi = g[0] / g[0].sum()
        A = xi / xi.sum(axis=1, keepdims=True)
        w = g.sum(axis=0)
        means = (g.T @ X) / w[:, None]
        var = np.array([(g[:, k, None] * (X - means[k]) ** 2).sum(axis=0) / w[k] for k in range(K)]) + var_floor
        if ll - prev < tol:
            converged = True
            break
        prev = ll
    return {"pi": pi, "A": A, "means": means, "var": var, "loglik": ll, "iterations": it, "converged": converged}


def filtered(X, m):
    la = forward(log_emission(X, m["means"], m["var"]), np.log(m["pi"]), np.log(m["A"]))
    return np.exp(la - lse(la, axis=1)[:, None])


# ---------------------------------------------------------------- pipeline
def run(px, asset, others, train_end, model=None, stats=None, K=3):
    f = features(px, asset, others)
    if stats is None:
        tr = f.loc[:train_end]
        stats = (tr.mean(), tr.std(ddof=0))
    Z = ((f - stats[0]) / stats[1]).values
    if model is None:
        n_train = int((f.index <= pd.Timestamp(train_end)).sum())
        model = fit_hmm(Z[:n_train], K, order_col=list(f.columns).index("vol_ratio"))
        model["n_train"] = n_train
    return f, Z, filtered(Z, model), model, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", required=True, help="CSV: date column, then price levels per series")
    ap.add_argument("--asset", required=True, help="series whose regime is measured")
    ap.add_argument("--corr-assets", default=None,
                    help="comma separated series for average correlation (default: all other columns)")
    ap.add_argument("--states", type=int, default=3, choices=[2, 3])
    ap.add_argument("--train-end", default=None,
                    help="last date used to fit the model (default: halfway through the feature history)")
    ap.add_argument("--exposures", default=None,
                    help="target exposure per regime, ordered low_vol..high_vol, e.g. 1.0,0.6,0.2")
    ap.add_argument("--as-of", default=None, help="report date (default: last date in the file)")
    ap.add_argument("--history", default=None, help="write daily probabilities to this CSV")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args()

    px = pd.read_csv(a.prices, parse_dates=[0], index_col=0).sort_index()
    if a.asset not in px.columns:
        fail(f"'{a.asset}' not in {list(px.columns)}")
    others = [c for c in px.columns if c != a.asset] if a.corr_assets is None else \
        [c.strip() for c in a.corr_assets.split(",") if c.strip()]
    missing = [c for c in others if c not in px.columns]
    if missing:
        fail(f"correlation series not found: {missing}")
    px = px[[a.asset] + others].astype(float)
    if px.isna().any().any() or (px <= 0).any().any():
        fail("prices must be positive with no gaps; align the series before running")
    if a.as_of:
        px = px.loc[:a.as_of]
    K = a.states
    exposures = None
    if a.exposures:
        exposures = np.array([float(x) for x in a.exposures.split(",")])
        if len(exposures) != K:
            fail(f"--exposures needs {K} values, one per regime")

    f_all = features(px, a.asset, others)
    if len(f_all) < 300:
        fail(f"only {len(f_all)} dates with complete features; need several years of daily prices")
    default_train = a.train_end is None
    train_end = f_all.index[len(f_all) // 2] if default_train else pd.Timestamp(a.train_end)

    f, Z, P, model, stats = run(px, a.asset, others, train_end, K=K)
    if model["n_train"] < 250:
        fail(f"training period has {model['n_train']} dates; need at least 250")

    # order states from lowest to highest volatility ratio so labels are stable
    raw_means = model["means"] * stats[1].values + stats[0].values
    order = np.argsort(raw_means[:, list(f.columns).index("vol_ratio")])
    P, labels = P[:, order], LABELS[K]
    A = model["A"][np.ix_(order, order)]

    # no future data: rerun features and filter on prices cut at a check date, compare probabilities
    t_check = f.index[max(model["n_train"], (model["n_train"] + len(f)) // 2) - 1]
    _, _, Pc, _, _ = run(px.loc[:t_check], a.asset, others, train_end, model=model, stats=stats, K=K)
    no_lookahead = float(np.abs(Pc[-1][order] - P[f.index.get_loc(t_check)]).max())

    last = f.index[-1]
    p_last = P[-1]
    res = {
        "as_of": f"{last:%Y-%m-%d}",
        "probabilities": {lab: round(float(p), 6) for lab, p in zip(labels, p_last)},
        "most_likely": labels[int(np.argmax(p_last))],
        "features": {c: float(v) for c, v in f.iloc[-1].items()},
        "regime_means": {lab: {c: float(raw_means[order[i], j]) for j, c in enumerate(f.columns)}
                         for i, lab in enumerate(labels)},
        "transition_matrix": A.tolist(),
        "expected_duration_days": {lab: float(1 / (1 - A[i, i])) for i, lab in enumerate(labels)},
    }
    if exposures is not None:
        res["target_exposures"] = {lab: float(x) for lab, x in zip(labels, exposures)}
        res["blended_exposure"] = float(p_last @ exposures)

    checks = [
        {"name": "probabilities_sum_to_one", "passed": float(np.abs(P.sum(axis=1) - 1).max()) < 1e-9,
         "detail": f"max error {float(np.abs(P.sum(axis=1) - 1).max()):.2e}"},
        {"name": "no_future_data", "passed": no_lookahead < 1e-10,
         "detail": f"probabilities on {t_check:%Y-%m-%d} recomputed from data up to that date differ by {no_lookahead:.2e}"},
        {"name": "model_converged", "passed": bool(model["converged"]),
         "detail": f"{model['iterations']} EM iterations"},
        {"name": "report_date_after_training", "passed": last > train_end,
         "detail": f"training ends {train_end:%Y-%m-%d}, report date {last:%Y-%m-%d}"},
    ]
    out = {
        "skill": SKILL, "version": VERSION,
        "inputs": {"asset": a.asset, "corr_assets": others, "states": K,
                   "train_start": f"{f.index[0]:%Y-%m-%d}", "train_end": f"{train_end:%Y-%m-%d}",
                   "train_end_defaulted": default_train, "n_train": model["n_train"],
                   "windows": {"short_vol": SHORT, "long_vol": LONG, "trend": TREND, "correlation": CORR}},
        "results": res, "checks": checks,
    }
    out["passed"] = all(c["passed"] for c in checks)
    if a.history:
        h = pd.DataFrame(P, index=f.index, columns=labels)
        if exposures is not None:
            h["blended_exposure"] = P @ exposures
        h["in_training"] = h.index <= train_end
        h.to_csv(a.history, index_label="date", float_format="%.6f")
    text = json.dumps(out, indent=2, default=_np)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
