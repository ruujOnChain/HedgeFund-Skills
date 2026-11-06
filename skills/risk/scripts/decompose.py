#!/usr/bin/env python3
"""Decompose portfolio volatility by position and, with a factor model, by factor.

Three ways to supply risk:
  1. Factor model:      --exposures, --factor-cov, --specific-var
  2. Full covariance:   --cov
  3. Sample estimate:   --returns  (covariance estimated from historical returns)

Every number is computed here; the agent only reads the JSON this prints.
Exit codes: 0 all checks pass, 1 a check failed, 2 the inputs broke the data contract.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

SKILL, VERSION = "risk", "1.0"


def _np(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def fail(msg):
    print(f"DATA CONTRACT: {msg}", file=sys.stderr)
    sys.exit(2)


def read_weights(path):
    df = pd.read_csv(path)
    if not {"asset", "weight"}.issubset(df.columns):
        fail(f"{path} needs columns 'asset' and 'weight'")
    if df["asset"].duplicated().any():
        fail(f"duplicate assets in {path}: {df.loc[df['asset'].duplicated(), 'asset'].tolist()}")
    w = df.set_index("asset")["weight"].astype(float)
    if not np.isfinite(w.values).all():
        fail("weights contain missing or infinite values")
    return w


def read_matrix(path, name):
    m = pd.read_csv(path, index_col=0)
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)
    if not np.isfinite(m.values.astype(float)).all():
        fail(f"{name} contains missing or infinite values")
    return m.astype(float)


def require_assets(w, index, name):
    missing = [a for a in w.index if a not in index]
    if missing:
        fail(f"assets in weights but not in {name}: {missing}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="CSV with columns asset, weight")
    ap.add_argument("--exposures", help="CSV: asset rows, factor columns (B)")
    ap.add_argument("--factor-cov", help="CSV: factor by factor covariance (Sigma_f)")
    ap.add_argument("--specific-var", help="CSV with columns asset, specific_var (D)")
    ap.add_argument("--cov", help="CSV: asset by asset covariance")
    ap.add_argument("--returns", help="CSV: date column then one column per asset (simple returns)")
    ap.add_argument("--min-obs", type=int, default=60, help="minimum rows when using --returns")
    ap.add_argument("--periods-per-year", type=float, default=None,
                    help="annualize: covariance is per period. Default 1 (already annual), 252 for --returns")
    ap.add_argument("--top", type=int, default=10, help="number of largest contributors to list")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args()

    w = read_weights(a.weights)
    assets = list(w.index.astype(str))
    w.index = assets
    mode = None
    B = F = D = None

    if a.exposures or a.factor_cov or a.specific_var:
        if not (a.exposures and a.factor_cov and a.specific_var):
            fail("factor mode needs --exposures, --factor-cov and --specific-var together")
        mode = "factor_model"
        B = read_matrix(a.exposures, "exposures")
        F = read_matrix(a.factor_cov, "factor covariance")
        s = pd.read_csv(a.specific_var)
        if not {"asset", "specific_var"}.issubset(s.columns):
            fail("--specific-var needs columns asset, specific_var")
        s["asset"] = s["asset"].astype(str)
        s = s.set_index("asset")["specific_var"].astype(float)
        require_assets(w, B.index, "exposures")
        require_assets(w, s.index, "specific variances")
        if list(F.index) != list(F.columns):
            fail("factor covariance must have the same factors as rows and columns, in the same order")
        if set(B.columns) != set(F.columns):
            fail(f"exposure factors {list(B.columns)} differ from covariance factors {list(F.columns)}")
        B = B.loc[assets, list(F.columns)]
        D = s.loc[assets]
        if (D < 0).any():
            fail("specific variances must be non negative")
        cov = B.values @ F.values @ B.values.T + np.diag(D.values)
        ppy = a.periods_per_year or 1.0
    elif a.cov:
        mode = "covariance"
        C = read_matrix(a.cov, "covariance")
        require_assets(w, C.index, "covariance rows")
        require_assets(w, C.columns, "covariance columns")
        cov = C.loc[assets, assets].values
        ppy = a.periods_per_year or 1.0
    elif a.returns:
        mode = "sample_covariance"
        R = pd.read_csv(a.returns, index_col=0)
        R.columns = R.columns.astype(str)
        require_assets(w, R.columns, "returns")
        R = R[assets].astype(float)
        if R.isna().any().any():
            fail("returns contain gaps; align dates and fill or drop before running")
        if len(R) < a.min_obs:
            fail(f"only {len(R)} observations, need at least {a.min_obs}")
        cov = np.atleast_2d(np.cov(R.values, rowvar=False, ddof=1))
        ppy = a.periods_per_year or 252.0
    else:
        fail("supply a factor model, --cov, or --returns")

    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    cov = 0.5 * (cov + cov.T)
    wv = w.values
    var = float(wv @ cov @ wv)
    if var <= 0:
        fail("portfolio variance is not positive; check weights and covariance")
    sig = np.sqrt(var)
    mrc = cov @ wv / sig                     # marginal risk, d sigma / d w_i
    rc = wv * mrc                            # contribution of each position
    scale = np.sqrt(ppy)

    out = {
        "skill": SKILL, "version": VERSION, "mode": mode,
        "periods_per_year": ppy,
        "inputs": {"n_assets": len(assets), "gross_exposure": float(np.abs(wv).sum()), "net_exposure": float(wv.sum())},
        "results": {"total_volatility": sig * scale},
    }

    pos = pd.DataFrame({"weight": wv, "marginal_risk": mrc * scale, "contribution": rc * scale,
                        "share": rc / sig}, index=assets)
    pos = pos.reindex(pos["contribution"].abs().sort_values(ascending=False).index)
    out["results"]["positions"] = [
        {"asset": k, **{c: float(v) for c, v in r.items()}} for k, r in pos.head(a.top).iterrows()
    ]
    if len(pos) > a.top:
        rest = pos.iloc[a.top:]
        out["results"]["other_positions"] = {"count": int(len(rest)), "contribution": float(rest["contribution"].sum()),
                                             "share": float(rest["share"].sum())}

    checks = []
    eig_min = float(np.linalg.eigvalsh(cov).min())
    tol_psd = -1e-10 * max(1.0, float(np.abs(np.diag(cov)).max()))
    checks.append({"name": "covariance_positive_semidefinite", "passed": eig_min >= tol_psd,
                   "detail": f"smallest eigenvalue {eig_min:.3e}"})
    gap = abs(rc.sum() - sig)
    checks.append({"name": "position_contributions_sum_to_total", "passed": gap <= 1e-9 * max(1.0, sig),
                   "detail": f"sum {rc.sum() * scale:.12g} vs total {sig * scale:.12g}"})

    if mode == "factor_model":
        x = B.values.T @ wv                                  # portfolio factor exposures
        frc = x * (F.values @ x) / sig                       # factor contributions
        src = wv ** 2 * D.values / sig                       # specific contributions
        fac = pd.DataFrame({"exposure": x, "contribution": frc * scale, "share": frc / sig}, index=list(F.columns))
        fac = fac.reindex(fac["contribution"].abs().sort_values(ascending=False).index)
        out["results"]["factors"] = [{"factor": k, **{c: float(v) for c, v in r.items()}} for k, r in fac.iterrows()]
        out["results"]["factor_total"] = {"contribution": float(frc.sum() * scale), "share": float(frc.sum() / sig)}
        out["results"]["specific_total"] = {"contribution": float(src.sum() * scale), "share": float(src.sum() / sig)}
        g2 = abs(frc.sum() + src.sum() - sig)
        checks.append({"name": "factor_plus_specific_sum_to_total", "passed": g2 <= 1e-9 * max(1.0, sig),
                       "detail": f"factor {frc.sum() * scale:.6g} + specific {src.sum() * scale:.6g} vs total {sig * scale:.6g}"})

    out["checks"] = checks
    out["passed"] = all(c["passed"] for c in checks)
    text = json.dumps(out, indent=2, default=_np)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
