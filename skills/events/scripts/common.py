"""Shared loading and indexing helpers for the event study scripts."""
import sys

import numpy as np
import pandas as pd


def fail(msg):
    print(f"DATA CONTRACT: {msg}", file=sys.stderr)
    sys.exit(2)


def load_returns(path, asset, market):
    """Read a total return price file and return aligned simple returns for asset and market."""
    px = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    for col in (asset, market):
        if col not in px.columns:
            fail(f"column '{col}' not found in {path}; columns are {list(px.columns)}")
    px = px[[asset, market]].astype(float)
    if px.index.duplicated().any():
        fail("duplicate dates in the price file")
    if px.isna().any().any():
        bad = px[px.isna().any(axis=1)].index[:5].strftime("%Y-%m-%d").tolist()
        fail(f"missing prices on {bad}; prices must be complete total return levels")
    if (px <= 0).any().any():
        fail("prices must be positive total return levels")
    ret = px.pct_change().iloc[1:]
    return ret


def event_index(ret, event_date):
    """Position of the first trading day on or after the event date (day 0)."""
    d = pd.Timestamp(event_date)
    pos = int(ret.index.searchsorted(d, side="left"))
    if pos >= len(ret):
        fail(f"no trading day on or after {event_date} in the data")
    return pos


def ols(y, x):
    n = len(y)
    xm, ym = x.mean(), y.mean()
    beta = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    alpha = float(ym - beta * xm)
    resid = y - alpha - beta * x
    sigma = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    r2 = float(1 - (resid ** 2).sum() / ((y - ym) ** 2).sum())
    return alpha, beta, sigma, r2
