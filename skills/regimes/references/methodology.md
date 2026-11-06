# Regime detection: methodology

## Features

All features at date `t` use prices up to and including `t`. With daily log returns `r`:

| Feature | Definition | Window |
| --- | --- | --- |
| `vol_ratio` | standard deviation of `r` over 21 days divided by the same over 252 days | 21 / 252 |
| `trend` | sum of `r` over 126 days divided by (standard deviation over 126 days times sqrt(126)) | 126 |
| `drawdown` | price divided by its 252 day high, minus one | 252 |
| `avg_corr` | average pairwise correlation of daily returns across all series | 63 |

The features are relative: a ratio of volatilities or a volatility scaled trend means the same thing for an equity index and for a currency, so the skill works across assets without retuning. Features are standardized with the mean and standard deviation of the training period only.

## Model

A hidden Markov model with `K` states, transition matrix `A`, initial distribution `pi`, and Gaussian emissions with diagonal covariance (Hamilton, "A new approach to the economic analysis of nonstationary time series and the business cycle," *Econometrica*, 1989). Parameters are estimated by expectation maximization (Baum Welch) on the training period, initialized by splitting training dates into `K` groups by volatility ratio. A small floor on variances keeps the fit stable.

## Filtered probabilities

The forward recursion gives the filtered probability

    pi_t(k) = P(S_t = k | F_t)

which uses observations up to `t` only. Smoothed probabilities, which use the whole sample, are never reported, because they look ahead.

## Exposure

With a target exposure `w_k` per regime:

    w_t = sum_k pi_t(k) w_k

## Point in time guarantees

1. Features are trailing windows.
2. Standardization and model parameters come from the training period only.
3. The filter runs forward in time.
4. The script verifies all three by recomputing everything from prices cut at a check date and comparing.

## Labels and persistence

States are ordered by their average volatility ratio and labeled `low_vol`, `normal`, `high_vol` (or `low_vol`, `high_vol`). Expected duration of regime `k` is `1 / (1 - A_kk)` days.

## Limits

- Parameters are fixed after training. Refit periodically with a later training end, and record the version used.
- Regimes are statistical states, not economic diagnoses. Describe them by their feature profiles.
