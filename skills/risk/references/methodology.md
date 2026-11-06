# Factor risk decomposition: methodology

## Model

Returns follow a linear factor model, so the asset covariance is

    Sigma = B Sigma_f B' + D

where `B` is the N by K matrix of factor exposures, `Sigma_f` is the K by K factor covariance, and `D` is the diagonal matrix of specific variances (Grinold and Kahn, *Active Portfolio Management*).

## Total risk and contributions

For portfolio weights `w`:

    sigma_p   = sqrt(w' Sigma w)
    MR_i      = (Sigma w)_i / sigma_p          marginal risk, d sigma_p / d w_i
    RC_i      = w_i * MR_i                     contribution of position i

Because volatility is homogeneous of degree one in the weights, Euler's theorem gives

    sum_i RC_i = sigma_p

exactly. This identity is the main check of the skill.

## Factor and specific split

With portfolio factor exposures `x = B' w`:

    FRC_k = x_k (Sigma_f x)_k / sigma_p        contribution of factor k
    SRC_i = w_i^2 D_i / sigma_p                specific contribution of asset i

and `sum_k FRC_k + sum_i SRC_i = sigma_p`, since `w' Sigma w = x' Sigma_f x + w' D w`.

## Units

The script annualizes with `sqrt(periods_per_year)`. Supply annual covariances with the default of 1, or per period covariances with the matching frequency (252 daily, 52 weekly, 12 monthly).

## Interpretation notes

- A share above one for a position means the rest of the book partly offsets it; shares can be negative for hedges.
- Sample covariance from returns is noisy when assets are many relative to observations. Prefer a factor model for large books, and require at least several observations per asset.
- Marginal risk tells which trade changes risk most per unit of capital. It is a local measure; large trades change the covariance weights and should be rerun.
