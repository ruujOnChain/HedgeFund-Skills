# Position sizing: methodology

## Kelly criterion

For a position with annual expected excess return `mu` and volatility `sigma`, the long run growth rate of investing a fraction `f` of capital is, to second order,

    g(f) = f mu - (1/2) f^2 sigma^2

It is maximized at

    f* = mu / sigma^2          g(f*) = mu^2 / (2 sigma^2)

Reference: J. L. Kelly, "A new interpretation of information rate," *Bell System Technical Journal*, 1956.

## Fractional Kelly

Investing `c f*` gives growth `g(c f*) = (2c - c^2) g(f*)` and volatility `c f* sigma`. At `c = 1/2`, growth is three quarters of the maximum and volatility is half. Because `mu` is estimated with error and overbetting is costly (growth falls to zero at `2 f*`), fractional Kelly is the professional default.

## Volatility target

To hold position volatility at `sigma_target`:

    w_vt = sigma_target / sigma

Scaling by inverse volatility keeps risk steady as markets change (Moreira and Muir, "Volatility managed portfolios," *Journal of Finance*, 2017).

## Combination used by the skill

    weight = sign(mu) * min( |c f*|, sigma_target / sigma, max_weight )

Kelly expresses conviction from `mu`; the volatility target and the maximum weight are risk limits. The smallest of the three binds, and the output names it.

## Volatility estimation

With a returns file, `sigma` is the sample standard deviation times `sqrt(periods_per_year)`, or an exponentially weighted estimate when `--halflife` is set. Shorter halflives react faster to changes in volatility and are noisier.

## Limits

- The growth formula is a continuous time approximation; it is accurate when per period returns are small.
- The skill sizes a single position. Correlations between positions require the portfolio view from the risk skill.
