# Strategy validation: methodology

References:
- D. H. Bailey and M. López de Prado, "The Sharpe ratio efficient frontier," *Journal of Risk*, 2012.
- D. H. Bailey and M. López de Prado, "The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non normality," *Journal of Portfolio Management*, 2014.

## Estimates

From `T` periodic returns: the per period Sharpe ratio `SR = mean / standard deviation`, skewness `g3`, and kurtosis `g4` (non excess, so 3 for a normal distribution).

## Probabilistic Sharpe ratio

The probability that the true Sharpe ratio exceeds a benchmark `SR*`:

    PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )

It rewards longer track records and penalizes negative skew and fat tails.

## Deflated Sharpe ratio

When the strategy is the best of `N` independent trials whose Sharpe ratios have variance `V`, the expected maximum Sharpe ratio under no skill is approximately

    SR0 = sqrt(V) ( (1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e)) )

with `gamma = 0.5772...` the Euler and Mascheroni constant and `Z^-1` the inverse standard normal distribution. Then

    DSR = PSR(SR0)

For `N = 1`, there is no selection and the skill uses `SR0 = 0`.

## Minimum track record length

The number of periods for which PSR against `SR0` reaches confidence `1 - alpha`:

    MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (Z_alpha / (SR - SR0))^2

## Conventions

- All Sharpe ratios inside the formulas are per period, matching the return frequency. Annualized values (times `sqrt(periods_per_year)`) are reported for reading only.
- `N` counts independent trials. Highly correlated variations count for less than one each; if the user can group trials into clusters, the number of clusters is a reasonable `N`.
- Returns should be net of costs. Gross returns overstate every statistic here.
