# Event study: methodology

Reference: A. C. MacKinlay, "Event studies in economics and finance," *Journal of Economic Literature*, 1997.

## Timeline

Trading days are indexed relative to the event, with day 0 the first session on or after the event date.

    estimation window      gap          event window
    [-270 ........ -21]  [-20 .. -2]   [-1 .. +1]   (defaults)

The estimation window ends well before the event, so the reaction being measured never shapes the estimate of normal returns.

## Market model

On the estimation window, ordinary least squares fits

    R_t = alpha + beta R_m,t + e_t

and the residual standard deviation, with two estimated parameters, is

    sigma_AR = sqrt( sum e_t^2 / (n - 2) )

## Abnormal returns

Over the event window `[t1, t2]` of length `L = t2 - t1 + 1`:

    AR_t = R_t - (alpha + beta R_m,t)
    CAR  = sum_{t=t1}^{t2} AR_t
    t    = CAR / (sigma_AR sqrt(L))

The t statistic treats abnormal returns as independent with constant variance. The reported p value uses the normal approximation.

## Choices and their reasons

- **Total return prices.** Dividends otherwise appear as negative returns on ex dates.
- **Window of -1 to +1.** Captures leakage the day before and the full reaction for releases late in the session.
- **250 day estimation.** About one year, long enough for a stable beta and short enough to reflect the current business.
- **After the close.** Releases after the close move the next session; that session is day 0.

## Limits

- One event per run. For many events, run the skill per event and average CARs; the variance of the average needs cross sectional care when events cluster in time.
- Event induced variance can make the t statistic too large; a longer horizon or a variance adjusted test is the next step when that matters.
