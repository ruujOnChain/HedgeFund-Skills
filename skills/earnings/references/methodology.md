# Earnings preparation: methodology

## Implied move

For an at the money straddle with strike `K` close to spot `S0`, expiring shortly after the release:

    E[ |S_T - S_0| ]  ~  C_ATM + P_ATM

Why it holds: the straddle pays `|S_T - K|` at expiry. With `K` near `S0`, low rates and a short horizon, its price is close to the risk neutral expected absolute move. Under a normal approximation the expected absolute move is `sigma sqrt(T) S0 sqrt(2/pi)`, about `0.80 sigma sqrt(T) S0`, and the at the money straddle price is about `0.80 sigma sqrt(T) S0` as well, so the two agree.

The skill reports `implied_move_pct = (C + P) / S0` and breakevens `K +/- (C + P)`.

## Choosing the options

- **Expiry**: the first expiry on or after the reaction session. Later expiries add ordinary daily variance on top of the event.
- **Strike**: the strike nearest spot that has both a call and a put. Mid prices, `(bid + ask) / 2`.
- **Timing**: quotes and spot should be taken together, ideally near the close before the reaction session.

## Realized moves

The reaction session depends on timing:

| Timing | From | To |
| --- | --- | --- |
| `bmo` (before the open) | close of the previous session | close of the release date |
| `amc` (after the close) | close of the release date | close of the next session |

Moves are simple close to close returns; the skill summarizes their absolute values (mean, median, maximum) and the share that were positive.

## Comparison

    ratio = implied_move_pct / mean_abs_move

A ratio above one means the options price a larger move than the recent record; below one, smaller. The share of past moves exceeding the implied move gives a second view that is less sensitive to a single large release.

## Limits

- The approximation weakens for deep dividends, high rates, or strikes far from spot.
- The comparison uses close to close moves. Moves measured from the first trade after the release can differ.
- After the release, the events skill measures the abnormal reaction against the market, which separates the stock specific move from the market's.
