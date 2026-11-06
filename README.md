# Hedge Fund Skills

**Six agent skills that teach an AI agent the core playbooks of professional quants.**

Each skill is a folder that an agent discovers and loads on demand. It contains a precise trigger, an ordered procedure, Python scripts that compute every number, and the checks that the output must pass before anyone reads it. Drop the folders into any agent that supports the open [Agent Skills](https://agentskills.io) format.

Companion code to the paper _Hedge Fund Skills: Teaching an AI Agent the Playbooks of Professional Quants_ ([paper/](paper/)).

| Skill                                      | Answers                                          | Method                                                          | Scripts                                  |
| ------------------------------------------ | ------------------------------------------------ | --------------------------------------------------------------- | ---------------------------------------- |
| [`risk`](skills/risk/SKILL.md)             | Where does the risk in this portfolio come from? | Factor model, Euler risk contributions                          | `decompose.py`                           |
| [`events`](skills/events/SKILL.md)         | How did the market price this news?              | Market model event study, CAR and t                             | `fit_market_model.py`, `car.py`          |
| [`sizing`](skills/sizing/SKILL.md)         | How large should this position be?               | Fractional Kelly with a volatility target                       | `size.py`                                |
| [`regimes`](skills/regimes/SKILL.md)       | What kind of market are we in?                   | Relative features, Gaussian hidden Markov model, forward filter | `regimes.py`                             |
| [`validation`](skills/validation/SKILL.md) | Is this backtest result real?                    | Probabilistic and deflated Sharpe ratio                         | `psr_dsr.py`                             |
| [`earnings`](skills/earnings/SKILL.md)     | What is expected from this earnings release?     | Straddle implied move against past release day moves            | `implied_move.py`, `historical_moves.py` |

**Disclaimer:** For education and research only. Not investment advice, and the author holds no financial licenses. Outputs are not recommendations; use at your own risk.

## Why skills

An agent asked for a factor risk report can reason one out from its training, and it may compute it one way on Monday and another way on Tuesday. A skill fixes the method: the model understands the request, picks the skill, and explains the result, while every number travels through code.

```
skills/events/
├── SKILL.md                    trigger, data contract, procedure, checks, output
├── scripts/
│   ├── fit_market_model.py     estimation window, alpha, beta, residual volatility
│   └── car.py                  abnormal returns, CAR, t statistic
└── references/
    └── methodology.md          the mathematics, read only when a step needs it
```

The agent keeps only each skill's name and description in view (about a hundred tokens each), reads the full `SKILL.md` when a request matches, and opens the scripts and references only when a step calls for them.

## Install

The scripts need Python 3.9 or later with numpy and pandas:

```bash
pip install -r requirements.txt
```

**Claude Code.** Copy the skill folders into your personal skills directory, or into `.claude/skills/` inside a project to share them with your team:

```bash
mkdir -p ~/.claude/skills && cp -r skills/* ~/.claude/skills/
```

**Claude apps.** Zip each skill folder, then upload each zip under Customize > Skills > Create skill > Upload a skill. Code execution and file creation must be enabled in Settings > Capabilities ([guide](https://support.claude.com/en/articles/12512180-use-skills-in-claude)).

```bash
cd skills && for s in */; do zip -r "../${s%/}.zip" "$s" -x "*/__pycache__/*"; done
```

**Other agents.** Any agent that supports the Agent Skills format can load these folders from its skills directory; see its documentation for the location.

## Usage

Once installed, ask the agent in plain language, for example _"Where does the risk in this portfolio come from?"_ with your weights and risk model attached. The agent selects the skill, follows its procedure, runs the scripts, and explains the result.

The scripts also run directly from the command line:

```bash
# risk
python skills/risk/scripts/decompose.py --weights weights.csv \
  --exposures exposures.csv --factor-cov factor_cov.csv --specific-var specific_var.csv

# events
python skills/events/scripts/fit_market_model.py --prices prices.csv --asset XYZ --market INDEX \
  --event-date 2025-05-01 --out fit.json
python skills/events/scripts/car.py --fit fit.json --window -1 1

# sizing
python skills/sizing/scripts/size.py --mu 0.06 --sigma 0.25 --kelly-fraction 0.5 --target-vol 0.10

# regimes
python skills/regimes/scripts/regimes.py --prices prices.csv --asset INDEX --states 3 \
  --train-end 2019-12-31 --exposures 1.0,0.6,0.2

# validation
python skills/validation/scripts/psr_dsr.py --returns returns.csv --column STRAT --trial-returns all_trials.csv

# earnings
python skills/earnings/scripts/implied_move.py --spot 182.40 --chain chain.csv --reaction-date 2025-07-31
python skills/earnings/scripts/historical_moves.py --prices prices.csv --asset XYZ --events events.csv --implied 0.0655
```

Every script answers `--help`. Each `SKILL.md` describes the exact input files the skill expects.

## Input files

| Skill      | Files                                                                                                                                                                      |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| risk       | `weights.csv` (asset, weight) plus a factor model (`exposures.csv`, `factor_cov.csv`, `specific_var.csv`), a covariance (`cov.csv`), or historical returns (`returns.csv`) |
| events     | `prices.csv`: date, then total return price levels for the security and the market index                                                                                   |
| sizing     | expected excess return and volatility as numbers, or a returns file to estimate volatility                                                                                 |
| regimes    | `prices.csv`: date, then daily price levels; extra columns feed the correlation feature                                                                                    |
| validation | returns of the selected strategy, plus returns or Sharpe ratios of every trial, or the trial count and their Sharpe variance                                               |
| earnings   | spot with call and put prices, or `chain.csv` (expiry, strike, type, bid, ask); for history, `prices.csv` and `events.csv` (date, timing)                                  |

## Built in checks

Every script verifies its own output before the agent reports it.

| Skill      | Checks                                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| risk       | Covariance is positive semidefinite; contributions sum exactly to total volatility                                       |
| events     | Estimation window is complete and never overlaps the event window                                                        |
| sizing     | Kelly fraction satisfies its closed form; weight respects the volatility target and maximum weight                       |
| regimes    | Probabilities sum to one; probabilities recomputed from data cut at a past date match exactly, so no future data is used |
| validation | Enough observations; the trial count covers every trial supplied                                                         |
| earnings   | Quotes are valid; strike is near spot; expiry is close to the release; enough past releases were measured                |

## Output contract

Every script prints one JSON object with the same shape, so skills feed one another:

```json
{
  "skill": "validation",
  "version": "1.0",
  "inputs": { "...": "what was used" },
  "results": { "...": "the numbers" },
  "checks": [
    { "name": "enough_observations", "passed": true, "detail": "T = 1000" }
  ],
  "passed": true
}
```

Exit codes: `0` all checks pass, `1` a check failed (the JSON says which), `2` the inputs broke the data contract (the message on stderr says why). Every `SKILL.md` tells the agent to stop and report when `passed` is false.

## Why validation matters

In a simulation of 50 strategy variations with no true edge, the best one showed an annualized Sharpe ratio of 1.49 and a probabilistic Sharpe ratio of 0.998, which looks convincing on its own. The validation skill accounts for the 50 trials behind that selection: the bar rises to an annualized Sharpe of 1.29, and the deflated Sharpe ratio falls to 0.66, well short of 0.95. The result was the luck of the search, and the skill says so before any capital is committed.

## Writing your own skill

1. **Capture the procedure.** Have the person who does the task walk through it once, recording every choice: data, windows, adjustments, thresholds, and presentation.
2. **Write the trigger.** Draft the description from the questions people actually ask, in their own words.
3. **Move the arithmetic into scripts.** Each calculation becomes a small script with a clear input and output that runs without any model.
4. **Add checks.** Write the invariants the output must satisfy, and confirm the scripts on inputs with known answers.
5. **Test the trigger, then ship.** Send realistic requests to the agent, confirm it selects the skill, give it an owner and a version, and add it to the library.

Each skill here follows that pattern; copy one as a template.

## Repository layout

```
hedge-fund-skills/
├── skills/
│   ├── risk/          factor risk decomposition
│   ├── events/        event study
│   ├── sizing/        position sizing
│   ├── regimes/       regime detection
│   ├── validation/    strategy validation
│   └── earnings/      earnings preparation
├── paper/             the companion paper
├── requirements.txt
└── LICENSE
```

## References

- Agent Skills specification, [agentskills.io/specification](https://agentskills.io/specification)
- R. C. Grinold and R. N. Kahn, _Active Portfolio Management_, McGraw Hill, 2000.
- A. C. MacKinlay, "Event studies in economics and finance," _Journal of Economic Literature_, 1997.
- J. L. Kelly, "A new interpretation of information rate," _Bell System Technical Journal_, 1956.
- A. Moreira and T. Muir, "Volatility managed portfolios," _Journal of Finance_, 2017.
- J. D. Hamilton, "A new approach to the economic analysis of nonstationary time series and the business cycle," _Econometrica_, 1989.
- D. H. Bailey and M. López de Prado, "The Sharpe ratio efficient frontier," _Journal of Risk_, 2012.
- D. H. Bailey and M. López de Prado, "The deflated Sharpe ratio," _Journal of Portfolio Management_, 2014.

## License

[MIT](LICENSE)
