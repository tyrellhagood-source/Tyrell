# MoneyOps AI v1.0

A five-agent sports-betting **research** system. It reads live odds, checks them against an independent source, prices them with documented statistical models, and puts every candidate through a weighted consensus with hard gates. Picks are tracked against a **$25 virtual (paper) bankroll**, and accuracy is measured honestly over time.

> **What this is not:** it does not guarantee wins, does not promise an 80–90% hit rate, and never touches real money. There is no bank, payment or sportsbook integration anywhere in the code. When nothing passes every check, the system says **`NO QUALIFYING BETS FOUND.`**, and that is the normal, expected outcome on many days.

---

## 1. Quick start

```bash
cd moneyops-ai
python -m venv .venv && source .venv/bin/activate      # Python 3.11+ (tested on 3.13)
pip install -r requirements.txt
cp .env.example .env                                     # then put your ODDS_API_KEY in .env
python scripts/initialize_database.py                    # creates SQLite DB + $25 paper account
python scripts/check_providers.py                        # 1 FREE Odds API call + 1 ESPN call (+ AllSportsAPI, nba_api, nflreadpy)
python scripts/probe_allsports.py                        # optional: confirm AllSportsAPI field names with your key
streamlit run app.py                                     # dashboard  -> http://localhost:8501
uvicorn api.server:app --port 8000                       # optional REST API -> http://localhost:8000/docs
python scripts/run_daily_analysis.py                     # CLI daily run (add --loop to keep refreshing)
python -m pytest                                         # 130 automated tests
```

Run every command from inside `moneyops-ai/`.

---

## 2. Data providers: research and recommendation

Researched October 2026. Prices change, so confirm them on each provider's site before you subscribe.

| Provider | What it gives | Cost / limits | Verdict |
|---|---|---|---|
| **The Odds API (v4)** | Live odds from ~40–80 books (US/UK/EU/AU regions): moneyline (`h2h`), `spreads`, `totals`, player props (per-event endpoint), scores for the last 3 days. Historical odds snapshots on paid plans. | **Free: 500 credits/month.** Paid tiers reported at about $30 (20K), $59 (100K), $119 (5M), $249 (15M) per month. Cost per call = markets × regions; `/sports` and `/events` are free; historical calls cost 10×. | **Primary odds source (implemented).** |
| **ESPN public site API** | Schedules and game status, final scores, a full season of results, injuries. No key. | Free but **unofficial and undocumented**: no SLA, may change without notice, not licensed for commercial redistribution. | **Independent schedule/result/history source (implemented)** for personal research. |
| **AllSportsAPI** ([profile](https://github.com/api-evangelist/all-sports-api)) | Fixtures, live scores, results, standings, odds (`met=Odds`), probabilities (`met=Probabilities`) for football, basketball, tennis, cricket. Hockey, baseball and American football are sold as add-ons. Key passed as the `APIkey` query parameter. | Paid plans with a 14-day free trial (pricing not public in the profile). | **Optional second source (implemented):** independent schedule and results, extra bookmakers, outside probability opinion, history fallback. |
| **balldontlie** ([balldontlie.io](https://www.balldontlie.io)) | Teams, players and **games** (date, status, final score) for NBA, NFL, MLB, NCAAB, EPL and more. Key sent in the `Authorization` header. | **Free plan:** games only, 5 requests/min per sport. All-Star ($9.99/mo per sport) adds player stats and injuries; GOAT ($39.99) adds odds and props. | **Optional third source (implemented, free plan):** independent schedule and status check, final scores for settlement, backup season results. |
| **nba_api** ([swar/nba_api](https://github.com/swar/nba_api), v1.11.4, Feb 2026) | NBA player game logs from stats.nba.com. One `LeagueGameLog` request returns every player-game line of a season: points, rebounds, assists, threes, minutes, matchup. | Free, no key. Maintained in 2026. An unofficial client of the NBA's public stats site, which can block cloud IPs. | **NBA player-prop history and settlement (implemented).** |
| **nflreadpy** ([nflverse/nflreadpy](https://github.com/nflverse/nflreadpy), updated Aug 2026) | NFL weekly player stats (passing, rushing, receiving) and schedules with final scores, from nflverse data releases on GitHub. | Free, no key. Maintained in 2026. Stats are published after games and can lag about a day. | **NFL player-prop history and settlement (implemented).** |
| SportsDataIO | Licensed stats, injuries, player game logs, odds. | Free trial uses scrambled data. Odds plans about $99+/mo per its developer page; production pricing is sales-gated. | Best upgrade path for player props and commercial use. Not connected. |
| Sportradar | Official league data, odds comparison. | Trial only; quote-based (third parties estimate $500+/mo). | Enterprise option. Not connected. |

**Recommended starting setup:** The Odds API free tier plus ESPN.

- **Cost: $0/month.** One analysis run of 3 markets × 1 region across ~8 in-season sports uses about 24 credits, so the free tier covers **~15–20 full runs per month**.
- The free tier is enough for testing. For daily use or hourly refreshes, plan on The Odds API 20K tier (about $30/mo).
- If you need licensed data or player game logs, add SportsDataIO by subclassing `data/provider_base.ProviderBase`.

**Required subscriptions and estimated costs**

| Need | Subscription | Est. cost |
|---|---|---|
| Live odds (required) | The Odds API free key | $0 (500 credits/mo) |
| Daily scans across all sports + refreshes | The Odds API 20K | ~$30/mo |
| Player props at volume | The Odds API 100K+ | ~$59+/mo |
| Player stats for prop research / commercial-grade data | SportsDataIO or Sportradar | ~$99+/mo or a quote |
| Historical odds (true ATS/ROI backtests) | Odds API paid plan (historical endpoint) | included in paid tiers, 10× credits per call |

Sources: [Odds API pricing comparison (OddsPapi)](https://oddspapi.io/blog/?p=2498), [Odds API free-tier notes](https://oddspapi.io/blog/?p=2967), [The-Odds-API review](https://sportsapis.dev/apis/the-odds-api), [SportsDataIO developers](https://sportsdata.io/developers/getting-started), [SportsDataIO review](https://sportsapis.dev/apis/sportsdata-io), [Sportradar review](https://sportsapis.dev/apis/sportradar). Several of these are written by competitors, so treat their figures as estimates.

---

## 3. API configuration

Settings live in `.env`; `.env.example` lists them all. The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `ODDS_API_KEY` | (none) | **Required.** Read only from the environment; it is never logged or shown, and `apiKey=` is redacted from every log, error, cache entry and audit row. |
| `ODDS_API_REGIONS` / `ODDS_API_MARKETS` | `us` / `h2h,spreads,totals` | Credits per sport = markets × regions. |
| `MAX_CREDITS_PER_RUN` / `MIN_CREDITS_RESERVE` | 60 / 25 | Hard budget per run. A reserve is never spent. Sports skipped for budget reasons show as coverage gaps. |
| `ENABLED_SPORTS` | all supported | NFL, NCAAF, NBA, NCAAB, NCAAW, WNBA, MLB, NHL, EPL, MLS, UCL, LALIGA, BUNDESLIGA, SERIEA, LIGUE1, MMA |
| `SCAN_TENNIS` / `SCAN_OTHER_SPORTS` | false | Tennis tournaments and other active sports. Odds only; no model, so these can never be QUALIFIED. |
| `INCLUDE_PLAYER_PROPS` | false | Fetches props for up to `MAX_PROP_EVENTS_PER_RUN` games (credit-expensive). |
| `USE_ODDS_API_SCORES` | false | Extra settlement source; costs 2 credits per sport. |
| `ALLSPORTS_API_KEY` | (none) | Enables AllSportsAPI. It can also be the only odds source if `ODDS_API_KEY` is empty. |
| `ALLSPORTS_USE_ODDS` / `ALLSPORTS_USE_PROBABILITIES` / `ALLSPORTS_MAX_REQUESTS_PER_RUN` | true / true / 80 | What to pull, and the per-run request budget. |
| `ALLSPORTS_TIMEZONE` | UTC | Timezone sent with fixture requests and used to read their dates and times. |
| `ALLSPORTS_PATH_OVERRIDES` | (none) | Paths for add-on sports, e.g. `NHL:hockey,MLB:baseball`. |
| `BALLDONTLIE_API_KEY` | (none) | Enables balldontlie (free plan). |
| `BALLDONTLIE_REQUESTS_PER_MINUTE` / `BALLDONTLIE_MAX_REQUESTS_PER_RUN` / `BALLDONTLIE_PATHS` | 5 / 20 / (none) | Throttle for the free plan, per-run budget, and extra sport paths, e.g. `NHL:nhl/v1`. |
| `NBA_API_ENABLED` / `NFLREADPY_ENABLED` | true / true | Player game logs for NBA and NFL props (current and previous season). Used only when props are found. |
| `PLAYER_STATS_GAMES_PER_PLAYER` | 30 | Most recent games kept per player. Each season file is cached; past seasons are cached for 30 days. |
| Risk: `MAX_STAKE_PCT`, `KELLY_FRACTION`, `MAX_OPEN_EXPOSURE_PCT`, `MAX_DRAWDOWN_HALT_PCT`, `MIN_EV`, `MIN_BOOKS`, `MAX_MODEL_MARKET_GAP` | 1%, 0.25, 5%, 25%, +2%, 3, 10 pp | See the Risk Center page. |

Responses are cached in SQLite (`api_cache`) with short TTLs for odds (5 min) and long TTLs for finished-game history. HTTP 429 and 5xx errors are retried with backoff. 401/403 errors fail immediately. Quota headers (`x-requests-remaining/used/last`) are stored and shown on **Data Sources**.

---

## 4. Architecture

```
moneyops-ai/
  app.py                 Streamlit entry (10 pages)        api/server.py   FastAPI backend
  config.py              Settings from env                  services/       orchestrator, settlement, queries
  agents/                research, verification, statistics, probability, chief_auditor (+ base.py)
  data/                  provider_base (HTTP/retry/cache/status), odds_provider, sports_data_provider (ESPN),
                         allsports_provider (AllSportsAPI), balldontlie_provider, player_stats_provider (nba_api, nflreadpy),
                         schedules, injuries, results, sports_registry, cache, schemas
  models/                expected_value, probability (team-rating models), props (player model), parlay, calibration,
                         backtesting, performance, stats
  risk/                  bankroll ($25 paper ledger), position_sizing, risk_limits
  database/              SQLAlchemy models (integrity rules), repository, migrations/
  dashboard/             charts, components, pages/01..10
  scripts/               initialize_database, run_daily_analysis, check_providers, probe_allsports
  tests/                 130 tests + hand-written fixtures (tests only)
```

### Daily workflow (`services/analysis.py`)

1. Discover in-season sports (free `/sports` call).
2. Pull odds per sport within the credit budget.
3. Pull ~400 days of results from ESPN. Fit each sport's model **as of now** (only earlier games). Run the daily walk-forward backtest.
4. Fetch injuries. Agent 2 independently fetches the schedule.
5. Agents 1→4 run on every candidate (every market × selection × line).
6. Agent 5 audits each candidate, then the portfolio (duplicates, one pick per game, exposure cap).
7. Build parlays from qualified legs.
8. Persist everything: every candidate including rejected ones, odds snapshots (used for closing-line value), audit rows and coverage.

Settlement runs first on each daily run (`run_daily_analysis.py`) and grades finished games.

---

## 5. The five agents

Each agent writes its own structured `AgentReport` (verdict, score, summary, reasons, flags, data). No agent edits another's report, so disagreements stay visible in the **Five-Bot Control Room**.

| Agent | Job | Verdicts |
|---|---|---|
| **1 Live Data Researcher** | Every book's price with its own timestamp. Best *fresh* price. Multi-book **no-vig consensus** (median of per-book fair probabilities). Conflicts between books, outlier prices, missing history or injuries. Never fills gaps. | COMPLETE / INCOMPLETE / STALE |
| **2 Independent Fact-Checker** | Checks that the game exists on ESPN, AllSportsAPI and/or balldontlie with the same teams and orientation, that start times agree, and that the game is not final or postponed. When both sources are present they must not contradict each other. When two odds *providers* quote the same line, their consensus must agree within 4 pp. For props: the player's latest game log is with one of the two teams, he has played recently, and he is not listed Out or Doubtful. Also checks freshness, ≥ `MIN_BOOKS` books quoting the full market at that line, best price not an outlier, history dated before the game and from the current season, injuries retrieved. | VERIFIED / REJECTED / INSUFFICIENT_DATA |
| **3 Historical Statistician** | For props: the player's last-5/10/20 hit rate vs this line, home/away, vs this opponent, minutes trend; DNPs excluded. For team markets: replays *this exact bet* (same side and line) over each team's games before kickoff: last 5/10/20, season (detected from the off-season gap), full lookback, home/away role, vs comparable opponents (rating tertile), rest and back-to-back, strength of schedule, streaks, W-L-P counts, **Wilson 95% CI**. SUPPORTS only if the CI lower bound is above break-even. | SUPPORTS / NEUTRAL / CONTRADICTS / INSUFFICIENT_DATA |
| **4 Probability & Risk Engineer** | Model probability, market-implied and no-vig probability, a **blend** `p = 0.35·model + 0.65·market`, a bootstrap 5–95% interval, push probability, EV and conservative EV, ROI, fractional-Kelly stake capped at 1% of bankroll, profit and max loss. | PASS / FAIL / INSUFFICIENT_DATA |
| **5 Chief Auditor** | Recomputes the no-vig consensus, implied probability and EV **independently** with exact `Fraction` arithmetic; a mismatch is a veto. Re-checks freshness. **Vetoes** a model–market gap over 10 pp (usually missing injury or lineup information, not a market error). Lists disagreements, computes the weighted score, enforces the mandatory gates, removes duplicates and correlated same-game picks, enforces the exposure cap, ranks singles and parlays separately. | QUALIFIED / NEEDS_RESEARCH / REJECTED |

**Weighted consensus score.** This is not a majority vote:
data quality 0.12, verification 0.15, statistical evidence 0.12, model quality (backtest Brier skill) 0.10, expected value 0.20, market price (edge vs consensus × book depth) 0.10, historical validation (calibration error) 0.06, risk (interval width) 0.10, correlation 0.05.

**Mandatory gates for QUALIFIED:** fresh, complete price; Agent 2 VERIFIED; Agent 3 SUPPORTS or NEUTRAL; Agent 4 PASS; no auditor veto; pre-game; team market. Five agents agreeing is **not** proof. They share data sources, which is why the auditor recalculates independently and why calibration is tracked.

---

## 6. The math

- **American → decimal:** `+A → 1 + A/100`, `−A → 1 + 100/A`. Implied probability = `1/D`.
- **Vig removal:** multiplicative (`q_i / Σq`), with the power method also available. The consensus is the median across books.
- **EV per $1:** `p·D − 1`. With pushes: `p_win·(D−1) − p_loss`. An 80% favourite at −500 (D = 1.20) has EV = −4%. **High probability is not the same as value**, and Agent 4 flags this case.
- **Models (`models/probability.py`):**
  - **Gaussian** for basketball and football: ridge `points = μ + off[team] + def[opp] + HFA` with recency weights (120-day half-life) and a bivariate-normal score grid. Whole-number lines get push probabilities.
  - **Poisson** for MLB, NHL and soccer: Poisson GLM. Soccer gets a 3-way moneyline. NHL and MLB ties are split (overtime/shootout approximation).
  - Every market, plus same-game joint probabilities, is summed off the same score grid.
- **Parlays:**
  - Different games with no shared team: product of leg probabilities (labelled "independent").
  - Same game: joint probability from the model grid (labelled "modeled_joint").
  - Anything else: **UNCERTAIN**. Only Fréchet bounds `[max(0, Σp−(n−1)), min p]` are shown, and the parlay is never recommended.
  - Every parlay shows combined decimal and American odds, break-even probability, payout, profit, max loss, warnings and failure reasons.
- **Player props (`models/props.py`):** recency-weighted normal distribution of the player's last 20 appearances (weights halve every 10 games), discretised so whole-number lines get a push probability. Bootstrap interval. Blended 35/65 with the market's no-vig consensus, exactly like team markets. No adjustment for opponent, pace or teammates' injuries.
- **AllSportsAPI probabilities** are shown as an *outside opinion* and are never blended into the estimate. A gap of more than 10 pp is listed as a disagreement.
- **Staking:** `min(¼-Kelly, 1% of current balance)`, rounded down to the cent. This is $0.25 at $25. Stakes below $0.10 are dropped, and you are told that real books usually require about $1. Stakes are never raised after losses.

---

## 7. $25 virtual bankroll

A recommendation becomes a **Bet slip (PROPOSED)**, which you **Approve**. Approval creates an **OPEN** paper wager and debits the stake in the ledger. Settlement then writes the result **once**:

- **win** → payout = stake × D
- **push / void** → stake refunded
- **loss** → nothing returned

Parlay legs that push are dropped and the payout is recomputed from the remaining legs.

- Proposed or declined slips never touch the balance. An unplaced recommendation never counts as a win or loss.
- Approval is refused after kickoff, when the odds are stale (re-run analysis first), above the 1% cap, above 5% open exposure, or during a 25% drawdown halt.
- The ledger is **append-only** and accepted wagers are **immutable**: ORM hooks raise on any edit or delete, and settlement fields are written once. Recommendations, including losing and rejected ones, can never be deleted.
- Settlement uses ESPN, AllSportsAPI and balldontlie final scores, optionally plus Odds API scores. If any source says the game is not final, the system waits. If finals conflict, the wager stays open and the conflict is reported. Soccer settles only on the **90-minute** result: an ESPN score after extra time is not used, and AllSportsAPI's `event_ft_result` is required. **Props settle from official game logs** (nba_api for NBA, nflverse for NFL); a player who did not play means void. Manual settlement (labelled MANUAL, must cite a source) remains for anything else.
- **Reset** requires typing `RESET PAPER ACCOUNT`. It archives the account (history kept) and opens a new $25 account.
- CSV export is available for wagers, the ledger and all recommendations.

---

## 8. Season tracking and metrics

The Historical Results and Probability pages break performance down by each agent (picks it endorsed), consensus picks, singles and parlays, sport, league, market, probability bucket, odds bucket and model version.

| Metric | Definition |
|---|---|
| Win rate | W/(W+L). Pushes and voids are excluded. |
| Units | Flat 1u per pick |
| Yield | units/(W+L+P) |
| ROI | profit/staked, using real paper stakes |
| Max drawdown | Peak-to-trough fall in cumulative units |
| Streaks | Longest and current runs |
| Brier, log loss, calibration table, ECE | Calibration: do events given ~80% happen ~80% of the time? |
| **CLV** | taken odds / closing odds − 1. "Closing" is the last snapshot before start, and is used only if it was taken within 60 minutes of kickoff. |

**Backtests** (`models/backtesting.py`) are chronological:

- First 60% of games: training. Next 20%: validation, used to choose the ridge or L2 strength. Final 20%: test, with a **weekly walk-forward refit** that only predicts future games.
- They are stored separately (`backtest_runs`) and labelled **HYPOTHETICAL**. They are never mixed with paper results.
- They measure probability quality only. Without paid historical odds there is no profitability backtest.

---

## 9. Dashboard pages

1. **Today's Sports:** run button, coverage gaps, all games with start time, live flag, best moneylines, book count and quote age.
2. **Top Consensus Picks:** every field from the spec, plus why the pick could win or lose, agent verdicts, and a "create paper slip" button.
3. **Parlay Lab:** system parlays (qualified ones first, with rejection reasons) plus a custom builder.
4. **Five-Bot Control Room:** each agent's verdict, reasons, flags and raw output; the book-by-book price table; verification checks; score contributions; gates; recalculations.
5. **Probability Analytics:** model vs market, EV distribution, probability intervals vs break-even, backtest calibration, live calibration.
6. **Historical Results:** every breakdown in §8 and a cumulative-units chart.
7. **Virtual Bankroll:** balance, slips, open and settled wagers, ledger, settlement, reset, CSV export.
8. **Risk Center:** drawdown, exposure, limits, correlated positions, rejected picks with reasons.
9. **Data Sources:** provider status, quota, errors, coverage, connectivity test.
10. **Audit Log:** every agent decision and orchestrator event, filterable.

---

## 10. Testing

`python -m pytest` runs 130 tests covering:

- **Math:** odds conversion, implied and vig-adjusted probability, EV (including pushes and the "80% can be −EV" case), Kelly, Wilson intervals, streaks.
- **Models:** recovery of known synthetic strengths, leakage guard, push handling, soccer 3-way, NHL ties, joint probabilities, bootstrap.
- **Parlays:** independent, correlated and uncertain parlays; duplicates and conflicts.
- **Evaluation:** calibration and ECE, time-ordered splits, walk-forward backtest, metric definitions.
- **Bankroll:** win/loss/push/void, the 1% cap, exposure and drawdown halts, immutability, reset, CSV export.
- **Settlement:** grading rules; conflicts, postponements and unavailable results.
- **Data handling:** parser robustness, stale-odds rejection, missing-data handling, budget and reserve enforcement, retries, secret redaction.
- **Database and integration:** persistence, closing-line lookup, a full five-agent pipeline run, and the API.
- **AllSportsAPI:** parsers including the soccer 90-minute rule, 3-way vs 2-way markets, missing timestamps, unique league resolution, the request budget, redaction, cross-provider price agreement and disagreement, and result conflicts.
- **balldontlie:** game parsing (scheduled, live, final, postponed, date-only), header auth, pagination, throttling, budget, date-only verification, three-way result cross-check.
- **Player stats and props:** nba_api and nflverse parsing, no leakage, traded players, DNP and not-yet-published handling, prop model coherence, a full prop pipeline, the injury veto, and prop settlement.

The integration tests use **offline test doubles** with hand-written fixtures (`tests/fixtures/`) and synthetic games. These exist only in `tests/`; the application never falls back to them. **Passing tests says nothing about profitability.**

---

## 11. Implemented vs. unfinished

**Implemented and tested offline**

- [x] Five independent agents with structured outputs, visible disagreements and an auditor veto
- [x] The Odds API integration (odds, props, scores, quota and budget guard) and ESPN integration (schedule, history, results, injuries)
- [x] Gaussian and Poisson team models, bootstrap uncertainty, market blend, EV, Kelly-capped staking
- [x] Parlay engine with correlation handling; separate ranking for singles and parlays
- [x] $25 paper bankroll, immutable ledger, settlement, reset, CSV export
- [x] Full-season tracking, calibration, CLV, hypothetical walk-forward backtests
- [x] 10-page Streamlit dashboard, FastAPI backend, daily CLI with `--loop`
- [x] AllSportsAPI provider: second schedule and result source, extra books, cross-provider price check, probabilities, history fallback
- [x] balldontlie provider (free plan): third schedule/status check, third final-score source, backup season results, header auth, 5/min throttle
- [x] Player-stats provider: nba_api (NBA) and nflreadpy (NFL) game logs for prop research and settlement
- [x] 130 passing tests

**Not yet verified**

- [ ] **No live API request has been made yet.** The build environment's network policy blocked `api.the-odds-api.com`, `site.api.espn.com`, `apiv2.allsportsapi.com` `stats.nba.com`, `api.balldontlie.io` and GitHub's nflverse data releases, and no API key was available. The AllSportsAPI **odds and probability field names** (`odd_bookmakers`, `odd_epoch`, `odd_1/x/2`, `o+N/u+N`, `event_HW/D/AW`) could not be confirmed from public docs. Run `scripts/probe_allsports.py`; until fields match, those calls simply return nothing (fail-closed). stats.nba.com is known to block some cloud IPs; failures appear as coverage notes. The parsers follow the documented formats and are tested against hand-written fixtures, **but they have not been validated against a real response.**
  - Run `python scripts/check_providers.py` and then one analysis on your machine.
  - ESPN's undocumented format is the most likely place for a parsing adjustment.

**Not implemented (known gaps)**

- [ ] **Props beyond NBA/NFL** (NHL, MLB, WNBA, college) have no game-log mapping yet. AllSportsAPI Asian-handicap odds are skipped because the sign convention is unverified.
- [ ] **Injury-adjusted models.** Injuries are retrieved and displayed but the models do not adjust for them. The 10 pp model–market veto is the safeguard.
- [ ] **Live in-game models.** Live odds are displayed and labelled; live candidates are never QUALIFIED.
- [ ] **Tennis and MMA.** No independent schedule source or model, so odds only, shown as coverage gaps.
- [ ] **Historical ATS/O-U records and ROI backtests.** These need paid historical odds. Agent 3 replays today's line instead and says so.
- [ ] **Model refinements:** NFL key numbers (normal approximation), Dixon–Coles low-score correction for soccer, Asian quarter lines (skipped), extra-inning/overtime scoring (approximated).
- [ ] **Scheduler daemon.** `run_daily_analysis.py --loop` or cron is the scheduler. There is no Celery or APScheduler.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| "ODDS_API_KEY is not set" | Put the key in `moneyops-ai/.env` and restart. |
| `ProviderAuthError HTTP 401` | Wrong or expired key, or the plan does not cover the endpoint. |
| "Skipped: request would cost N credits" | Raise `MAX_CREDITS_PER_RUN`, reduce `ENABLED_SPORTS` or markets, or upgrade your plan. |
| ESPN "network error" / 403 | Your network or proxy blocks `site.api.espn.com`. Agent 2 then returns INSUFFICIENT_DATA and nothing qualifies, by design. |
| History shows 0 games or team names are unmatched | ESPN and odds names differ. Extend `_ALIASES` in `utils/teams.py`. |
| "odds on this recommendation are stale" when approving | Re-run analysis. Approval needs odds younger than `STALE_ODDS_MINUTES_PREGAME`. |
| Everything is REJECTED / NEEDS_RESEARCH | Expected on many days. Check the reasons on the Risk Center and Control Room pages. |

## 13. Safety

- Paper trading by default and by construction: there is no code path to a sportsbook, bank or payment system.
- The app never asks for sportsbook or bank credentials and never automates logins.
- Provider authentication and terms are respected: keys come from the environment, there is a credit budget, and caching uses provider TTLs.
- Sports betting is legal only in some jurisdictions. Use this as a research tool.
