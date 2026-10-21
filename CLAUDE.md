# Tyrell

This repo holds one project: `day-trader-skill/`, a pack of 11 Claude skills for
day trading, imported from https://github.com/Jakeschincariol/day-trader-skill (MIT).

## Layout

Each skill is a folder in `day-trader-skill/` with a `SKILL.md` (the instructions
Claude follows) and one standard-library Python script:

| skill | script | job |
| --- | --- | --- |
| trader-scanner | scan.py | finds setups (opening-range breakouts, flags, oversold) |
| trader-news | news.py | headlines, 8-Ks, moves for the watchlist |
| trader-entry | entry.py | calls the entry bar by bar |
| trader-exit | exit.py | stop and target, writes `trader/plan.json` |
| trader-timemachine | timemachine.py | 10,000 replays + history of the setup |
| trader-budget | size.py | position size, writes `trader/orders.json` |
| trader-risk | check.py | PASS / BLOCK per order |
| trader-broker | ticket.py | order tickets, never places without a yes |
| trader-alerts | alerts.py | price, stop, target and time alerts |
| trader-analyst | facts.py | earnings from SEC EDGAR XBRL |
| trader-pelosi | ptr.py | Congress trade disclosures |

The skills hand off through files in a `trader/` folder:
scanner -> news -> entry -> exit -> timemachine -> budget -> risk -> broker, with alerts while a trade is open.

## Rules

- Standard library only, Python 3.8+. No pip dependencies.
- Long side only; nothing places an order without risk passing and the user saying yes. Paper trading is the default. Keep it that way.
- Tests run offline on fixtures in `day-trader-skill/tests/fixtures/`. A change to a script needs a test.
- `tests/test_repo.py` checks the folder set, SKILL.md frontmatter and the safety block; update it when the skill set changes.

## Commands

```bash
cd day-trader-skill
python3 -m unittest discover -s tests          # all 315 tests, ~15s
python3 -m unittest tests.test_risk -v         # one skill
```
