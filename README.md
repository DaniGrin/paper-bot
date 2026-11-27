# paper-bot
DOGE 4h step-trail paper-trading bot, runs free on GitHub Actions.

## Fix log

**2026-11-27: closed bars only, and all of them.** The live loop only handled the **latest** closed 4h bar. GitHub runs the hourly schedule just a few times a day, so about 23% of bars were skipped (44% in November): no stop check and no entry signal on them. It now processes every closed bar since the last run.

Trading rules are unchanged. Results recorded **before** this date were produced with the defect and have not been rewritten; treat 2026-11-27 as the start of the clean data.
