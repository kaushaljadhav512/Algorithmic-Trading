# Algorithmic Trading Bot: WnCC Codewars

🥈 **First runner-up** (2nd place). See [`Result.pdf`](Result.pdf).

Our team's entry (**Stonks**: Kaushal Jadhav, Shrey Ganatra, Jahnvi Rohela) to *Codewars*, an algorithmic-trading competition held by the Web and Coding Club (WnCC) at IIT Bombay.

Each team writes a Python `Bot` that trades in a simulated stock market. The market has 20 securities, a fixed interest rate and changing market regimes. Bots are scored on final net worth across a range of market scenarios: bullish, bearish or neutral; stable or volatile; and low or high interest.

## Strategy

The bot is in [`Windows/bot.py`](Windows/bot.py). The same file is in `Mac-Intel/` and `Mac-M1/`.

It is a long-only **mean-reversion** strategy based on an exponential moving average:

1. **Warm-up (first 10 steps):** record prices and buy 900 shares of every security the balance can cover.
2. **Signal:** for each security, compute an EMA of recent prices (α = 0.2) over a rolling window of up to 100 steps, and the ratio `w = EMA / current price`.
3. **Trade:** when `w > 1`, the price is below its recent average, so buy `900 × w` shares. A deeper dip means a bigger position. Orders are placed only if the cash balance covers them.

`submission.py` is an unfinished experiment that tried to forecast prices with an ARIMA(2, 1, 2) model. It runs on synthetic data and is not wired into the trading loop.

## Results

Team Stonks finished as **first runner-up**. [`Result.pdf`](Result.pdf) has the official leaderboard: final scores for every team across all 12 market scenarios.

## Repository layout

| Path | Contents |
|---|---|
| `Windows/`, `Ubuntu/`, `Mac-Intel/`, `Mac-M1/` | Per-OS kit: market engine binary (`wnccATfifteen`), `main.py` harness and `bot.py` |
| `GameEngine_Code/main.cpp` | C++ source of the market simulator: price process, correlations, interest, order matching |
| `submission.py` | ARIMA forecasting experiment |
| `Result.pdf` | Competition results |

## Running locally

Requires Python 3 with `numpy` and `requests`.

```bash
cd Ubuntu            # or Windows / Mac-Intel / Mac-M1
cp ../Windows/bot.py .   # Ubuntu/bot.py is the empty template
python main.py       # answer "Y" to test locally, then enter any random seed
```

`main.py` starts the engine as a subprocess and calls `Bot.Run()` once per time step. The bot can use the engine API: `GetPrice`, `GetBalance`, `BuySecurity`, `SellSecurity`, `GetBeta`, `GetFeatures`, `GetMostCorrelated` and others.

Online "trade" mode used the competition's Heroku servers, which no longer respond, so only local testing mode works now.
