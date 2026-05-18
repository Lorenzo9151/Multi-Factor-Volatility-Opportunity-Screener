# Vol Opportunity Scanner

Screens equities for volatility trading opportunities across three signals:

| Signal | Definition |
|--------|------------|
| **VRP** | Implied vol minus realized vol at the 30/45/60-day tenor |
| **Skew** | 25-delta put IV minus 25-delta call IV at the 30d tenor, z-scored against history |
| **Term Spread** | 30d ATM IV minus 60d ATM IV, z-scored against history |

Skew and term spread share the same mean-reversion engine: it finds historical periods with a similar z-score and measures how often the signal reverted 25/50/75% toward the mean within 30 trading days.

---

## Install

```
pip install yfinance pandas numpy scipy wrds
```

WRDS access is required for historical OptionMetrics surface data. Set your username in `data_pipeline.py`:

```python
db = wrds.Connection(wrds_username='your_username')
```

---

## Run

```bash
python main.py
```

To change the ticker universe, edit the `TICKERS` constant at the top of `main.py`.

---

## Output

Four ranked tables are printed:

- **VRP** — variance risk premium by tenor (vol percentage points) plus an aggregate VRP score (0–100)
- **SKEW** — current 25-delta skew, its historical mean, z-score, mean-reversion probabilities, half-life, and skew score (0–100)
- **TERM STRUCTURE** — current 30d–60d term spread, historical mean, z-score, reversion probabilities, half-life, and term-structure score (0–100)
- **SUMMARY** — composite score = equal-weight average of the three sub-scores

Scores run 0–100. 50 = at historical mean; higher = more elevated vs history. Missing values print as `--`.

Latest yfinance data is always used. When the US market is closed, yfinance's last available close is used — there is no cached-data fallback.

---

## Project structure

```
.
├── main.py             # entry point
├── screener.py         # orchestration, VRP / skew scoring, mean-reversion engine
├── data_pipeline.py    # WRDS + yfinance fetching, IV extraction (Black-Scholes inversion)
├── term_structure.py   # term spread signal
├── README.md
└── .gitignore
```

---

## Limitations

- Tickers without OptionMetrics vsurfd coverage (e.g. MSTR) will have VRP only — skew and term-structure scores will be `None`.
- yfinance occasionally returns placeholder `impliedVolatility` values when the market is closed; the pipeline falls back to Black-Scholes inversion from the option's mid-price (`bid`/`ask`) or `lastPrice` so the IV remains usable.
- The composite score is a simple equal-weight average — no weighting by signal quality.
