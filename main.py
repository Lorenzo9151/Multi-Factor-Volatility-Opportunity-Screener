"""
Vol Opportunity Scanner — entry point.

    python main.py

To change the universe, edit TICKERS below.
"""

from screener import screen, save_cache, is_market_hours

TICKERS = ['AAPL', 'NVDA', 'TSLA', 'INTC', 'AMD', 'COIN', 'PLTR']


def _fmt(df, cols, na='--'):
    sub = df[[c for c in cols if c in df.columns]]
    return sub.to_string(index=False, na_rep=na)


def main():
    status = "open" if is_market_hours() else "closed (using last close prices)"
    print(f"Market {status} - running full screen on {len(TICKERS)} tickers...")

    results = screen(TICKERS)
    if results.empty:
        print("No results.")
        return

    save_cache(results)

    print("\n--- VRP ---")
    print(_fmt(results, ['ticker', 'vrp_30d_value', 'vrp_45d_value',
                         'vrp_60d_value', 'vrp_avg_value', 'vrp_score']))

    print("\n--- SKEW ---")
    print(_fmt(results, ['ticker', 'current_skew', 'skew_mean', 'skew_rolling_mean',
                         'skew_z', 'skew_z_static', 'prob_25', 'prob_50', 'prob_75',
                         'half_life', 'skew_score']))

    print("\n--- TERM STRUCTURE (30d - 60d IV) ---")
    print(_fmt(results, ['ticker', 'current_term_spread', 'term_spread_mean',
                         'term_spread_z', 'ts_prob_50', 'ts_half_life', 'ts_score']))

    print("\n--- SUMMARY ---")
    print(_fmt(results, ['ticker', 'vrp_score', 'skew_score',
                         'ts_score', 'composite_score']))


if __name__ == '__main__':
    main()
