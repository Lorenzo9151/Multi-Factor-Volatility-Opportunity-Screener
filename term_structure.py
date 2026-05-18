"""
Term Structure Module
=====================
Signal: 30-day ATM IV minus 60-day ATM IV (the "30/60 term spread").

Positive term_spread  = front-end IV above back-end (backwardated / event risk).
Negative term_spread  = front-end IV below back-end (contango / normal).

A high score means term_spread is elevated vs its own history — front-end IV
is unusually rich vs back-end, candidate for mean reversion.
"""

import pandas as pd
import yfinance as yf

from data_pipeline import (
    db,
    get_earnings_dates,
    get_macro_blackout_dates,
    atm_iv_from_chain,
    get_risk_free_rate,
    safe_current_price,
)
from screener import compute_reversion_probs


# ── Historical data ────────────────────────────────────────────────────────────

def get_historical_term_spread(secid, ticker=None,
                                exclude_earnings=True, earnings_window=3,
                                exclude_macro=True,
                                use_rolling=True, rolling_window=252):
    """
    Pull ATM (delta=50) call IV at 30d and 60d tenors from WRDS vsurfd.
    Return a daily series of term_spread = iv_30 - iv_60.
    """
    iv30_parts, iv60_parts = [], []

    for year in range(2020, 2026):
        try:
            df = db.raw_sql(f"""
                SELECT date, days, impl_volatility
                FROM optionm_all.vsurfd{year}
                WHERE secid = {secid}
                  AND delta = 50.0
                  AND cp_flag = 'C'
                  AND days BETWEEN 20 AND 70
                  AND impl_volatility IS NOT NULL
                ORDER BY date
            """)
            if df.empty:
                continue
            part_30 = (df[df['days'].between(25, 35)]
                       [['date', 'impl_volatility']]
                       .rename(columns={'impl_volatility': 'iv_30'}))
            part_60 = (df[df['days'].between(55, 65)]
                       [['date', 'impl_volatility']]
                       .rename(columns={'impl_volatility': 'iv_60'}))
            if not part_30.empty: iv30_parts.append(part_30)
            if not part_60.empty: iv60_parts.append(part_60)
        except Exception:
            continue

    if not iv30_parts or not iv60_parts:
        return None

    iv30 = pd.concat(iv30_parts).drop_duplicates('date').sort_values('date')
    iv60 = pd.concat(iv60_parts).drop_duplicates('date').sort_values('date')

    merged = pd.merge(iv30, iv60, on='date').dropna()
    if merged.empty:
        return None

    merged['term_spread'] = merged['iv_30'] - merged['iv_60']
    merged['date']        = pd.to_datetime(merged['date'])

    # ── Filters ────────────────────────────────────────────────────────────────
    blackout = set()
    if exclude_earnings and ticker:
        blackout.update(get_earnings_dates(ticker, window=earnings_window))
    if exclude_macro:
        blackout.update(get_macro_blackout_dates())
    if blackout:
        before = len(merged)
        merged = merged[~merged['date'].isin(blackout)]
        print(f"  TS filtered {before - len(merged)} blackout days -> {len(merged)} clean")

    if merged.empty:
        return None

    merged['term_spread_mean'] = merged['term_spread'].mean()
    merged['term_spread_std']  = merged['term_spread'].std()

    if use_rolling:
        merged = merged.sort_values('date').reset_index(drop=True)
        merged['term_spread_rolling_mean'] = (merged['term_spread']
                                              .rolling(rolling_window, min_periods=60)
                                              .mean())
        merged['term_spread_rolling_std']  = (merged['term_spread']
                                              .rolling(rolling_window, min_periods=60)
                                              .std())

    cols = ['date', 'term_spread', 'term_spread_mean', 'term_spread_std']
    if use_rolling:
        cols += ['term_spread_rolling_mean', 'term_spread_rolling_std']
    return merged[cols]


# ── Current data ───────────────────────────────────────────────────────────────

def get_current_term_spread(ticker):
    """
    ATM IV at the 30d and 60d expiries from yfinance.
    Returns term_spread = iv_30 - iv_60 (decimal).
    """
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None

    current_price = safe_current_price(stock)
    if current_price is None or current_price <= 0:
        return None

    today = pd.Timestamp.today().normalize()
    r     = get_risk_free_rate()

    def nearest_expiry(target_days):
        valid = [e for e in expirations if (pd.Timestamp(e) - today).days >= 14]
        if not valid:
            return None
        return min(valid, key=lambda e: abs((pd.Timestamp(e) - today).days - target_days))

    def get_atm_iv(expiry):
        if expiry is None:
            return None
        try:
            T = max((pd.Timestamp(expiry) - today).days / 365.0, 1e-6)
            calls = stock.option_chain(expiry).calls
            return atm_iv_from_chain(calls, current_price, T, r)
        except Exception:
            return None

    iv_30 = get_atm_iv(nearest_expiry(30))
    iv_60 = get_atm_iv(nearest_expiry(60))

    if iv_30 is None or iv_60 is None:
        return None

    spread = iv_30 - iv_60
    if abs(spread) > 0.50:
        print(f"  [{ticker}] TS spread {spread:+.1%} outside reasonable range — discarding")
        return None

    print(f"  [{ticker}] TS: IV30={iv_30:.1%}  IV60={iv_60:.1%}  spread={spread:+.1%}")
    return spread


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_term_structure_score(ticker, secid):
    """
    Score current term_spread vs its historical distribution.
    Reuses compute_reversion_probs from screener — no duplicated logic.
    """
    hist           = get_historical_term_spread(secid, ticker=ticker,
                                                 exclude_earnings=True,
                                                 earnings_window=3,
                                                 exclude_macro=True,
                                                 use_rolling=True,
                                                 rolling_window=252)
    current_spread = get_current_term_spread(ticker)

    if hist is None or current_spread is None:
        return None

    if 'term_spread_rolling_mean' in hist.columns:
        rolling_mean = hist['term_spread_rolling_mean'].dropna().iloc[-1]
        rolling_std  = hist['term_spread_rolling_std'].dropna().iloc[-1]
        static_mean  = hist['term_spread_mean'].iloc[0]
        static_std   = hist['term_spread_std'].iloc[0]
        z_rolling    = (current_spread - rolling_mean) / rolling_std
        z_static     = (current_spread - static_mean)  / static_std
        z = z_rolling
    else:
        static_mean  = hist['term_spread_mean'].iloc[0]
        static_std   = hist['term_spread_std'].iloc[0]
        z            = (current_spread - static_mean) / static_std
        rolling_mean = rolling_std = None
        z_static     = z

    # compute_reversion_probs expects columns named skew/skew_mean/skew_std
    hist_rev = hist.rename(columns={
        'term_spread':      'skew',
        'term_spread_mean': 'skew_mean',
        'term_spread_std':  'skew_std',
    })
    reversion = compute_reversion_probs(hist_rev, current_spread)

    ts_score = round(max(0, min(100, 50 + (z / 3) * 50)), 1)

    def _r(x): return round(x * 100, 2) if x is not None else None

    return {
        'ts_score':                 ts_score,
        'current_term_spread':      _r(current_spread),
        'term_spread_mean':         _r(static_mean),
        'term_spread_rolling_mean': _r(rolling_mean),
        'term_spread_z':            round(z, 2),
        'term_spread_z_static':     round(z_static, 2),
        'ts_prob_25':               reversion['prob_25']     if reversion else None,
        'ts_prob_50':               reversion['prob_50']     if reversion else None,
        'ts_prob_75':               reversion['prob_75']     if reversion else None,
        'ts_half_life':             reversion['half_life']   if reversion else None,
        'ts_n_analogues':           reversion['n_analogues'] if reversion else None,
    }
