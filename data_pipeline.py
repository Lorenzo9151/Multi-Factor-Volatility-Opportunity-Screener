"""
Data pipeline for the Vol Opportunity Scanner.

All raw data fetching and cleaning lives here:
- WRDS connection and historical queries (skew, earnings, secid lookup)
- yfinance prices, option chains, risk-free rate
- ATM IV extraction with Black-Scholes inversion fallback
- Current VRP / skew snapshots from yfinance

Calculation formulas (VRP = IV - RV, skew = IV_put - IV_call, etc.) are
intentionally kept here next to the data they derive from, since they
are inseparable from the data preparation. Higher-level scoring lives
in screener.py.
"""

import wrds
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq


db = wrds.Connection(wrds_username='lorenzo123')


# ── WRDS lookups ───────────────────────────────────────────────────────────────

def get_secid(ticker):
    result = db.raw_sql(f"""
        SELECT secid FROM optionm_all.securd
        WHERE ticker = '{ticker}'
        LIMIT 1
    """)
    return result['secid'].iloc[0]


def get_earnings_dates(ticker, start_year=2020, end_year=2024, window=3):
    """
    Get earnings announcement dates from IBES and expand into blackout windows.
    window = trading days to exclude before AND after each announcement.
    Returns a set of dates to exclude from historical calculations.
    """
    try:
        earnings = db.raw_sql(f"""
            SELECT DISTINCT actdats_act as earnings_date
            FROM ibes.det_epsus
            WHERE ticker = '{ticker}'
            AND actdats_act >= '{start_year}-01-01'
            AND actdats_act <= '{end_year}-12-31'
            AND measure = 'EPS'
            ORDER BY actdats_act
        """)

        if earnings.empty:
            return set()

        blackout_dates = set()
        for date in pd.to_datetime(earnings['earnings_date']):
            for offset in range(-window, window + 1):
                blackout_dates.add(date + pd.Timedelta(days=offset))

        print(f"  [{ticker}] {len(earnings)} earnings dates -> {len(blackout_dates)} blackout days")
        return blackout_dates

    except Exception as e:
        print(f"  [{ticker}] Earnings filter error: {e}")
        return set()


def get_macro_blackout_dates():
    """
    Hardcoded major macro vol events to exclude from historical baseline.
    Adjustable — add or remove events as needed.
    """
    events = {
        'COVID crash':        ('2020-02-20', '2020-04-30'),
        '2022 rate shock':    ('2022-01-01', '2022-07-01'),
        'SVB crisis':         ('2023-03-08', '2023-03-31'),
        'Aug 2024 VIX spike': ('2024-08-01', '2024-08-15'),
    }

    blackout_dates = set()
    for name, (start, end) in events.items():
        dates = pd.date_range(start=start, end=end, freq='D')
        blackout_dates.update(dates)

    print(f"  Macro blackout: {len(blackout_dates)} days excluded across {len(events)} events")
    return blackout_dates


def get_historical_skew(secid, ticker=None,
                         exclude_earnings=True, earnings_window=3,
                         exclude_macro=True,
                         use_rolling=True, rolling_window=252):
    dfs = []
    for year in range(2020, 2026):
        try:
            df = db.raw_sql(f"""
                SELECT date, delta, cp_flag, impl_volatility
                FROM optionm_all.vsurfd{year}
                WHERE secid = {secid}
                AND days BETWEEN 25 AND 35
                AND delta IN (25.0, -25.0)
                ORDER BY date
            """)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    raw = pd.concat(dfs).drop_duplicates().sort_values('date')

    calls = raw[raw['delta'] == 25.0][['date', 'impl_volatility']].rename(
        columns={'impl_volatility': 'iv_call'})
    puts = raw[raw['delta'] == -25.0][['date', 'impl_volatility']].rename(
        columns={'impl_volatility': 'iv_put'})

    merged = pd.merge(calls, puts, on='date').dropna()
    if merged.empty:
        return None

    merged['skew'] = merged['iv_put'] - merged['iv_call']
    merged['date'] = pd.to_datetime(merged['date'])

    blackout = set()
    if exclude_earnings and ticker:
        blackout.update(get_earnings_dates(ticker, window=earnings_window))
    if exclude_macro:
        blackout.update(get_macro_blackout_dates())

    if blackout:
        before = len(merged)
        merged = merged[~merged['date'].isin(blackout)]
        after  = len(merged)
        print(f"  Filtered {before - after} blackout days -> {after} clean days remaining")

    if merged.empty:
        return None

    merged['skew_mean'] = merged['skew'].mean()
    merged['skew_std']  = merged['skew'].std()

    if use_rolling:
        merged = merged.sort_values('date').reset_index(drop=True)
        merged['skew_rolling_mean'] = merged['skew'].rolling(
            window=rolling_window, min_periods=60).mean()
        merged['skew_rolling_std'] = merged['skew'].rolling(
            window=rolling_window, min_periods=60).std()

    return merged[['date', 'skew', 'skew_mean', 'skew_std'] +
                  (['skew_rolling_mean', 'skew_rolling_std'] if use_rolling else [])]


# ── yfinance helpers ───────────────────────────────────────────────────────────

def get_risk_free_rate():
    try:
        tbill = yf.Ticker('^IRX')
        rate = tbill.fast_info['last_price'] / 100
        return rate
    except Exception:
        return 0.04


def safe_current_price(stock):
    """Get last close — fast_info is flaky when market is closed, so fall back to history."""
    try:
        p = float(stock.fast_info['last_price'])
        if 0 < p < 1e8 and not np.isnan(p):
            return p
    except Exception:
        pass
    try:
        h = stock.history(period='5d')
        if not h.empty:
            return float(h['Close'].iloc[-1])
    except Exception:
        pass
    return None


# ── Black-Scholes IV ───────────────────────────────────────────────────────────

def _bs_call_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _implied_vol_call(S, K, T, r, market_price):
    """Brent's-method IV inversion. Returns None on no convergence."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if market_price <= intrinsic or market_price >= S:
        return None
    try:
        return float(brentq(
            lambda sigma: _bs_call_price(S, K, T, r, sigma) - market_price,
            1e-4, 5.0, xtol=1e-4, maxiter=100
        ))
    except Exception:
        return None


def atm_iv_from_chain(calls_df, spot, T, r):
    """
    Return ATM IV using yfinance's column if it looks real, otherwise
    invert Black-Scholes from each strike's mid/last price. Returns None
    if no strike yields a usable IV.

    yfinance has been returning placeholder IV values (0.00001, etc.)
    with bid=ask=0 — option *prices* are still real, just the IV column
    is broken, so BS inversion is the only reliable path.
    """
    if calls_df is None or calls_df.empty or spot <= 0 or T <= 0:
        return None

    df = calls_df.copy()
    atm = df[abs(df['strike'] - spot) / spot < 0.05]
    if atm.empty:
        atm = df.iloc[(df['strike'] - spot).abs().argsort()[:7]]

    # Step 1: trust yfinance IV if it looks plausible
    yf_iv = atm['impliedVolatility']
    yf_iv = yf_iv[(yf_iv > 0.05) & (yf_iv < 3.0)]
    if len(yf_iv) >= 3:
        return float(yf_iv.median())

    # Step 2: BS inversion from price
    ivs = []
    for _, row in atm.iterrows():
        K = float(row['strike'])
        bid = float(row.get('bid') or 0)
        ask = float(row.get('ask') or 0)
        price = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(row.get('lastPrice') or 0)
        if price <= 0:
            continue
        iv = _implied_vol_call(spot, K, T, r, price)
        if iv is not None and 0.05 < iv < 3.0:
            ivs.append(iv)

    if not ivs:
        return None
    return float(np.median(ivs))


def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return None
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    except Exception:
        return None


# ── Current snapshots ──────────────────────────────────────────────────────────

def get_current_vrp(ticker):
    """ATM IV at 30/45/60d from yfinance plus rolling RV from price history."""
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        print(f"  [{ticker}] No option expirations available")
        return None

    current_price = safe_current_price(stock)
    if current_price is None or current_price <= 0:
        print(f"  [{ticker}] Could not get current price")
        return None

    today = pd.Timestamp.today().normalize()
    r     = get_risk_free_rate()

    def get_iv_for_expiry(expiry):
        if expiry is None:
            return None
        try:
            T = max((pd.Timestamp(expiry) - today).days / 365.0, 1e-6)
            calls = stock.option_chain(expiry).calls
            return atm_iv_from_chain(calls, current_price, T, r)
        except Exception:
            return None

    def nearest_expiry(target_days):
        target = today + pd.Timedelta(days=target_days)
        valid = [e for e in expirations if (pd.Timestamp(e) - today).days >= 14]
        if not valid:
            return None
        return min(valid, key=lambda e: abs((pd.Timestamp(e) - target).days))

    iv_30 = get_iv_for_expiry(nearest_expiry(30))
    iv_45 = get_iv_for_expiry(nearest_expiry(45))
    iv_60 = get_iv_for_expiry(nearest_expiry(60))

    hist = stock.history(period='6mo')
    if hist.empty or len(hist) < 30:
        print(f"  [{ticker}] Insufficient price history for RV")
        return None

    hist = hist.copy()
    hist['log_ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist = hist.dropna(subset=['log_ret'])

    def _rv(window):
        if len(hist) < window:
            return None
        s = hist['log_ret'].iloc[-window:].std()
        if np.isnan(s):
            return None
        return float(s * np.sqrt(252))

    rv_30 = _rv(30)
    rv_45 = _rv(45)
    rv_60 = _rv(60)

    def _pct(v): return f'{v:.1%}' if v is not None else 'N/A'
    print(f"  Spot:{current_price:.2f}  IV  30d:{_pct(iv_30)}  45d:{_pct(iv_45)}  60d:{_pct(iv_60)}")
    print(f"               RV  30d:{_pct(rv_30)}  45d:{_pct(rv_45)}  60d:{_pct(rv_60)}")

    def _vrp(iv, rv):
        if iv is None or rv is None:
            return None
        v = iv - rv
        if abs(v) > 1.0:
            return None
        return v

    return {
        'vrp_30': _vrp(iv_30, rv_30),
        'vrp_45': _vrp(iv_45, rv_45),
        'vrp_60': _vrp(iv_60, rv_60),
    }


def get_current_skew(ticker):
    """25-delta put IV minus 25-delta call IV at the 30d tenor (yfinance)."""
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None

    current_price = stock.fast_info['last_price']
    today = pd.Timestamp.today()
    r = get_risk_free_rate()

    valid = [e for e in expirations if (pd.Timestamp(e) - today).days >= 14]
    if not valid:
        return None

    expiry = min(valid, key=lambda e: abs((pd.Timestamp(e) - today).days - 30))
    T = (pd.Timestamp(expiry) - today).days / 365
    print(f"  [{ticker}] Skew expiry={expiry}, T={T:.3f}")

    try:
        chain = stock.option_chain(expiry)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # 25-delta options are never more than 20% OTM for short dated options
        calls = calls[
            (calls['strike'] >= current_price * 0.80) &
            (calls['strike'] <= current_price * 1.20) &
            (calls['impliedVolatility'] > 0.01)
        ].copy()
        puts = puts[
            (puts['strike'] >= current_price * 0.80) &
            (puts['strike'] <= current_price * 1.20) &
            (puts['impliedVolatility'] > 0.01)
        ].copy()

        if calls.empty or puts.empty:
            print(f"  [{ticker}] Empty after strike range filter")
            return None

        calls['delta'] = calls.apply(lambda row: black_scholes_delta(
            current_price, row['strike'], T, r,
            row['impliedVolatility'], 'call'), axis=1)
        puts['delta'] = puts.apply(lambda row: black_scholes_delta(
            current_price, row['strike'], T, r,
            row['impliedVolatility'], 'put'), axis=1)

        calls = calls.dropna(subset=['delta'])
        puts  = puts.dropna(subset=['delta'])

        if calls.empty or puts.empty:
            return None

        call_25 = calls.iloc[(calls['delta'] - 0.25).abs().argsort()[:1]]
        put_25  = puts.iloc[(puts['delta'] - (-0.25)).abs().argsort()[:1]]

        iv_call = call_25['impliedVolatility'].iloc[0]
        iv_put  = put_25['impliedVolatility'].iloc[0]

        skew = iv_put - iv_call
        if abs(skew) > 0.30:
            print(f"  [{ticker}] Skew {skew:.2%} outside reasonable range - discarding")
            return None

        print(f"  [{ticker}] 25d Call IV={iv_call:.2%}, 25d Put IV={iv_put:.2%}, Skew={skew:.2%}")
        return skew

    except Exception as e:
        print(f"  [{ticker}] Exception: {e}")
        return None
