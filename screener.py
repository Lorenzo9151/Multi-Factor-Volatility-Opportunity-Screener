"""
Vol Opportunity Scanner — main orchestrator.

Combines VRP, Skew, and Term Structure signals into a composite score.
Raw data fetching and cleaning lives in data_pipeline.py; this module
contains the scoring math, the mean-reversion engine, and the per-ticker
orchestration loop.
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

from data_pipeline import (
    get_secid,
    get_current_vrp,
    get_current_skew,
    get_historical_skew,
)

# Force UTF-8 stdout so unicode characters in print statements don't crash
# the script under Windows console cp1252.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


CACHE_FILE = 'screening_cache.json'


def save_cache(results_df):
    """Save screening results to disk with timestamp."""
    cache = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data': results_df.to_dict(orient='records')
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, default=str)
    print(f"Results cached at {cache['timestamp']}")


def is_market_hours():
    """Check if US market is currently open."""
    now = datetime.now()
    if now.weekday() > 4:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0)
    market_close = now.replace(hour=16, minute=0,  second=0)
    return market_open <= now <= market_close


# ── Scoring ────────────────────────────────────────────────────────────────────

def vrp_to_score(vrp_value):
    score = 50 + (vrp_value / 0.40) * 50
    return round(max(0, min(100, score)), 1)


def compute_vrp_score(ticker, secid):
    current = get_current_vrp(ticker)
    if current is None:
        return None

    available = {k: v for k, v in current.items() if v is not None}
    if not available:
        return None

    vrp_score = round(sum(vrp_to_score(v) for v in available.values()) / len(available), 1)
    avg_val   = round(sum(available.values()) / len(available) * 100, 2)

    def _val(key): return round(current[key] * 100, 2) if current[key] is not None else None

    return {
        'vrp_score':     vrp_score,
        'vrp_30d_value': _val('vrp_30'),
        'vrp_45d_value': _val('vrp_45'),
        'vrp_60d_value': _val('vrp_60'),
        'vrp_avg_value': avg_val,
    }


def compute_reversion_probs(hist_skew, current_skew):
    """
    Mean-reversion engine. Finds historical periods (analogues) where the
    z-score was close to today's, and measures how often the series reverted
    25 / 50 / 75% of the way back to the mean within 30 trading days.
    Used by both skew and term structure signals.
    """
    mean = hist_skew['skew_mean'].iloc[0]
    std  = hist_skew['skew_std'].iloc[0]
    current_z = (current_skew - mean) / std

    hist_skew = hist_skew.copy().reset_index(drop=True)
    hist_skew['z'] = (hist_skew['skew'] - mean) / std
    analogues = hist_skew[abs(hist_skew['z'] - current_z) <= 0.5].index.tolist()

    if len(analogues) < 10:
        return None

    times_to_50 = []
    hits_25, hits_50, hits_75 = 0, 0, 0

    for idx in analogues:
        s0 = hist_skew.loc[idx, 'skew']
        end = min(idx + 30, len(hist_skew) - 1)
        future = hist_skew.loc[idx + 1:end, 'skew']

        if future.empty:
            continue

        denom = s0 - mean
        if abs(denom) < 0.001:
            continue

        rt = (s0 - future) / denom
        max_rt = rt.max()

        if max_rt >= 0.25: hits_25 += 1
        if max_rt >= 0.50: hits_50 += 1
        if max_rt >= 0.75: hits_75 += 1

        half_days = rt[rt >= 0.50]
        if not half_days.empty:
            times_to_50.append(half_days.index[0] - idx)

    n = len(analogues)
    return {
        'prob_25':     round(hits_25 / n * 100, 1),
        'prob_50':     round(hits_50 / n * 100, 1),
        'prob_75':     round(hits_75 / n * 100, 1),
        'half_life':   round(np.median(times_to_50), 1) if times_to_50 else None,
        'n_analogues': n,
    }


def compute_skew_score(ticker, secid):
    hist = get_historical_skew(secid, ticker=ticker,
                                exclude_earnings=True, earnings_window=3,
                                exclude_macro=True,
                                use_rolling=True, rolling_window=252)
    current_skew = get_current_skew(ticker)

    if hist is None or current_skew is None:
        return None

    if 'skew_rolling_mean' in hist.columns:
        rolling_mean = hist['skew_rolling_mean'].dropna().iloc[-1]
        rolling_std  = hist['skew_rolling_std'].dropna().iloc[-1]
        static_mean  = hist['skew_mean'].iloc[0]
        static_std   = hist['skew_std'].iloc[0]
        z_rolling = (current_skew - rolling_mean) / rolling_std
        z_static  = (current_skew - static_mean)  / static_std
        z = z_rolling
    else:
        static_mean  = hist['skew_mean'].iloc[0]
        static_std   = hist['skew_std'].iloc[0]
        z            = (current_skew - static_mean) / static_std
        rolling_mean = None
        z_static     = z
        rolling_std  = None

    reversion  = compute_reversion_probs(hist, current_skew)
    skew_score = round(max(0, min(100, 50 + (z / 3) * 50)), 1)

    return {
        'skew_score':        skew_score,
        'current_skew':      round(current_skew * 100, 2),
        'skew_mean':         round(static_mean * 100, 2),
        'skew_rolling_mean': round(rolling_mean * 100, 2) if rolling_mean is not None else None,
        'skew_z':            round(z, 2),
        'skew_z_static':     round(z_static, 2),
        'prob_25':           reversion['prob_25']     if reversion else None,
        'prob_50':           reversion['prob_50']     if reversion else None,
        'prob_75':           reversion['prob_75']     if reversion else None,
        'half_life':         reversion['half_life']   if reversion else None,
        'n_analogues':       reversion['n_analogues'] if reversion else None,
    }


# ── Orchestration ──────────────────────────────────────────────────────────────

def score_stock(ticker):
    # Lazy import to avoid circular dependency (term_structure imports compute_reversion_probs)
    from term_structure import compute_term_structure_score

    secid = get_secid(ticker)

    vrp_result  = compute_vrp_score(ticker, secid)
    skew_result = compute_skew_score(ticker, secid)
    ts_result   = compute_term_structure_score(ticker, secid)

    vrp_score  = vrp_result['vrp_score']   if vrp_result  else None
    skew_score = skew_result['skew_score'] if skew_result else None
    ts_score   = ts_result['ts_score']     if ts_result   else None

    scores    = [s for s in [vrp_score, skew_score, ts_score] if s is not None]
    composite = round(sum(scores) / len(scores), 1) if scores else None

    def _g(d, k): return d[k] if d else None

    return {
        'ticker':            ticker,
        # VRP
        'vrp_30d_value':     _g(vrp_result, 'vrp_30d_value'),
        'vrp_45d_value':     _g(vrp_result, 'vrp_45d_value'),
        'vrp_60d_value':     _g(vrp_result, 'vrp_60d_value'),
        'vrp_avg_value':     _g(vrp_result, 'vrp_avg_value'),
        'vrp_score':         vrp_score,
        # Skew
        'current_skew':      _g(skew_result, 'current_skew'),
        'skew_mean':         _g(skew_result, 'skew_mean'),
        'skew_z':            _g(skew_result, 'skew_z'),
        'prob_25':           _g(skew_result, 'prob_25'),
        'prob_50':           _g(skew_result, 'prob_50'),
        'prob_75':           _g(skew_result, 'prob_75'),
        'half_life':         _g(skew_result, 'half_life'),
        'skew_rolling_mean': _g(skew_result, 'skew_rolling_mean'),
        'skew_z_static':     _g(skew_result, 'skew_z_static'),
        'skew_score':        skew_score,
        # Term structure (30d minus 60d ATM IV spread)
        'current_term_spread': _g(ts_result, 'current_term_spread'),
        'term_spread_mean':    _g(ts_result, 'term_spread_mean'),
        'term_spread_z':       _g(ts_result, 'term_spread_z'),
        'ts_prob_50':          _g(ts_result, 'ts_prob_50'),
        'ts_half_life':        _g(ts_result, 'ts_half_life'),
        'ts_score':            ts_score,
        # Summary
        'composite_score':   composite,
    }


def screen(tickers):
    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        try:
            results.append(score_stock(ticker))
        except Exception as e:
            print(f"  [{ticker}] Skipped - {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values('composite_score', ascending=False, na_position='last').reset_index(drop=True)
    return df
