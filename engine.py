"""Shared RTS engine for Config C / Config D bots. Requires: pip install ccxt pandas numpy
Keep engine.py, config_c.py, config_d.py in the SAME folder."""
import os, time
import numpy as np, pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
TF_MS = {"1h": 3600_000, "2h": 7200_000, "4h": 14400_000, "1d": 86400_000}
FEE = 0.00055  # Bybit taker

# ------------------------------------------------------------------ data
def fetch_ohlcv(symbol, tf="4h", start="2024-01-01", end="2026-07-25"):
    import ccxt
    safe = symbol.replace("/", "_").replace(":", "_")
    cache = os.path.join(DATA_DIR, f"{safe}_{tf}.csv")
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    s = ex.parse8601(start + "T00:00:00Z"); e = ex.parse8601(end + "T00:00:00Z")
    if os.path.exists(cache):
        h = pd.read_csv(cache)
        if h.ts.min() <= s and h.ts.max() >= e - TF_MS[tf]:
            return h[(h.ts >= s) & (h.ts <= e)].reset_index(drop=True)
    rows, since = [], s
    while since < e:
        b = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not b:
            break
        rows += b; since = b[-1][0] + TF_MS[tf]
        if len(b) < 1000 and since < e:
            time.sleep(ex.rateLimit / 1000)
        if b[-1][0] >= e:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df = df[(df.ts >= s) & (df.ts <= e)].reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df

# ------------------------------------------------------------------ indicators
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rma(s, n): return s.ewm(alpha=1 / n, adjust=False).mean()
def atr(df, n):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)

def supertrend(df, period=10, mult=3.0):
    a = atr(df, period).values; hl2 = ((df.high + df.low) / 2).values; c = df.close.values; n = len(df)
    ub = hl2 + mult * a; lb = hl2 - mult * a; fu = ub.copy(); fl = lb.copy(); d = np.ones(n)
    for i in range(1, n):
        fu[i] = min(ub[i], fu[i - 1]) if c[i - 1] <= fu[i - 1] else ub[i]
        fl[i] = max(lb[i], fl[i - 1]) if c[i - 1] >= fl[i - 1] else lb[i]
        d[i] = 1 if c[i] > fu[i - 1] else (-1 if c[i] < fl[i - 1] else d[i - 1])
    return d

# ------------------------------------------------------------------ strategy: protected ATR Trail Rider
def signal_atr_trail(df, maxloss=0.08):
    """Enter on Supertrend flip aligned with EMA200; exit on flip-against OR hard stop; flat otherwise."""
    d = supertrend(df, 10, 3.0); e = ema(df.close, 200).values; c = df.close.values; n = len(df)
    pos = 0; entry = np.nan; st = np.zeros(n)
    for i in range(n):
        if pos == 1:
            if c[i] <= entry * (1 - maxloss) or d[i] == -1: pos = 0
        elif pos == -1:
            if c[i] >= entry * (1 + maxloss) or d[i] == 1: pos = 0
        if pos == 0 and not np.isnan(e[i]):
            if d[i] == 1 and c[i] > e[i]: pos = 1; entry = c[i]
            elif d[i] == -1 and c[i] < e[i]: pos = -1; entry = c[i]
        st[i] = pos
    return st

# ------------------------------------------------------------------ backtest pieces
def bar_idx(df): return pd.to_datetime(df.ts.values, unit="ms")

def trend_base(df, state, vol_target=None):
    """Per-bar return @1x. If vol_target set: inverse-vol position scaling (cap 2x) = volatility targeting."""
    c = df.close.values; ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1
    pos = np.zeros(len(c)); pos[1:] = state[:-1]          # decide on close[i], hold next bar (no lookahead)
    if vol_target is not None:
        rv = pd.Series(ret).rolling(30).std().bfill().values
        scale = np.clip(np.divide(vol_target, np.where(rv == 0, np.nan, rv)), 0, 2.0)
        pos = pos * np.nan_to_num(scale, nan=1.0)
    turn = np.abs(np.concatenate([[pos[0]], np.diff(pos)]))
    return pd.Series(pos * ret - turn * FEE, index=bar_idx(df))

def funding_base(symbol, idx, start="2024-01-01", end="2026-07-25"):
    """Market-neutral funding sleeve (long spot + short perp): per-4h-bar funding received."""
    import ccxt
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = ex.parse8601(start + "T00:00:00Z"); e = ex.parse8601(end + "T00:00:00Z"); rows = []
    while since < e:
        b = ex.fetch_funding_rate_history(symbol, since=since, limit=200)
        if not b:
            break
        rows += b; since = b[-1]["timestamp"] + 1
        if len(b) < 200:
            break
        time.sleep(ex.rateLimit / 1000)
    f = pd.Series([r["fundingRate"] for r in rows],
                  index=pd.to_datetime([r["timestamp"] for r in rows], unit="ms"))
    f = f[~f.index.duplicated()]
    return f.resample("4h").sum().reindex(idx).fillna(0.0)

def median_vol_target(dfs):
    return float(np.median([pd.Series(np.diff(np.log(df.close.values))).rolling(30).std().median() for df in dfs]))

# ------------------------------------------------------------------ report
def report(base, capital, label, leverage=1.0):
    step = np.clip(1 + leverage * base.values, 0, None)
    eq = np.cumprod(step); s = pd.Series(eq, index=base.index)
    total = (eq[-1] - 1) * 100
    days = max((s.index[-1] - s.index[0]).days, 1)
    cagr = (eq[-1] ** (365 / days) - 1) * 100
    peak = np.maximum.accumulate(eq); dd = (eq / peak - 1).min() * 100
    wr = s.resample("W").last().dropna().pct_change().dropna()
    yl = s.resample("YE").last(); yr = yl.pct_change();
    if len(yl): yr.iloc[0] = yl.iloc[0] - 1
    print(f"\n{'='*64}\n {label}   (lev {leverage:g}x)\n{'='*64}")
    print(f" Capital {capital:,.0f}  ->  {capital*eq[-1]:,.0f}   (x{eq[-1]:.2f})")
    print(f" Total {total:+.0f}%   CAGR {cagr:+.0f}%   MaxDD {dd:.0f}%")
    print(f" Weekly:  avg {wr.mean()*100:+.2f}%   median {wr.median()*100:+.2f}%   "
          f"positive {(wr>0).mean()*100:.0f}%   worst {wr.min()*100:+.1f}%   best {wr.max()*100:+.1f}%")
    print(" Per year: " + "  ".join(f"{i.year}:{v*100:+.0f}%" for i, v in yr.items()))
    out = os.path.join(DATA_DIR, label.split()[0].lower() + "_equity.csv")
    s.rename("equity").to_csv(out)
    print(f" Equity curve -> {out}")
    return s
