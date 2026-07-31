"""PAPER TRADER — forward-test the step-trail bot on LIVE prices with VIRTUAL money. No API keys.
Reads public Bybit 4h candles; simulates entries/exits + step-trail exactly like the backtest.
State + trade log persist to disk, so it survives restarts. Needs engine.py; pip install ccxt pandas numpy.

  python paper_trader.py --once                 # one check now
  python paper_trader.py --symbol DOGE           # loop forever (paper)
  python paper_trader.py --replay 60             # replay the last 60 days instantly (no waiting)
"""
import os, sys, time, json, math, argparse, traceback
import numpy as np, pandas as pd
from engine import ema, atr

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.00055

def log(sym, msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, f"paper_{sym}.log"), "a") as f:
        f.write(line + "\n")

def sfile(sym): return os.path.join(HERE, f"paper_{sym}_state.json")

def load_state(sym, cap):
    if os.path.exists(sfile(sym)):
        return json.load(open(sfile(sym)))
    return {"equity": cap, "side": 0, "entry": 0.0, "R": 0.0, "peak": 0.0, "qty": 0.0,
            "stop": 0.0, "last_bar_ts": 0, "trades": 0, "wins": 0}

def save_state(sym, s): json.dump(s, open(sfile(sym), "w"), indent=2)

def fetch_df(ex, symbol, tf, limit=320):
    o = ex.fetch_ohlcv(symbol, tf, limit=limit)
    return pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])

def stop_R(peak, activate): return math.floor(peak-activate) if peak >= activate else -1.0

def handle_bar(bar, e, a, up, dn, s, args, sym):
    """Simulate one CLOSED bar. bar = (ts,o,h,l,c). e/a/up/dn scalars for this bar."""
    ts, o, h, l, c = bar
    if s["side"] != 0:
        side, entry, R = s["side"], s["entry"], s["R"]
        # 1) check the CURRENT stop against this bar first (pessimistic)
        cur_stop = s["stop"]
        hit = (l <= cur_stop) if side == 1 else (h >= cur_stop)
        if hit:
            pnl = side*(cur_stop-entry)*s["qty"] - 2*FEE*entry*s["qty"]
            s["equity"] += pnl
            s["trades"] += 1
            if pnl > 0: s["wins"] += 1
            log(sym, f"  EXIT {'LONG' if side==1 else 'SHORT'} @ {cur_stop:.6f}  "
                     f"PnL {pnl:+.2f}  equity {s['equity']:.2f}  (WR {s['wins']}/{s['trades']})")
            s.update({"side":0,"entry":0.0,"R":0.0,"peak":0.0,"qty":0.0,"stop":0.0})
        else:
            # 2) ratchet: update peak from this bar, raise the step-trail stop
            peak_now = (h-entry)/R if side == 1 else (entry-l)/R
            s["peak"] = max(s["peak"], peak_now)
            lvl = stop_R(s["peak"], args.activate)
            new_stop = entry + lvl*R if side == 1 else entry - lvl*R
            s["stop"] = new_stop
    if s["side"] == 0 and not (np.isnan(up) or np.isnan(e) or np.isnan(a)):
        long_sig  = c > up and c > e
        short_sig = c < dn and c < e
        if long_sig or short_sig:
            side = 1 if long_sig else -1
            R = args.k*a
            qty = (s["equity"]*args.risk)/R
            entry = c
            stop = entry - R if side == 1 else entry + R
            s.update({"side":side,"entry":entry,"R":R,"peak":0.0,"qty":qty,"stop":stop})
            log(sym, f"  ENTER {'LONG' if side==1 else 'SHORT'} @ {entry:.6f}  qty {qty:.2f}  "
                     f"stop {stop:.6f}  (risk {args.risk:.0%})")
    s["last_bar_ts"] = int(ts)

def indicators(df, don=20):
    return (ema(df.close,200).values, atr(df,14).values,
            df.high.rolling(don).max().shift(1).values, df.low.rolling(don).min().shift(1).values)

def main():
    import ccxt
    p = argparse.ArgumentParser(description="Paper trader (no keys) — step-trail bot on live prices")
    p.add_argument("--symbol", default="DOGE")
    p.add_argument("--tf", default="4h")
    p.add_argument("--k", type=float, default=1.5)
    p.add_argument("--activate", type=float, default=2.0)
    p.add_argument("--risk", type=float, default=0.02)
    p.add_argument("--capital", type=float, default=10000)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--replay", type=int, default=0, help="replay last N days instantly then exit")
    args = p.parse_args()
    sym = args.symbol.upper(); symbol = f"{sym}/USDT:USDT"
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    # ---- REPLAY: run the bot bar-by-bar over recent history, instantly ----
    if args.replay:
        s = {"equity": args.capital, "side":0,"entry":0.0,"R":0.0,"peak":0.0,"qty":0.0,"stop":0.0,
             "last_bar_ts":0,"trades":0,"wins":0}
        need = args.replay*6 + 260   # 4h bars in N days + warmup
        df = fetch_df(ex, symbol, args.tf, limit=min(need, 1000))
        e,a,up,dn = indicators(df)
        start = max(len(df)-args.replay*6, 210)
        log(sym, f"=== REPLAY last {args.replay}d {symbol} (paper, ${args.capital:,.0f}) ===")
        for i in range(start, len(df)-1):   # exclude the still-forming last bar
            handle_bar(tuple(df.iloc[i][["ts","open","high","low","close"]]), e[i],a[i],up[i],dn[i], s, args, sym)
        ret = (s["equity"]/args.capital-1)*100
        log(sym, f"=== REPLAY done: equity ${s['equity']:,.2f} ({ret:+.1f}%)  "
                 f"trades {s['trades']}  WR {100*s['wins']/max(s['trades'],1):.0f}% ===")
        return

    # ---- LIVE PAPER: loop, act on each newly closed bar ----
    s = load_state(sym, args.capital)
    log(sym, f"=== PAPER live {symbol} {args.tf} | equity ${s['equity']:,.2f} | "
             f"k{args.k} activate{args.activate}R risk{args.risk:.0%} ===")
    while True:
        try:
            df = fetch_df(ex, symbol, args.tf)
            closed = df.iloc[-2]
            if int(closed.ts) > s["last_bar_ts"]:
                e,a,up,dn = indicators(df); j = len(df)-2
                log(sym, f"bar {pd.to_datetime(int(closed.ts),unit='ms')}  close {closed.close:.6f}  "
                         f"side {s['side']}  equity {s['equity']:.2f}")
                handle_bar(tuple(closed[["ts","open","high","low","close"]]), e[j],a[j],up[j],dn[j], s, args, sym)
                save_state(sym, s)
        except Exception:
            log(sym, "ERROR:\n"+traceback.format_exc())
        if args.once: break
        time.sleep(args.poll)

if __name__ == "__main__":
    main()
