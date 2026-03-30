# Institutional Data Pipeline — Technical Spec
## Phase 2 Sandbox · `phase2_sandbox/data_pipeline/`
## Replacing yfinance for Forex

---

## 1. The Problem With yfinance

| Issue | Impact on live trading |
|-------|----------------------|
| Stale quotes (15-min delay on free tier) | Manager checks fill against wrong price |
| No true bid/ask spread | Entry/SL math uses mid only — underestimates slippage |
| Rate limits hit during multi-pair scans | Silent empty DataFrames, missed signals |
| Gaps on illiquid sessions (Fri 5 PM – Sun 5 PM) | Weekend dealing range computed incorrectly |
| No volume data for forex (yf returns 0) | Can't build RVOL filter for forex |

---

## 2. Chosen Sources: OANDA Primary, Massive/Polygon Fallback

### OANDA v20 REST API — Primary Source

**Why OANDA:**
- Free with any practice account (no paid tier needed for paper trading)
- True bid/ask candles (not mid — this matters for spread modeling)
- Nanosecond timestamps, 5500+ instruments
- Streaming endpoint for live price feed
- Same API works for live execution later (no migration needed)

**Key endpoints:**

```
# Candles (historical OHLC, bid/ask or mid)
GET /v3/instruments/{instrument}/candles
  ?price=BA              # B=bid, A=ask, M=mid — use BA for spread capture
  &granularity=M15       # S5,S10,S15,S30,M1,M2,M4,M5,M10,M15,M30,H1,H2,H3,H4,H6,H8,H12,D,W,M
  &count=500             # max 5000 per request
  &from=RFC3339_timestamp
  &to=RFC3339_timestamp
  &includeFirst=True

# Streaming live price (SSE — Server-Sent Events)
GET /v3/accounts/{accountID}/pricing/stream
  ?instruments=EUR_USD,GBP_USD

# Current price snapshot
GET /v3/accounts/{accountID}/pricing
  ?instruments=EUR_USD,GBP_USD
```

**Authentication:**
```
Authorization: Bearer <OANDA_API_TOKEN>
```
Practice server: `https://api-fxpractice.oanda.com`
Live server:     `https://api-fxtrade.oanda.com`

**Rate limits:**
- REST: 120 req/sec (effectively unlimited for our use case)
- Streaming: 2 simultaneous connections per account
- Candles: max 5000 bars per request; paginate for longer history

**Instrument naming:** OANDA uses underscores → `EUR_USD`, `GBP_USD`
(not `EURUSD=X` like yfinance or `C:EURUSD` like Polygon)

**Key difference from yfinance:**
OANDA returns true bid AND ask candles. For Goldbach:
- Use `mid = (bid_close + ask_close) / 2` for level computation
- Use `ask` for BUY entries/SL/TP (you pay the spread on entry)
- Use `bid` for SELL entries/SL/TP

This alone will improve SL accuracy by ~1-2 pips per trade.

---

### Massive/Polygon Forex API — Fallback Source

**Why as fallback (not primary):**
- Free tier: 5 API calls/min (already hitting limits in our equity engine)
- No bid/ask split — mid price only
- 2-year history available (better for backtesting than OANDA's ~6 months)

**Key endpoints:**

```
# Aggregate bars (OHLC)
GET /v2/aggs/ticker/{forexTicker}/range/{multiplier}/{timespan}/{from}/{to}
  forexTicker: C:EURUSD  (C: prefix for forex)
  multiplier:  15
  timespan:    minute
  from:        2026-01-01
  to:          2026-03-30
  limit:       50000      # max per request
  sort:        asc

# Last quote (snapshot)
GET /v1/last_quote/currencies/{from}/{to}
  e.g. /v1/last_quote/currencies/EUR/USD
```

**Use Massive/Polygon for:**
- Long-horizon backtesting (2-year data, OANDA only has ~6 months)
- Secondary validation when OANDA is unavailable

---

## 3. Module Architecture: `OandaFetcher`

```python
# phase2_sandbox/data_pipeline/oanda_fetcher.py

class OandaFetcher:
    """
    Async-capable OANDA v20 data fetcher.
    
    Design principles:
    - Async-first (aiohttp) with sync wrapper for backward compat
    - Automatic pagination for requests > 5000 bars
    - Bid/ask split preserved through the pipeline
    - Hard timeout + retry with exponential backoff
    - Local cache: ~/.quantalgo/cache/oanda_{pair}_{gran}_{date}.parquet
    - Fail-safe: falls back to Massive/Polygon on any OANDA error
    """

    def get_candles(
        self,
        instrument: str,           # "EUR_USD"
        granularity: str,          # "M15"
        count: int = 500,
        from_dt: datetime = None,
        to_dt: datetime = None,
        price: str = "BA",         # "BA" = bid+ask, "M" = mid only
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        time, bid_o, bid_h, bid_l, bid_c,
              ask_o, ask_h, ask_l, ask_c,
              mid_o, mid_h, mid_l, mid_c,
              volume, complete
        All timestamps in UTC, converted to US/Eastern on return.
        """
        ...

    async def stream_prices(
        self,
        instruments: List[str],    # ["EUR_USD", "GBP_USD"]
        callback: Callable,        # called on each price tick
    ) -> None:
        """
        SSE streaming price feed.
        Calls callback(instrument, bid, ask, timestamp) on each tick.
        Used by forex_manager.py for real-time fill detection.
        """
        ...

    def get_live_price(
        self,
        instrument: str,
    ) -> Dict[str, float]:
        """
        Returns: {bid, ask, mid, spread_pips, timestamp}
        Sync wrapper around /v3/accounts/{id}/pricing
        """
        ...
```

---

## 4. Resilience Pattern: Circuit Breaker + Waterfall

```
Request comes in
     │
     ▼
┌────────────┐   success   ┌─────────────────┐
│   OANDA    │────────────▶│  Return data    │
│  v20 REST  │             └─────────────────┘
└────────────┘
     │ fail (timeout/5xx/rate limit)
     ▼
┌────────────┐   success   ┌─────────────────┐
│  Massive/  │────────────▶│  Return data    │
│  Polygon   │             │  + log warning  │
└────────────┘             └─────────────────┘
     │ fail
     ▼
┌────────────┐   success   ┌─────────────────┐
│  yfinance  │────────────▶│  Return data    │
│  (legacy)  │             │  + WARN: stale  │
└────────────┘             └─────────────────┘
     │ all fail
     ▼
┌────────────────────────────────────────────┐
│  FAIL-SAFE: block all signals              │
│  Send Telegram: "Data unavailable"         │
│  Write to results/data_errors.log          │
└────────────────────────────────────────────┘
```

**Circuit breaker state:**
- CLOSED (normal): route to OANDA
- OPEN (after 3 consecutive failures): skip to fallback for 5 min
- HALF-OPEN (after 5 min): try OANDA once; if success → CLOSED

---

## 5. Cache Strategy

```
~/.quantalgo/cache/
  oanda_EUR_USD_M15_2026-03-30.parquet     # intraday (TTL: 15 min)
  oanda_EUR_USD_D_2026-03.parquet          # daily (TTL: 1 day)
  oanda_EUR_USD_M15_history_2yr.parquet    # backtest (TTL: indefinite)
```

- **Parquet format** (not CSV): 10x faster read, typed columns, smaller
- **TTL enforcement** in `_load_cache()`: check file mtime vs TTL
- **Atomic writes** same fcntl pattern as trade_state.json

---

## 6. Spread-Aware Goldbach Levels

With true bid/ask data, dealing range computation improves:

```python
# Current (yfinance — mid only):
prev_high = prev_session["High"].max()    # mid high
prev_low  = prev_session["Low"].min()     # mid low

# Phase 2 (OANDA — bid/ask):
prev_high = prev_session["ask_h"].max()   # highest ask (resistance at ask)
prev_low  = prev_session["bid_l"].min()   # lowest bid (support at bid)

# Entry fill price (spread-aware):
if direction == "buy":
    fill_price = ask_price  # you buy at the ask
else:
    fill_price = bid_price  # you sell at the bid
```

This removes the systematic ~1 pip underestimation of entry cost in
the current goldbach_forex.py.

---

## 7. Build Order

```
Week 1:  data_pipeline/oanda_fetcher.py    — core REST client + pagination
Week 1:  data_pipeline/cache_manager.py    — parquet cache + TTL
Week 2:  data_pipeline/circuit_breaker.py  — waterfall fallback logic
Week 2:  data_pipeline/price_stream.py     — async SSE streaming
Week 3:  Update forex_manager.py to use    — swap yfinance → OandaFetcher
         OandaFetcher.get_live_price()       for fill detection only
Week 4:  Update goldbach_forex.py to use   — full spread-aware scanning
         bid/ask candles for level math
```

---

## 8. Getting an OANDA API Key (5 minutes, free)

1. Go to `https://www.oanda.com/us-en/trading/how-to-trade/forex/`
2. Open a **free practice account** (no real money)
3. Go to `My Services → Manage API Access`
4. Generate a Personal Access Token
5. Add to `~/quantalgo/.env`:
   ```
   OANDA_API_TOKEN=your_token_here
   OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxxx-xxx
   OANDA_ENV=practice   # or "live" when ready
   ```

---

*Spec Owner: Principal Quant Architect*
*Status: APPROVED FOR SANDBOX BUILD*
*Estimated effort: 2 weeks part-time*
