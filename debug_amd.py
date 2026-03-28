#!/usr/bin/env python3
"""Debug script: Find exactly why amd_cycle generates 0 trades."""
import pandas as pd
from data_fetcher import download_data

df = download_data("GBPJPY", period="2y", interval="1h")
print(f"Total bars: {len(df)}")
print(f"Index type: {type(df.index)}")
print(f"Index dtype: {df.index.dtype}")
print(f"Timezone: {df.index.tz}")
print(f"First 3 index values: {df.index[:3].tolist()}")
print()

# Step 1: Can we convert timezone?
df_work = df.copy()
if not isinstance(df_work.index, pd.DatetimeIndex):
    print("FAIL: Index is NOT DatetimeIndex")
    try:
        df_work.index = pd.to_datetime(df_work.index)
        print("  -> Converted to DatetimeIndex")
    except Exception as e:
        print(f"  -> Conversion failed: {e}")
        exit()
else:
    print("PASS: Index IS DatetimeIndex")

# Step 2: Strip timezone
print(f"\nTimezone before strip: {df_work.index.tz}")
if df_work.index.tz is not None:
    try:
        df_work.index = df_work.index.tz_localize(None)
        print("PASS: tz_localize(None) worked")
    except Exception as e1:
        print(f"FAIL: tz_localize(None) raised: {e1}")
        try:
            df_work.index = df_work.index.tz_convert(None)
            print("PASS: tz_convert(None) worked instead")
        except Exception as e2:
            print(f"FAIL: tz_convert(None) also raised: {e2}")
            exit()
else:
    print("No timezone to strip")

print(f"Timezone after strip: {df_work.index.tz}")

# Step 3: Group by date
df_work["_date"] = df_work.index.date
groups = df_work.groupby("_date")
print(f"\nTotal unique dates: {len(groups)}")

# Step 4: Check each day
skip_reasons = {"too_few_bars": 0, "asian_no_range": 0, "no_sweep": 0, "would_signal": 0}
total_days = 0

for date, day_df in groups:
    total_days += 1
    n_bars = len(day_df)
    
    if n_bars < 6:
        skip_reasons["too_few_bars"] += 1
        continue

    split1 = n_bars // 3
    split2 = 2 * n_bars // 3

    asian = day_df.iloc[:split1]
    london = day_df.iloc[split1:split2]
    ny = day_df.iloc[split2:]

    if len(asian) < 2 or len(london) < 2 or len(ny) < 1:
        skip_reasons["too_few_bars"] += 1
        continue

    asian_high = asian["High"].max()
    asian_low = asian["Low"].min()
    asian_range = asian_high - asian_low
    
    if asian_range <= 0:
        skip_reasons["asian_no_range"] += 1
        continue

    london_high = london["High"].max()
    london_low = london["Low"].min()

    swept_lows = london_low < asian_low
    swept_highs = london_high > asian_high

    if not swept_lows and not swept_highs:
        skip_reasons["no_sweep"] += 1
        continue

    skip_reasons["would_signal"] += 1
    
    # Print first 3 examples
    if skip_reasons["would_signal"] <= 3:
        print(f"\n  EXAMPLE {skip_reasons['would_signal']}: {date}")
        print(f"    Bars: {n_bars} | Asian: {len(asian)} | London: {len(london)} | NY: {len(ny)}")
        print(f"    Asian range: {asian_low:.3f} - {asian_high:.3f} ({asian_range:.3f})")
        print(f"    London: {london_low:.3f} - {london_high:.3f}")
        print(f"    Swept lows: {swept_lows} | Swept highs: {swept_highs}")
        entry = float(ny["Open"].iloc[0])
        sl_buffer = asian_range * 0.3
        if swept_lows:
            sl = london_low - sl_buffer
            tp = entry + asian_range * 1.5
            print(f"    BUY: entry={entry:.3f} sl={sl:.3f} tp={tp:.3f} valid={entry > sl and tp > entry}")
        elif swept_highs:
            sl = london_high + sl_buffer
            tp = entry - asian_range * 1.5
            print(f"    SELL: entry={entry:.3f} sl={sl:.3f} tp={tp:.3f} valid={entry < sl and tp < entry}")

print(f"\n{'='*50}")
print(f"RESULTS: {total_days} total days")
for reason, count in skip_reasons.items():
    pct = count / total_days * 100 if total_days > 0 else 0
    print(f"  {reason}: {count} ({pct:.1f}%)")
print(f"{'='*50}")

if skip_reasons["would_signal"] > 0:
    print(f"\nCONCLUSION: Logic DOES find setups ({skip_reasons['would_signal']} days).")
    print("The bug is in bar_idx mapping back to the original DataFrame.")
else:
    print(f"\nCONCLUSION: No setups found. Check the sweep conditions.")
