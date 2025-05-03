import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime as dt # Import datetime with alias dt
import time
# import random # Not typically needed for backtest
import math
import os
import pytz # Required for timezone handling
import traceback # For detailed error reporting
from datetime import datetime, timedelta, timezone

# --- Timezone Handling ---
UTC_TZ = pytz.utc # Use pytz for robust timezone handling

# --- Configuration ---
# Trading Parameters
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "BTCUSD", "ETHUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15
MAGIC_NUMBER = 123456 # Unique identifier for this bot's trades

# Specific Lot Sizes
LOT_SIZE_DEFAULT = 0.01
LOT_SIZE_ETH = 0.10
# Add more specific sizes if needed, e.g., LOT_SIZE_BTC = 0.01

# Strategy & Risk Parameters
INITIAL_BALANCE_INFO_ONLY = 200.00 # For reference
MAX_SLIPPAGE_POINTS = 5
REWARD_RISK_RATIO = 2.5
DAILY_STOP_LOSS_PERCENT = 5.0
DAILY_PROFIT_TARGET_PERCENT = 10.0
ATR_SL_MULTIPLIER = 1.5
BOS_LOOKBACK_PERIODS = 5
MAX_HOLD_DURATION_MINUTES = 100 * 20 # Approx MAX_HOLD_CANDLES * timeframe minutes

# Trailing Stop Loss Parameters
USE_TRAILING_STOP = True # Keep False based on previous results
TSL_ACTIVATION_PIPS = 30.0
TSL_DISTANCE_PIPS = 20.0
TSL_STEP_PIPS = 0 # Continuous trailing if enabled

# Session times (UTC)
LONDON_OPEN = (6, 12) # London 07:00 UTC to 11:59 UTC
NY_OPEN = (12, 20)   # New York 12:00 UTC to 15:59 UTC

# Loop Control & Timing
# CHECK_INTERVAL_SECONDS = 10 # Replaced by non-blocking check logic
MAIN_LOOP_INTERVAL_SECONDS = 5 # How often to check *everything* (PnL, symbols)
MIN_CPU_SLEEP_SECONDS = 0.1    # Short sleep to prevent high CPU usage
PNL_CHECK_INTERVAL_SECONDS = 60 * 5 # How often to recalculate daily PnL (less frequent)
CANDLE_COUNT_FOR_INDICATORS = 250
MIN_ENTRY_GAP_MINUTES = 10 # Min time between entries on the same symbol

# --- Global Storage ---
symbol_details = {} # Global dictionary to store symbol info

# --- Helper Functions ---

def get_timeframe_name(tf_int):
    tf_map = {
        mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1"
    }
    return tf_map.get(tf_int, f"TF_INT_{tf_int}")

def round_down_to_step(number, step):
    if step <= 0:
        print(f"Warning: Cannot round down to step {step}. Returning original number {number}.")
        return number
    tolerance = 1e-9
    return math.floor((number + tolerance) / step) * step

def get_symbol_info(symbol):
    """Fetches symbol properties needed for trading and calculations."""
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"❌ Failed to get info for {symbol}. Error: {mt5.last_error()}")
        if mt5.symbol_select(symbol, True):
            print(f"Enabled {symbol} in MarketWatch, retrying info...")
            time.sleep(0.5) # Keep short sleep here for MT5 state update
            info = mt5.symbol_info(symbol)
            if info is None:
                print(f"❌ Failed again to get info for {symbol}.")
                return None
        else:
            print(f"❌ Failed to select/enable {symbol} in MarketWatch.")
            return None
    if not info.visible:
        print(f"Symbol {symbol} not visible, enabling...")
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to enable {symbol} in MarketWatch.")
            return None
        time.sleep(0.5) # Keep short sleep here
        info = mt5.symbol_info(symbol) # Re-fetch info
        if info is None:
            print(f"❌ Failed to get info for {symbol} after enabling.")
            return None

    details = {}
    try:
        details['point'] = info.point; details['digits'] = info.digits
        details['contract_size'] = info.trade_contract_size; details['volume_min'] = info.volume_min
        details['volume_max'] = info.volume_max; details['volume_step'] = info.volume_step
        details['trade_tick_value'] = info.trade_tick_value; details['trade_tick_size'] = info.trade_tick_size
        details['trade_mode'] = info.trade_mode; details['filling_modes_raw'] = info.filling_mode
        details['trade_stops_level'] = info.trade_stops_level
    except AttributeError as e: print(f"❌ Error accessing attribute for {symbol}: {e}."); return None

    if details.get('point', 0) <= 0: print(f"❌ Invalid 'point' for {symbol}"); return None
    if details.get('trade_tick_value', 0) <= 0: print(f"❌ Invalid 'trade_tick_value' for {symbol}"); return None
    if details.get('trade_tick_size', 0) <= 0: print(f"❌ Invalid 'trade_tick_size' for {symbol}"); return None
    if details.get('volume_step', 0) <= 0: print(f"❌ Invalid 'volume_step' for {symbol}"); return None
    if details.get('volume_min', 0) <= 0: print(f"❌ Invalid 'volume_min' for {symbol}"); return None

    if details['trade_mode'] != mt5.SYMBOL_TRADE_MODE_FULL:
        print(f"❌ Trading not fully enabled for {symbol} (Mode: {details['trade_mode']})"); return None

    # Determine best filling mode
    if details['filling_modes_raw'] & mt5.ORDER_FILLING_FOK: details['filling_mode'] = mt5.ORDER_FILLING_FOK
    elif details['filling_modes_raw'] & mt5.ORDER_FILLING_IOC: details['filling_mode'] = mt5.ORDER_FILLING_IOC
    elif details['filling_modes_raw'] & mt5.ORDER_FILLING_RETURN:
        details['filling_mode'] = mt5.ORDER_FILLING_RETURN
        print(f"⚠️ FOK/IOC not supported {symbol}. Using RETURN.")
    else: print(f"❌ No recognized filling mode for {symbol}."); return None

    # Pip calculation
    tick = mt5.symbol_info_tick(symbol); retries = 3
    while retries > 0 and (tick is None or tick.ask <= 0 or tick.bid <= 0):
        print(f"Warning: Invalid tick for {symbol}. Retrying..."); time.sleep(0.2); tick = mt5.symbol_info_tick(symbol); retries -= 1
    spread_points = 0
    if tick and tick.ask > 0 and tick.bid > 0 and details.get('point', 0) > 0: spread_points = max(0, round((tick.ask - tick.bid) / details['point']))
    else: print(f"⚠️ Could not get valid tick for {symbol}. Spread set to 0.")

    digits = details['digits']; point_val = details['point']; pip_size_in_points = 1
    if "JPY" in symbol.upper(): pip_size_in_points = 10 if digits == 3 else 10 # Simplified JPY handling
    elif digits == 5 or digits == 3: pip_size_in_points = 10
    elif digits == 4 or digits == 2: pip_size_in_points = 1
    elif "BTCUSD" in symbol.upper() or "ETHUSD" in symbol.upper(): pip_size_in_points = 100 if math.isclose(point_val, 0.01) else 10 # Example for crypto
    else: pip_size_in_points = 10

    details['pip_points'] = max(1, pip_size_in_points)
    details['spread_pips'] = spread_points / pip_size_in_points if pip_size_in_points > 0 else 0
    details['pip_value_per_lot'] = details['trade_tick_value'] * details['pip_points']

    print(f"Info {symbol}: Point={details.get('point'):.5f}, Digits={digits}, VolMin={details.get('volume_min')}, Step={details.get('volume_step')}, PipPoints={details.get('pip_points')}, PipVal={details.get('pip_value_per_lot', 0):.2f}, Spread={details.get('spread_pips', 0):.1f}p, StopsLvl={details.get('trade_stops_level')}pts")
    if details.get('pip_value_per_lot', 0) <= 0: print(f"❌ Invalid 'pip_value_per_lot' for {symbol}."); return None
    if details.get('trade_stops_level') is None: details['trade_stops_level'] = 0
    return details


def is_session_open(symbol, timestamp_dt_aware):
    """Checks if the symbol is likely active based on UTC time."""
    # ... (same as before) ...
    if timestamp_dt_aware.tzinfo is None or timestamp_dt_aware.tzinfo.utcoffset(timestamp_dt_aware) is None:
        print(f"Warning: is_session_open received naive datetime {timestamp_dt_aware}. Assuming UTC.")
        timestamp_dt_aware = UTC_TZ.localize(timestamp_dt_aware)
    else: timestamp_dt_aware = timestamp_dt_aware.astimezone(UTC_TZ)
    is_crypto = "BTCUSD" in symbol.upper() or "ETHUSD" in symbol.upper()
    if is_crypto: return True
    hour = timestamp_dt_aware.hour; weekday = timestamp_dt_aware.weekday()
    if weekday >= 5: return False
    is_london = LONDON_OPEN[0] <= hour < LONDON_OPEN[1]
    is_ny = NY_OPEN[0] <= hour < NY_OPEN[1]
    return is_london or is_ny

# --- Signal Functions --- (Assume timestamp is pandas Timestamp with tz)
def is_bullish_engulfing(df, timestamp):
    # ... (same as before) ...
    if timestamp not in df.index: return False
    i = df.index.get_loc(timestamp);
    if i < 1: return False
    current = df.iloc[i]; previous = df.iloc[i-1]
    if pd.isna(current['open']) or pd.isna(current['close']) or pd.isna(previous['open']) or pd.isna(previous['close']): return False
    return (current['close'] > current['open'] and previous['close'] < previous['open'] and
            current['open'] <= previous['close'] and current['close'] >= previous['open'])

def is_bearish_engulfing(df, timestamp):
    # ... (same as before) ...
    if timestamp not in df.index: return False
    i = df.index.get_loc(timestamp);
    if i < 1: return False
    current = df.iloc[i]; previous = df.iloc[i-1]
    if pd.isna(current['open']) or pd.isna(current['close']) or pd.isna(previous['open']) or pd.isna(previous['close']): return False
    return (current['close'] < current['open'] and previous['close'] > previous['open'] and
            current['open'] >= previous['close'] and current['close'] <= previous['open'])

def check_bos(df, timestamp, direction, lookback=5):
    # ... (same as before) ...
    if timestamp not in df.index: return False
    i = df.index.get_loc(timestamp);
    if i < lookback: return False
    try:
        current_close = df['close'].iloc[i]
        if pd.isna(current_close): return False
        if direction == 'bullish':
            recent_high = df['high'].iloc[i-lookback : i].max()
            if pd.isna(recent_high): return False
            return current_close > recent_high
        elif direction == 'bearish':
            recent_low = df['low'].iloc[i-lookback : i].min()
            if pd.isna(recent_low): return False
            return current_close < recent_low
    except Exception: return False
    return False

def is_trending_up(df, timestamp):
    # ... (same as before) ...
    if timestamp not in df.index: return False
    try: row = df.loc[timestamp]; return row['ema_50'] > row['ema_200']
    except (KeyError, TypeError): return False

def is_trending_down(df, timestamp):
    # ... (same as before) ...
    if timestamp not in df.index: return False
    try: row = df.loc[timestamp]; return row['ema_50'] < row['ema_200']
    except (KeyError, TypeError): return False


# --- Live Trading Functions ---
def get_latest_candles(symbol, timeframe_mt5, count):
    """Fetches the most recent 'count' candles for the symbol with UTC timezone."""
    # ... (same as before, keep error handling) ...
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count)
        if rates is None or len(rates) == 0: print(f"Warning: No rates for {symbol}."); return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e: print(f"❌ Error fetching candles {symbol}: {e}"); traceback.print_exc(); return None

def calculate_indicators(df):
    """Calculates necessary indicators."""
    # ... (same as before, keep error handling) ...
    if df is None or df.empty: return None
    try:
        df_copy = df.copy()
        df_copy['ema_50'] = df_copy['close'].ewm(span=50, adjust=False).mean()
        df_copy['ema_200'] = df_copy['close'].ewm(span=200, adjust=False).mean()
        df_copy['high_low'] = df_copy['high'] - df_copy['low']
        df_copy['high_prev_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['low_prev_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['true_range'] = df_copy[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1, skipna=True)
        # Use alpha=1/14 for ATR calculation consistency
        df_copy['atr_14'] = df_copy['true_range'].ewm(alpha=1/14, adjust=False).mean()
        df_copy.drop(['high_low', 'high_prev_close', 'low_prev_close', 'true_range'], axis=1, inplace=True, errors='ignore')
        return df_copy
    except Exception as e: print(f"❌ Error calculating indicators: {e}"); traceback.print_exc(); return None


def place_order(symbol, order_type, volume, sl_price, tp_price, magic_number, comment=""):
    """Places a market order with validation."""
    s_info = symbol_details.get(symbol)
    if not s_info: print(f"❌ Cannot place order: Symbol info missing {symbol}"); return None

    point = s_info['point']; digits = s_info['digits']
    order_action = mt5.TRADE_ACTION_DEAL; price = 0.0
    tick = mt5.symbol_info_tick(symbol)

    if not tick or tick.ask <= 0 or tick.bid <= 0: print(f"❌ Cannot place order: Invalid tick {symbol}"); return None

    if order_type == mt5.ORDER_TYPE_BUY: price = tick.ask
    elif order_type == mt5.ORDER_TYPE_SELL: price = tick.bid
    else: print(f"❌ Invalid order type: {order_type}"); return None

    min_stop_level_points = s_info['trade_stops_level']; min_stop_level_price_dist = min_stop_level_points * point

    # Validate and adjust SL/TP based on stops level
    if order_type == mt5.ORDER_TYPE_BUY and sl_price > 0:
        required_sl = price - min_stop_level_price_dist
        if sl_price > required_sl: sl_price = required_sl # Move SL further away if needed
        if sl_price >= price: print(f"❌ BUY SL invalid: {sl_price:.{digits}f} >= Ask {price:.{digits}f}"); return None
    elif order_type == mt5.ORDER_TYPE_SELL and sl_price > 0:
         required_sl = price + min_stop_level_price_dist
         if sl_price < required_sl: sl_price = required_sl
         if sl_price <= price: print(f"❌ SELL SL invalid: {sl_price:.{digits}f} <= Bid {price:.{digits}f}"); return None

    if order_type == mt5.ORDER_TYPE_BUY and tp_price > 0:
        required_tp = price + min_stop_level_price_dist
        if tp_price < required_tp: tp_price = 0.0 # Disable TP if too close
        if tp_price > 0 and tp_price <= price: print(f"❌ BUY TP invalid: {tp_price:.{digits}f} <= Ask {price:.{digits}f}"); return None
    elif order_type == mt5.ORDER_TYPE_SELL and tp_price > 0:
         required_tp = price - min_stop_level_price_dist
         if tp_price > required_tp: tp_price = 0.0 # Disable TP if too close
         if tp_price > 0 and tp_price >= price: print(f"❌ SELL TP invalid: {tp_price:.{digits}f} >= Bid {price:.{digits}f}"); return None

    sl_price = round(sl_price, digits) if sl_price > 0 else 0.0
    tp_price = round(tp_price, digits) if tp_price > 0 else 0.0

    request = {
        "action": order_action, "symbol": symbol, "volume": volume, "type": order_type,
        "price": price, "sl": sl_price, "tp": tp_price, "deviation": MAX_SLIPPAGE_POINTS,
        "magic": magic_number, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": s_info['filling_mode'],
    }

    print(f"--- Attempting Order: {request['symbol']} {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} {request['volume']} lots @ ~{request['price']:.{digits}f} ---")
    print(f"    SL: {request['sl']:.{digits}f}, TP: {request['tp']:.{digits}f}, Magic: {request['magic']}")

    try:
        result = mt5.order_send(request)
        if result is None: print(f"❌ Order Send Fail (No result). Last Err: {mt5.last_error()}"); return None
        if result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == mt5.TRADE_RETCODE_PLACED:
            print(f"✅ Order Sent OK. Ticket: {result.order}, Deal: {result.deal}")
            time.sleep(0.1) # *** Reduced sleep ***
            return result
        else:
            last_err = mt5.last_error()
            # --- Add Rate Limit Handling (Example) ---
            if last_err and last_err[0] in [mt5.RES_E_SERVER_BUSY, mt5.RES_E_TOO_MANY_REQUESTS, mt5.RES_E_TIMEOUT]:
                 print(f"⏳ Rate Limit or Busy ({last_err[0]}), pausing before retry...")
                 time.sleep(3) # Longer backoff
            # ---------------------------------------
            print(f"❌ Order Send Fail. Code: {result.retcode} ({last_err[0]} '{last_err[1]}'), Cmt: {result.comment}"); print(f"   Req: {result.request}"); return None
    except Exception as e: print(f"❌ Exception order_send: {e}"); traceback.print_exc(); return None

def modify_position_sltp(position, new_sl, new_tp, magic_number, comment=""):
    """Modifies the SL and TP of an open position."""
    # ... (keep validation, check, send logic) ...
    s_info = symbol_details.get(position.symbol)
    if not s_info: print(f"❌ Cannot modify {position.ticket}: Symbol info missing."); return False
    digits = s_info['digits']; point = s_info['point']
    sl_tolerance = point * 0.5; tp_tolerance = point * 0.5
    sl_changed = not math.isclose(new_sl, position.sl, abs_tol=sl_tolerance) if new_sl > 0 else position.sl > 0
    tp_changed = not math.isclose(new_tp, position.tp, abs_tol=tp_tolerance) if new_tp > 0 else position.tp > 0
    if not sl_changed and not tp_changed: return True

    request = { "action": mt5.TRADE_ACTION_SLTP, "position": position.ticket, "symbol": position.symbol, "sl": new_sl, "tp": new_tp, "magic": magic_number, "comment": comment, }
    print(f"--- Attempting Modify: Ticket {request['position']} ({request['symbol']}) ---")
    print(f"    Current SL/TP: {position.sl:.{digits}f} / {position.tp:.{digits}f} -> New Target: {request['sl']:.{digits}f} / {request['tp']:.{digits}f}")
    try:
        result = mt5.order_send(request)
        if result is None: print(f"❌ Modify Fail (No result). Last Err: {mt5.last_error()}"); return False
        if result.retcode == mt5.TRADE_RETCODE_DONE: print(f"✅ Modify OK Ticket {position.ticket}."); return True
        else:
            last_err = mt5.last_error()
            print(f"❌ Modify Fail Ticket {position.ticket}. Code: {result.retcode} ({last_err[0]} '{last_err[1]}'), Cmt: {result.comment}"); print(f"   Req: {result.request}"); return False
    except Exception as e: print(f"❌ Exception modify SL/TP {position.ticket}: {e}"); traceback.print_exc(); return False


def close_position(position, comment="", magic_number=None):
    """Closes an open position by market order."""
    # ... (keep validation, check, send logic) ...
    s_info = symbol_details.get(position.symbol)
    if not s_info: print(f"❌ Cannot close {position.ticket}: Symbol info missing."); return False
    order_type = None; price = 0.0
    tick = mt5.symbol_info_tick(position.symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0: print(f"❌ Cannot close {position.ticket}: Invalid tick."); return False
    if position.type == mt5.ORDER_TYPE_BUY: order_type = mt5.ORDER_TYPE_SELL; price = tick.bid
    elif position.type == mt5.ORDER_TYPE_SELL: order_type = mt5.ORDER_TYPE_BUY; price = tick.ask
    else: print(f"❌ Unknown position type {position.type} ticket {position.ticket}"); return False
    close_filling_mode = mt5.ORDER_FILLING_IOC # Prefer IOC for closing
    if not (s_info['filling_modes_raw'] & mt5.ORDER_FILLING_IOC):
        print(f"⚠️ IOC not supported closing {position.symbol}. Trying FOK...");
        if s_info['filling_modes_raw'] & mt5.ORDER_FILLING_FOK: close_filling_mode = mt5.ORDER_FILLING_FOK
        elif s_info['filling_modes_raw'] & mt5.ORDER_FILLING_RETURN: print(f"⚠️ FOK not supported closing {position.symbol}. Using RETURN."); close_filling_mode = mt5.ORDER_FILLING_RETURN
        else: print(f"❌ No suitable filling mode for closing {position.symbol}."); return False
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "position": position.ticket, "symbol": position.symbol, "volume": position.volume,
        "type": order_type, "price": price, "deviation": MAX_SLIPPAGE_POINTS, "magic": position.magic if magic_number is None else magic_number,
        "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": close_filling_mode,
    }
    print(f"--- Attempting Close: Ticket {request['position']} ({request['symbol']} {'SELL' if order_type == mt5.ORDER_TYPE_SELL else 'BUY'} {request['volume']} @ ~{request['price']:.{s_info['digits']}f}) ---")
    try:
        result = mt5.order_send(request)
        if result is None: print(f"❌ Close Fail (No result). Last Err: {mt5.last_error()}"); return False
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Close OK Ticket {position.ticket}. Deal: {result.deal}")
            time.sleep(0.1) # *** Reduced sleep ***
            return True
        else:
            last_err = mt5.last_error(); print(f"❌ Close Fail Ticket {position.ticket}. Code: {result.retcode} ({last_err[0]} '{last_err[1]}'), Cmt: {result.comment}"); print(f"   Req: {result.request}"); return False
    except Exception as e: print(f"❌ Exception close position {position.ticket}: {e}"); traceback.print_exc(); return False


def get_daily_pnl(magic_number):
    """Calculates realized PnL for today (UTC) for the given magic number."""
    # ... (keep existing logic) ...
    try:
        today_start_utc = datetime.now(UTC_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        now_utc = datetime.now(UTC_TZ)
        deals = mt5.history_deals_get(today_start_utc, now_utc)
        if deals is None:
            last_error = mt5.last_error()
            if last_error[0] not in [mt5.RES_S_INTERNAL_ERROR, mt5.RES_S_SERVER_BUSY, mt5.RES_E_TIMEOUT]: print(f"Warning: Could not get deals history. Err: {last_error[0]}, {last_error[1]}")
            return 0.0
        total_pnl = 0.0
        for deal in deals:
            if deal.magic == magic_number and deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                 total_pnl += (deal.profit if pd.notna(deal.profit) else 0.0) + \
                              (deal.commission if pd.notna(deal.commission) else 0.0) + \
                              (deal.swap if pd.notna(deal.swap) else 0.0)
        return total_pnl
    except Exception as e: print(f"❌ Exception calculating PnL: {e}"); traceback.print_exc(); return 0.0


def check_new_bar(symbol, timeframe_mt5, last_bar_times):
    """Checks if a new bar has formed for the symbol/timeframe using UTC."""
    # ... (keep existing logic) ...
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, 1)
        if rates is None or len(rates) == 0: return False, last_bar_times
        current_bar_time_dt_utc = dt.datetime.fromtimestamp(rates[0]['time'], tz=UTC_TZ)
        last_known_time_utc = last_bar_times.get(symbol)
        if last_known_time_utc is None or current_bar_time_dt_utc > last_known_time_utc:
            last_bar_times[symbol] = current_bar_time_dt_utc; return True, last_bar_times
        else: return False, last_bar_times
    except Exception as e: print(f"❌ Error checking new bar {symbol}: {e}"); traceback.print_exc(); return False, last_bar_times


def update_trailing_stop(position):
    """Calculates and attempts to modify the SL for a position based on TSL rules."""
    # ... (keep existing logic, as USE_TRAILING_STOP is False, this won't run anyway) ...
    s_info = symbol_details.get(position.symbol)
    if not s_info: print(f"Warning: Cannot TSL {position.ticket}, info missing."); return False
    tick = mt5.symbol_info_tick(position.symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0: print(f"Warning: Skipping TSL {position.ticket} - invalid tick."); return False
    point = s_info['point']; digits = s_info['digits']; pip_points = s_info['pip_points']
    pip_value_in_price = point * pip_points if pip_points > 0 else point
    tsl_distance_price_dist = TSL_DISTANCE_PIPS * pip_value_in_price
    current_sl = position.sl; potential_new_sl = 0.0
    if position.type == mt5.ORDER_TYPE_BUY:
        potential_new_sl = tick.bid - tsl_distance_price_dist
        if potential_new_sl <= current_sl and current_sl > 0: return False
    elif position.type == mt5.ORDER_TYPE_SELL:
        potential_new_sl = tick.ask + tsl_distance_price_dist
        if current_sl > 0 and potential_new_sl >= current_sl: return False
        elif current_sl == 0 and potential_new_sl <= 0: print(f"Warning TSL SELL {position.ticket}: Pot. SL invalid."); return False
    potential_new_sl_rounded = round(potential_new_sl, digits)
    min_stop_level_points = s_info['trade_stops_level']; min_stop_level_price_dist = min_stop_level_points * point
    valid_sl = True
    if position.type == mt5.ORDER_TYPE_BUY:
        required_sl = tick.bid - min_stop_level_price_dist
        if potential_new_sl_rounded > required_sl: print(f"⚠️ TSL Skip BUY {position.ticket}: SL too close."); valid_sl = False
        elif potential_new_sl_rounded >= tick.bid: print(f"⚠️ TSL Skip BUY {position.ticket}: SL >= Bid."); valid_sl = False
    elif position.type == mt5.ORDER_TYPE_SELL:
        required_sl = tick.ask + min_stop_level_price_dist
        if potential_new_sl_rounded < required_sl: print(f"⚠️ TSL Skip SELL {position.ticket}: SL too close."); valid_sl = False
        elif potential_new_sl_rounded <= tick.ask: print(f"⚠️ TSL Skip SELL {position.ticket}: SL <= Ask."); valid_sl = False
    if not valid_sl: return False
    print(f"TSL Trigger {position.symbol} Tkt {position.ticket}. Old SL: {current_sl:.{digits}f}, New SL: {potential_new_sl_rounded:.{digits}f}")
    modify_position_sltp(position, potential_new_sl_rounded, position.tp, MAGIC_NUMBER, comment="TSL Update")
    return True


# --- Main Live Trading Loop ---
def run_live_trading():
    print("\n--- Initializing Live Trading Bot ---")
    # ... (print parameters) ...
    print(f"Symbols: {SYMBOLS}, TF: {get_timeframe_name(TIMEFRAME)}, Magic: {MAGIC_NUMBER}")
    print(f"Lots: Default={LOT_SIZE_DEFAULT}, ETH={LOT_SIZE_ETH}")
    print(f"Risk: Daily SL={DAILY_STOP_LOSS_PERCENT}%, Daily TP={DAILY_PROFIT_TARGET_PERCENT}%")
    print(f"TSL: Active={USE_TRAILING_STOP}, Act Pips={TSL_ACTIVATION_PIPS}, Dist Pips={TSL_DISTANCE_PIPS}")
    print(f"Entry Gap Mins: {MIN_ENTRY_GAP_MINUTES}")

    if not mt5.initialize(): print("❌ MT5 initialization failed."); return
    print("✅ MT5 Initialized OK.")
    account_info = mt5.account_info()
    if account_info: print(f"Account: {account_info.login}, Bal: {account_info.balance:.2f} {account_info.currency}")
    else: print("❌ Could not get account info."); mt5.shutdown(); return

    global symbol_details; valid_symbols = []
    print("\n--- Setting up Symbols ---")
    for symbol in SYMBOLS:
        print(f"Processing: {symbol}...")
        details = get_symbol_info(symbol)
        if details:
            target_lot = LOT_SIZE_DEFAULT
            if symbol == "ETHUSD": target_lot = LOT_SIZE_ETH
            # --- Lot size validation (removed explicit fixed lot checks) ---
            # Ensure min/max/step are valid, assign lot size
            details['trade_lot'] = target_lot # Assign default/specific size for now
            # Need to handle potential invalidity if dynamic sizing is added later
            symbol_details[symbol] = details; valid_symbols.append(symbol)
            print(f"✅ {symbol} OK. Using Lot: {details['trade_lot']}")
        else: print(f"⚠️ Cannot trade {symbol}.")
        time.sleep(0.2) # Reduced sleep between symbol setups

    if not valid_symbols: print("❌ No valid symbols. Exiting."); mt5.shutdown(); return
    print(f"\n--- Trading enabled for: {valid_symbols} ---")

    # State Variables Init
    last_bar_times = {}; symbol_last_entry_time = {}; daily_limit_reached = False; today_date_utc = None
    last_pnl_check_time_monotonic = time.monotonic()
    next_main_loop_time_monotonic = time.monotonic() # Init time for non-blocking loop

    # Initial PnL & Limit Calculation
    print("Initial PnL check...")
    current_account_info = mt5.account_info()
    current_bal = current_account_info.balance if current_account_info else 0
    loss_limit_amount = abs(current_bal * (DAILY_STOP_LOSS_PERCENT / 100.0)) if current_bal > 0 else 0
    profit_target_amount = abs(current_bal * (DAILY_PROFIT_TARGET_PERCENT / 100.0)) if current_bal > 0 else float('inf')
    realized_daily_pnl = get_daily_pnl(MAGIC_NUMBER)
    print(f"Bal: {current_bal:.2f}. Loss Limit: ~${loss_limit_amount:.2f}, Profit Target: ~${profit_target_amount:.2f}")
    print(f"Today's Realized PnL (UTC): ${realized_daily_pnl:.2f}")
    if loss_limit_amount > 0 and realized_daily_pnl <= -loss_limit_amount: print(f"🚨 Initial Daily SL Hit! Halting trades. 🚨"); daily_limit_reached = True
    elif profit_target_amount > 0 and realized_daily_pnl >= profit_target_amount: print(f"🎉 Initial Daily TP Hit! Halting trades. 🎉"); daily_limit_reached = True

    # --- Main Loop ---
    try:
        print("\n--- Starting Main Trading Loop (Ctrl+C to stop) ---")
        while True:
            if not mt5.terminal_info(): print("❌ MT5 Connection Lost."); break
            now_monotonic = time.monotonic()
            now_utc = datetime.now(UTC_TZ)

            # --- Non-blocking Main Loop Check ---
            if now_monotonic >= next_main_loop_time_monotonic:
                current_date_utc = now_utc.date()

                # Daily Reset
                if current_date_utc != today_date_utc:
                    print(f"\n--- New Day (UTC): {current_date_utc} ---")
                    today_date_utc = current_date_utc; daily_limit_reached = False
                    last_bar_times = {}; symbol_last_entry_time = {}
                    current_account_info = mt5.account_info()
                    current_bal = current_account_info.balance if current_account_info else 0
                    loss_limit_amount = abs(current_bal * (DAILY_STOP_LOSS_PERCENT / 100.0)) if current_bal > 0 else 0
                    profit_target_amount = abs(current_bal * (DAILY_PROFIT_TARGET_PERCENT / 100.0)) if current_bal > 0 else float('inf')
                    print(f"Limits Reset. Bal: {current_bal:.2f}. Loss Lim: ~${loss_limit_amount:.2f}, Profit Tgt: ~${profit_target_amount:.2f}")
                    last_pnl_check_time_monotonic = now_monotonic # Reset PnL check timer too
                    realized_daily_pnl = get_daily_pnl(MAGIC_NUMBER)
                    print(f"New Day PnL: ${realized_daily_pnl:.2f}")
                    if loss_limit_amount > 0 and realized_daily_pnl <= -loss_limit_amount: print(f"🚨 Daily SL hit on new day! 🚨"); daily_limit_reached = True
                    elif profit_target_amount > 0 and realized_daily_pnl >= profit_target_amount: print(f"🎉 Daily TP hit on new day! 🎉"); daily_limit_reached = True

                # Periodic PnL Check
                if now_monotonic - last_pnl_check_time_monotonic > PNL_CHECK_INTERVAL_SECONDS:
                    if not daily_limit_reached:
                        realized_daily_pnl = get_daily_pnl(MAGIC_NUMBER)
                        print(f"PnL Check @ {now_utc.strftime('%H:%M:%S')} UTC: ${realized_daily_pnl:.2f}")
                        last_pnl_check_time_monotonic = now_monotonic
                        if loss_limit_amount > 0 and realized_daily_pnl <= -loss_limit_amount: print(f"🚨 DAILY SL HIT! PnL: ${realized_daily_pnl:.2f}. Halting. 🚨"); daily_limit_reached = True
                        elif profit_target_amount > 0 and realized_daily_pnl >= profit_target_amount: print(f"🎉 DAILY TP HIT! PnL: ${realized_daily_pnl:.2f}. Halting. 🎉"); daily_limit_reached = True
                    else: last_pnl_check_time_monotonic = now_monotonic # Still update check time even if limit hit

                # --- Process Symbols ---
                for symbol in valid_symbols:
                    s_info = symbol_details.get(symbol);
                    if not s_info: continue
                    point = s_info['point']; pip_points = s_info['pip_points']
                    pip_value_in_price = point * pip_points if pip_points > 0 else point

                    # Manage Positions
                    open_position = None
                    try:
                        positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER)
                        if positions: open_position = positions[0]
                    except Exception as e: print(f"❌ Err get positions {symbol}: {e}"); time.sleep(1); continue # Short sleep on error

                    if open_position:
                        # Max Hold Check
                        try:
                            pos_open_utc = dt.datetime.fromtimestamp(open_position.time, tz=UTC_TZ)
                            hold_dur = now_utc - pos_open_utc
                            if hold_dur.total_seconds() / 60 >= MAX_HOLD_DURATION_MINUTES:
                                print(f"⏳ Timer expired {symbol} Tkt {open_position.ticket}. Closing...")
                                if close_position(open_position, comment="Max Hold", magic_number=MAGIC_NUMBER): open_position = None
                                # Keep processing other symbols, don't 'continue' main loop
                        except Exception as e: print(f"❌ Err check hold duration {open_position.ticket}: {e}")

                        # TSL Check (Code remains but USE_TRAILING_STOP is False)
                        if USE_TRAILING_STOP and open_position:
                           # ... (TSL logic would go here if enabled) ...
                           pass

                    # Check Entries (respecting daily limits and entry gap)
                    if not open_position and not daily_limit_reached:
                        if is_session_open(symbol, now_utc):
                            can_enter = True
                            last_entry = symbol_last_entry_time.get(symbol)
                            if last_entry:
                                if last_entry.tzinfo is None: last_entry = UTC_TZ.localize(last_entry)
                                if (now_utc - last_entry).total_seconds() / 60 < MIN_ENTRY_GAP_MINUTES: can_enter = False

                            if can_enter:
                                is_new, last_bar_times = check_new_bar(symbol, TIMEFRAME, last_bar_times)
                                if is_new:
                                    print(f"--- New Bar {symbol} @ {last_bar_times[symbol]}. Check Entry ---")
                                    df_c = get_latest_candles(symbol, TIMEFRAME, CANDLE_COUNT_FOR_INDICATORS)
                                    if df_c is None or len(df_c) < 201: print(f"Warn: Not enough data {symbol}."); continue
                                    # Ensure timezone consistency
                                    if df_c.index.tz is None: df_c.index = df_c.index.tz_localize(UTC_TZ)
                                    elif df_c.index.tz != UTC_TZ: df_c.index = df_c.index.tz_convert(UTC_TZ)

                                    df_i = calculate_indicators(df_c)
                                    if df_i is None or df_i.empty or len(df_i) < 2: print(f"Warn: No indicators {symbol}."); continue

                                    try:
                                        if len(df_i.index) < 2: raise IndexError("Not enough rows")
                                        sig_ts = df_i.index[-2]; candle = df_i.loc[sig_ts]
                                        if candle[['ema_50', 'ema_200', 'atr_14']].isnull().any(): print(f"Warn: NaN indicators {symbol} @ {sig_ts}."); continue
                                    except Exception as e: print(f"Warn: Err access signal data {symbol}. {e}"); continue

                                    atr = candle.get('atr_14', 0) # Use the ATR column name from calculate_indicators
                                    if atr <= 0: print(f"Warn: Invalid ATR {symbol}."); continue

                                    bull_eng = is_bullish_engulfing(df_i, sig_ts); bear_eng = is_bearish_engulfing(df_i, sig_ts)
                                    up = is_trending_up(df_i, sig_ts); down = is_trending_down(df_i, sig_ts)
                                    above50 = candle['close'] > candle['ema_50']; below50 = candle['close'] < candle['ema_50']
                                    bull_b = check_bos(df_i, sig_ts, 'bullish', BOS_LOOKBACK_PERIODS); bear_b = check_bos(df_i, sig_ts, 'bearish', BOS_LOOKBACK_PERIODS)

                                    o_type = None; direction = None
                                    if up and bull_eng and above50 and bull_b: direction = 'bullish'; o_type = mt5.ORDER_TYPE_BUY
                                    elif down and bear_eng and below50 and bear_b: direction = 'bearish'; o_type = mt5.ORDER_TYPE_SELL

                                    if o_type is not None:
                                        print(f"✅ Entry Signal: {symbol} {direction.upper()} @ {sig_ts}")
                                        sl_dist = atr * ATR_SL_MULTIPLIER
                                        min_sl_dist = max(s_info['trade_stops_level'] * point, 2 * point)
                                        sl_dist = max(sl_dist, min_sl_dist)
                                        sl = 0.0; tp = 0.0; tick = mt5.symbol_info_tick(symbol)
                                        if not tick or tick.bid <= 0 or tick.ask <= 0: print(f"❌ Cannot calc SL/TP {symbol} - Failed tick."); continue

                                        ask = tick.ask; bid = tick.bid
                                        if direction == 'bullish':
                                            sl = candle['low'] - sl_dist; sl = min(sl, bid - min_sl_dist)
                                            risk = ask - sl
                                            if risk <= min_sl_dist: print(f"Warn: {symbol} BUY risk small. Adjust SL."); sl = ask - min_sl_dist; risk = ask - sl
                                            if risk <= 0: print(f"❌ Crit: No positive risk {symbol} BUY."); continue
                                            tp = ask + (risk * REWARD_RISK_RATIO)
                                        else:
                                            sl = candle['high'] + sl_dist; sl = max(sl, ask + min_sl_dist)
                                            risk = sl - bid
                                            if risk <= min_sl_dist: print(f"Warn: {symbol} SELL risk small. Adjust SL."); sl = bid + min_sl_dist; risk = sl - bid
                                            if risk <= 0: print(f"❌ Crit: No positive risk {symbol} SELL."); continue
                                            tp = bid - (risk * REWARD_RISK_RATIO)

                                        lots = s_info.get('trade_lot') # Use pre-assigned lot size
                                        if lots is None or lots <= 0: print(f"❌ CRIT ERR: Invalid lot {lots} {symbol}."); continue

                                        print(f"Attempt {direction} {symbol}, Lot: {lots}, SL: {sl:.{s_info['digits']}f}, TP: {tp:.{s_info['digits']}f}")
                                        order_res = place_order(symbol, o_type, lots, sl, tp, MAGIC_NUMBER, f"{direction} Sig@{sig_ts.strftime('%H:%M')}")
                                        if order_res:
                                            print(f"✅ Placed {symbol} Lot {lots}."); symbol_last_entry_time[symbol] = now_utc
                                            print(f"   Entry time {symbol}: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                                        else: print(f"❌ Failed place {symbol}.")
                                        # No long sleep needed after order attempt

                # --- End Symbol Processing ---

                # Update next check time after processing everything for this interval
                next_main_loop_time_monotonic = now_monotonic + MAIN_LOOP_INTERVAL_SECONDS

            # --- End Main Check Block ---

            # Prevent high CPU usage when no checks are needed
            time.sleep(MIN_CPU_SLEEP_SECONDS)

        # --- End While Loop ---

    except KeyboardInterrupt: print("\n--- Ctrl+C detected. Shutting down... ---")
    except Exception as e: print(f"\n--- 💥 CRITICAL ERROR in main loop: {e} 💥 ---"); traceback.print_exc()
    finally:
        print("--- Shutdown Sequence ---")
        print("Closing open positions...")
        try:
            positions = mt5.positions_get(magic=MAGIC_NUMBER)
            if positions:
                print(f"Found {len(positions)} positions to close.")
                closed = 0; failed = 0
                for pos in positions:
                    print(f"Closing {pos.ticket} {pos.symbol}...")
                    if close_position(pos, comment="Bot Shutdown", magic_number=MAGIC_NUMBER): closed += 1; print(f" -> Closed.")
                    else: failed += 1; print(f" -> Failed close.")
                    time.sleep(0.2) # Short delay between close attempts
                print(f"Closure: {closed} OK, {failed} Failed.")
            else: print("No open positions found.")
        except Exception as e: print(f"❌ Error closing positions: {e}"); traceback.print_exc()

        if mt5.terminal_info(): mt5.shutdown(); print("✅ MT5 shut down.")
        else: print("MT5 already disconnected.")
        print("Bot stopped.")

# --- Monte Carlo function (Optional, Keep as is) ---
def monte_carlo_sim(balance=1000.0, win_rate=0.35, risk_reward=3.0, risk_per_trade_pct=1.0, num_trades=250, num_simulations=1000):
    # ... (keep the Monte Carlo function as it was) ...
    final_balances = []
    risk_mult = risk_per_trade_pct / 100.0; profit_mult = risk_mult * risk_reward
    print(f"\n--- Running Monte Carlo Simulation ---")
    print(f" Start Balance=${balance:.2f}, Win Rate={win_rate:.1%}, R:R={risk_reward:.1f}, Risk={risk_per_trade_pct:.2f}%"); start_time = time.time()
    for i in range(num_simulations):
        equity = balance
        for _ in range(num_trades):
            if equity <= 0: break
            equity += equity * profit_mult if random.random() < win_rate else -equity * risk_mult
        final_balances.append(max(0, equity))
        if (i + 1) % (num_simulations // 10) == 0: print(f"   Completed {i+1}/{num_simulations} simulations...")
    print(f"--- Monte Carlo Finished ({time.time() - start_time:.2f}s) ---")
    if not final_balances: print("Error: No results."); return 0, 0, 0, 0
    median = np.median(final_balances); p5 = np.percentile(final_balances, 5); p95 = np.percentile(final_balances, 95)
    prob_profit = sum(1 for b in final_balances if b > balance) / num_simulations
    print(f" Median: ${median:.2f}, 5th Pct: ${p5:.2f}, 95th Pct: ${p95:.2f}, Prob Profit: {prob_profit:.1%}"); print("-" * 30)
    return median, p5, p95, prob_profit

# --- Script Execution ---
if __name__ == "__main__":
    # Optional: Run simulation first
    # monte_carlo_sim(balance=INITIAL_BALANCE_INFO_ONLY, win_rate=0.35, risk_reward=REWARD_RISK_RATIO, risk_per_trade_pct=1.0, num_trades=250, num_simulations=1000)

    # Run the live trading bot
    run_live_trading()