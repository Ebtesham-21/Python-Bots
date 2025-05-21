import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta 
import numpy as np
from datetime import datetime, timedelta, time as dt_time, timezone # <<< Import timezone
import time as os_time # For sleep

# --- Configuration (Same as your last backtest run) ---
SYMBOLS = ["GBPUSD", "USDCAD"] # Multiple symbols now
ENTRY_TIMEFRAME_STR = "M1"
BOX_TREND_TIMEFRAME_STR = "M15"

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
ENTRY_TIMEFRAME_MT5 = None
BOX_TREND_TIMEFRAME_MT5 = None

EMA_PERIOD = 20 
ATR_PERIOD = 14
BOX_START_HOUR_UTC = 6
BOX_END_HOUR_UTC = 9
TRADE_START_HOUR_UTC = 9
TRADE_END_HOUR_UTC = 15
MIN_BOX_HEIGHT_PIPS = 10.0
MAX_BOX_HEIGHT_PIPS = 50.0
TP_BOX_HEIGHT_MULTIPLIER = 1.5
SL_BUFFER_PIPS = 2.0
RISK_PER_TRADE_PERCENT = 1.0
MAX_DAILY_TRADES_PER_SYMBOL = 2
DAILY_PROFIT_LOCK_PERCENT = 3.0
DAILY_LOSS_LOCK_PERCENT = 2.0
DAILY_MAX_RISK_PERCENT = 1.0
SPREAD_PIPS_ASSUMED = 1.5 
COMMISSION_PER_LOT = 0

# --- Global Variables for Live Bot State ---
pip_value_dict = {}
open_trades_managed_by_bot = [] 

# Daily state variables
daily_box_info = {symbol: {} for symbol in SYMBOLS}
daily_trend_bias = {symbol: "none" for symbol in SYMBOLS}
daily_trades_taken_this_day = {symbol: 0 for symbol in SYMBOLS}
daily_pnl_this_day = {symbol: 0.0 for symbol in SYMBOLS}
daily_trade_lock_active = {symbol: False for symbol in SYMBOLS} 

# Daily Risk Tracking Variables
account_balance_at_day_start = 0 
daily_total_risk_committed = 0.0 

# --- MT5 Connection Functions ---
def initialize_mt5_connection():
    global ENTRY_TIMEFRAME_MT5, BOX_TREND_TIMEFRAME_MT5, account_balance_at_day_start
    entry_tf_mt5_val = TIMEFRAME_MAP.get(ENTRY_TIMEFRAME_STR)
    box_trend_tf_mt5_val = TIMEFRAME_MAP.get(BOX_TREND_TIMEFRAME_STR)
    if not mt5.initialize():
        print(f"{datetime.now()} Initialize failed, error code =", mt5.last_error()); return False
    print(f"{datetime.now()} MetaTrader 5 Initialized Successfully")
    # Set timezone to UTC for MT5 communication
    mt5_timezone = timezone.utc # <--- Explicitly use UTC
    print(f"Using Timezone: {mt5_timezone}")

    account_info = mt5.account_info()
    if account_info:
        print(f"Account: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
        account_balance_at_day_start = account_info.balance 
    else:
        print(f"{datetime.now()} Could not get account info. Error:", mt5.last_error()); mt5.shutdown(); return False
    if entry_tf_mt5_val is None or box_trend_tf_mt5_val is None:
        print(f"{datetime.now()} Error: Invalid timeframe string(s) in config."); mt5.shutdown(); return False
    ENTRY_TIMEFRAME_MT5 = entry_tf_mt5_val; BOX_TREND_TIMEFRAME_MT5 = box_trend_tf_mt5_val
    print(f"Using Entry Timeframe: {ENTRY_TIMEFRAME_STR}, Box/Trend Timeframe: {BOX_TREND_TIMEFRAME_STR}")
    return True

def shutdown_mt5():
    mt5.shutdown(); print(f"{datetime.now()} MetaTrader 5 Shutdown")

# --- Helper Functions ---
def get_pip_size_and_digits(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{datetime.now()} Failed to get symbol info for {symbol}, error code =", mt5.last_error())
        return 0.0001, 5 
    digits = symbol_info.digits
    return (0.01 if "JPY" in symbol else 0.0001), digits

def get_latest_candles(symbol, timeframe_mt5, count):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count)
        if rates is None or len(rates) == 0:
            print(f"{datetime.now()} No rates for {symbol} TF {timeframe_mt5}, error: {mt5.last_error()}"); return pd.DataFrame()
        df = pd.DataFrame(rates); 
        # MT5 timestamp is usually UTC epoch seconds
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True) 
        df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"{datetime.now()} Error getting candles for {symbol}: {e}")
        return pd.DataFrame()


def calculate_indicators_for_tf(df, ema_period=EMA_PERIOD, atr_period=ATR_PERIOD):
    if df.empty or len(df) < max(ema_period, atr_period): return df 
    try:
        df[f'EMA{ema_period}'] = df.ta.ema(length=ema_period)
        df['ATR'] = df.ta.atr(length=atr_period)
    except Exception as e:
        print(f"{datetime.now()} Error calculating indicators: {e}")
        # Return df without indicators if calculation fails
    return df

def get_current_spread(symbol):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            pip_s, _ = get_pip_size_and_digits(symbol)
            spread = round((tick.ask - tick.bid) / pip_s, 1) # Spread in pips
            return spread
        else:
            print(f"{datetime.now()} Could not get tick for {symbol}. Using assumed spread.")
            return SPREAD_PIPS_ASSUMED 
    except Exception as e:
         print(f"{datetime.now()} Error getting spread for {symbol}: {e}")
         return SPREAD_PIPS_ASSUMED

def calculate_lot_size(symbol, account_balance_current, risk_percent, sl_pips):
    if sl_pips <= 0: return 0.0
    try:
        symbol_info = mt5.symbol_info(symbol)
        pip_size, _ = get_pip_size_and_digits(symbol)
        account_info = mt5.account_info()

        if symbol_info is None or account_info is None: return 0.01 # Fallback

        risk_amount = account_balance_current * (risk_percent / 100.0)
        
        # Use order_calc_margin for a more robust way to handle different instruments/currencies if needed
        # Simplified pip value calculation (assuming USD account for non-JPY quote pairs)
        # THIS SECTION REMAINS A POTENTIAL POINT OF FAILURE FOR OTHER ACCOUNT CURRENCIES
        if account_info.currency == "USD":
            if "JPY" in symbol: value_per_pip_per_lot = (pip_size * 100000) / mt5.symbol_info_tick(symbol).bid # Approx calculation for JPY quote
            else: value_per_pip_per_lot = pip_size * 100000 # e.g., 0.0001 * 100000 = 10 USD for EURUSD
        else:
            # Need conversion logic if account currency is not USD
            print(f"Warning: Account currency is {account_info.currency}. Pip value calculation assumes USD.")
            if "JPY" in symbol: value_per_pip_per_lot = 1000.0 * (1/mt5.symbol_info_tick("USDJPY").bid) # Rough USD equivalent
            else: value_per_pip_per_lot = 10.0 # Assume roughly 10 USD equivalent
            
        sl_value_per_lot = sl_pips * value_per_pip_per_lot
        if sl_value_per_lot <= 0: return 0.0

        volume = risk_amount / sl_value_per_lot
        
        # Normalize
        volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
        volume = max(symbol_info.volume_min, volume)
        volume = min(symbol_info.volume_max, volume)
        
        return round(volume, int(abs(np.log10(symbol_info.volume_step))) if symbol_info.volume_step > 0 else 2)

    except Exception as e:
        print(f"{datetime.now()} Error calculating lot size for {symbol}: {e}")
        return 0.01 # Fallback

# --- Strategy Logic ---
def reset_all_daily_variables_live(current_utc_date):
    global account_balance_at_day_start, daily_total_risk_committed
    global daily_box_info, daily_trend_bias, daily_trades_taken_this_day, daily_pnl_this_day, daily_trade_lock_active
    
    current_account_info = mt5.account_info()
    if current_account_info:
        account_balance_at_day_start = current_account_info.balance
    else:
        print(f"{datetime.now()} Could not get account info for daily reset, using previous day's start balance.")
    
    daily_total_risk_committed = 0.0
    print(f"--- Start of Day {current_utc_date} (UTC) --- Balance: {account_balance_at_day_start:.2f} ---")
    
    for symbol in SYMBOLS:
        daily_box_info[symbol] = {}; daily_trend_bias[symbol] = "none"
        daily_trades_taken_this_day[symbol] = 0; daily_pnl_this_day[symbol] = 0.0
        daily_trade_lock_active[symbol] = False

def calculate_london_box_and_trend_live(symbol, pip_s):
    global daily_box_info, daily_trend_bias
    
    now_utc = datetime.now(timezone.utc) # <<< Ensure using UTC now()
    
    # Already calculated for today?
    if daily_box_info[symbol].get('calc_time_utc', now_utc - timedelta(days=1)).date() == now_utc.date():
        return
    # Too early to calculate?
    if now_utc.hour < BOX_END_HOUR_UTC:
        return

    print(f"[{now_utc}] Attempting to calculate box and trend for {symbol} for day {now_utc.date()}")

    candles_to_fetch_m15 = (BOX_END_HOUR_UTC - BOX_START_HOUR_UTC) * 4 + EMA_PERIOD + 5 
    
    m15_data = get_latest_candles(symbol, BOX_TREND_TIMEFRAME_MT5, candles_to_fetch_m15 * 2) 
    if m15_data.empty or len(m15_data) < candles_to_fetch_m15:
        print(f"[{now_utc}] Not enough M15 data for {symbol} to calculate box/trend."); return
    
    m15_data = calculate_indicators_for_tf(m15_data, ema_period=EMA_PERIOD)
    if f'EMA{EMA_PERIOD}' not in m15_data.columns:
        print(f"[{now_utc}] Failed to calculate EMA for {symbol}."); return

    box_start_dt_utc = datetime.combine(now_utc.date(), dt_time(BOX_START_HOUR_UTC, 0), tzinfo=timezone.utc) # <<< Use timezone.utc
    box_end_dt_utc_target = datetime.combine(now_utc.date(), dt_time(BOX_END_HOUR_UTC, 0), tzinfo=timezone.utc) # <<< Use timezone.utc

    session_candles = m15_data[
        (m15_data.index.date == now_utc.date()) &
        (m15_data.index.time >= dt_time(BOX_START_HOUR_UTC, 0)) &
        (m15_data.index.time < dt_time(BOX_END_HOUR_UTC, 0))
    ]
    
    if session_candles.empty or len(session_candles) < ((BOX_END_HOUR_UTC - BOX_START_HOUR_UTC) * 4 * 0.8):
        daily_box_info[symbol] = {'valid': False, 'calc_time_utc': now_utc}
        daily_trend_bias[symbol] = "none"
        print(f"[{now_utc}] Not enough M15 session candles for {symbol}. Found: {len(session_candles)}"); return

    box_high = session_candles['High'].max(); box_low = session_candles['Low'].min()
    box_height_pips = (box_high - box_low) / pip_s
    is_valid_box = MIN_BOX_HEIGHT_PIPS <= box_height_pips <= MAX_BOX_HEIGHT_PIPS
    
    daily_box_info[symbol] = {'high': box_high, 'low': box_low, 'height_pips': box_height_pips, 'valid': is_valid_box, 'calc_time_utc': now_utc}

    trend_candle_start_time = dt_time(BOX_END_HOUR_UTC -1, 45) if BOX_END_HOUR_UTC > 0 else dt_time(23,45)
    target_trend_candle_dt = datetime.combine(now_utc.date(), trend_candle_start_time, tzinfo=timezone.utc) # <<< Use timezone.utc

    if target_trend_candle_dt in m15_data.index:
        trend_candle = m15_data.loc[target_trend_candle_dt]; ema_val = trend_candle.get(f'EMA{EMA_PERIOD}')
        if pd.notna(ema_val):
            if trend_candle['Close'] > ema_val: daily_trend_bias[symbol] = "bullish"
            elif trend_candle['Close'] < ema_val: daily_trend_bias[symbol] = "bearish"
            else: daily_trend_bias[symbol] = "none"
        else:
            daily_trend_bias[symbol] = "none"; daily_box_info[symbol]['valid'] = False
            print(f"[{now_utc}] EMA not available for trend calc for {symbol} at {target_trend_candle_dt}")
    else:
        daily_trend_bias[symbol] = "none"; daily_box_info[symbol]['valid'] = False
        print(f"[{now_utc}] Trend candle not found for {symbol} at {target_trend_candle_dt}.")

    if daily_box_info[symbol]['valid']: print(f"[{now_utc}] {symbol} Box: {box_low:.5f}-{box_high:.5f} ({box_height_pips:.1f} pips). Trend: {daily_trend_bias[symbol]}. Valid: {is_valid_box}")
    else: print(f"[{now_utc}] {symbol} Box Invalid. Height: {box_height_pips if 'height_pips' in daily_box_info[symbol] else 'N/A'} pips. Trend: {daily_trend_bias[symbol]}.")


def check_and_manage_positions():
    global open_trades_managed_by_bot
    
    trades_to_remove_from_managed_list = []
    try:
        current_positions = mt5.positions_get()
        if current_positions is None:
            print(f"{datetime.now(timezone.utc)}: Could not get positions, error {mt5.last_error()}"); return
        active_broker_tickets = {pos.ticket for pos in current_positions}
        for i, managed_trade in enumerate(open_trades_managed_by_bot):
            if managed_trade['ticket'] not in active_broker_tickets:
                print(f"{datetime.now(timezone.utc)} Trade {managed_trade['ticket']} for {managed_trade['symbol']} no longer active on broker. Removing from managed list.")
                trades_to_remove_from_managed_list.append(i)
    except Exception as e:
        print(f"{datetime.now(timezone.utc)} Error checking positions: {e}")
        return # Avoid modifying list if check failed

    for idx in sorted(trades_to_remove_from_managed_list, reverse=True):
        try:
            del open_trades_managed_by_bot[idx]
        except IndexError:
             print(f"{datetime.now(timezone.utc)} Error removing trade at index {idx}, list may have changed.")


# --- Main Live Trading Loop ---
def live_trading_loop():
    global account_balance_at_day_start, daily_total_risk_committed
    global daily_box_info, daily_trend_bias, daily_trades_taken_this_day, daily_pnl_this_day, daily_trade_lock_active
    
    if not initialize_mt5_connection(): return

    for symbol in SYMBOLS: 
        pip_s, _ = get_pip_size_and_digits(symbol)
        if pip_s == 0: print(f"Could not get pip size for {symbol}, skipping it."); continue
        pip_value_dict[symbol] = pip_s
    
    if not pip_value_dict: print("Pip sizes not determined for any symbol. Exiting."); shutdown_mt5(); return

    last_checked_minute = -1
    last_processed_day_live = None

    while True: 
        try: # Add main loop error handling
            now_utc = datetime.now(timezone.utc) # <<< Use UTC time directly
            current_utc_date = now_utc.date()

            # --- Daily Reset Logic ---
            if last_processed_day_live != current_utc_date:
                account_info = mt5.account_info() 
                current_bal = account_info.balance if account_info else account_balance_at_day_start 
                reset_all_daily_variables_live(current_bal)
                last_processed_day_live = current_utc_date
                for symbol_calc in SYMBOLS:
                    if symbol_calc in pip_value_dict: calculate_london_box_and_trend_live(symbol_calc, pip_value_dict[symbol_calc])
            
            # --- Recalculate Box/Trend if needed (e.g., bot started mid-session) ---
            if now_utc.hour >= BOX_END_HOUR_UTC:
                for symbol_check in SYMBOLS:
                    if symbol_check in daily_box_info and (not daily_box_info[symbol_check] or daily_box_info[symbol_check].get('calc_time_utc').date() != now_utc.date()):
                        if symbol_check in pip_value_dict:
                            print(f"[{now_utc}] Recalculating box/trend for {symbol_check} as it might have been missed.")
                            calculate_london_box_and_trend_live(symbol_check, pip_value_dict[symbol_check])

            # --- Main Logic: Check for signals ---
            if now_utc.minute != last_checked_minute:
                last_checked_minute = now_utc.minute
                print(f"\n[{now_utc}] Checking for signals...")
                check_and_manage_positions() 

                current_account_info = mt5.account_info() 
                if not current_account_info:
                    print(f"{datetime.now(timezone.utc)} Failed to get account info for trade checks. Skipping cycle.")
                    os_time.sleep(10); continue
                current_balance_for_risk_calc = current_account_info.balance

                for symbol in SYMBOLS:
                    if symbol not in pip_value_dict: continue 
                    pip_size, digits = get_pip_size_and_digits(symbol)

                    if daily_trade_lock_active[symbol]: continue
                    
                    current_symbol_daily_pnl_percent = (daily_pnl_this_day[symbol] / account_balance_at_day_start) * 100 if account_balance_at_day_start > 0 else 0
                    if current_symbol_daily_pnl_percent >= DAILY_PROFIT_LOCK_PERCENT or current_symbol_daily_pnl_percent <= -DAILY_LOSS_LOCK_PERCENT:
                        daily_trade_lock_active[symbol] = True
                        print(f"[{now_utc}] {symbol} Daily P/L lock for symbol engaged. PNL: {current_symbol_daily_pnl_percent:.2f}%")
                        continue
                    
                    if not (TRADE_START_HOUR_UTC <= now_utc.hour < TRADE_END_HOUR_UTC): continue

                    box = daily_box_info.get(symbol); trend = daily_trend_bias.get(symbol)

                    if not box or not box.get('valid') or trend == "none": continue
                    
                    if daily_trades_taken_this_day[symbol] >= MAX_DAILY_TRADES_PER_SYMBOL: continue
                    
                    has_open_trade_for_symbol = any(t['symbol'] == symbol for t in open_trades_managed_by_bot)
                    if has_open_trade_for_symbol: continue

                    m1_candles = get_latest_candles(symbol, ENTRY_TIMEFRAME_MT5, 2) 
                    if len(m1_candles) < 2: continue
                    
                    current_entry_candle = m1_candles.iloc[-1]; prev_entry_candle = m1_candles.iloc[-2]

                    entry_price_raw = 0; sl_price = 0; tp_price = 0; trade_type = None; order_type = None; tick = None

                    try: # Get tick info within try block
                        tick = mt5.symbol_info_tick(symbol)
                        if not tick:
                             print(f"[{now_utc}] Failed to get tick for {symbol}. Skipping signal check.")
                             continue
                    except Exception as e:
                        print(f"[{now_utc}] Error getting tick for {symbol}: {e}")
                        continue

                    if trend == "bullish":
                        if current_entry_candle['Close'] > box['high'] and prev_entry_candle['Close'] <= box['high']:
                            entry_price_raw = tick.ask 
                            sl_price = box['low'] - (SL_BUFFER_PIPS * pip_size)
                            tp_price = entry_price_raw + (box['height_pips'] * TP_BOX_HEIGHT_MULTIPLIER * pip_size)
                            trade_type = 'long'; order_type = mt5.ORDER_TYPE_BUY
                    elif trend == "bearish":
                        if current_entry_candle['Close'] < box['low'] and prev_entry_candle['Close'] >= box['low']:
                            entry_price_raw = tick.bid 
                            sl_price = box['high'] + (SL_BUFFER_PIPS * pip_size)
                            tp_price = entry_price_raw - (box['height_pips'] * TP_BOX_HEIGHT_MULTIPLIER * pip_size)
                            trade_type = 'short'; order_type = mt5.ORDER_TYPE_SELL
                    
                    if trade_type:
                        print(f"[{now_utc}] Potential {trade_type} signal for {symbol} at {entry_price_raw:.{digits}f}")
                        sl_pips_calc = abs(entry_price_raw - sl_price) / pip_size 
                        if sl_pips_calc <= 0.5: 
                            print(f"[{now_utc}] {symbol} SL pips too small ({sl_pips_calc:.1f}). Skipping."); continue

                        risk_amount_for_this_trade = current_balance_for_risk_calc * (RISK_PER_TRADE_PERCENT / 100.0)
                        max_total_daily_risk_value = account_balance_at_day_start * (DAILY_MAX_RISK_PERCENT / 100.0)

                        if daily_total_risk_committed + risk_amount_for_this_trade > max_total_daily_risk_value:
                            print(f"[{now_utc}] PORTFOLIO Daily Max Risk Limit ({DAILY_MAX_RISK_PERCENT}%) would be exceeded for {symbol}. Committed: {daily_total_risk_committed:.2f}, Trade risk: {risk_amount_for_this_trade:.2f}, Limit: {max_total_daily_risk_value:.2f}. Skipping.")
                            continue
                        
                        lot_size = calculate_lot_size(symbol, current_balance_for_risk_calc, RISK_PER_TRADE_PERCENT, sl_pips_calc)
                        if lot_size == 0.0:
                            print(f"[{now_utc}] {symbol} Calculated lot size is 0. Skipping trade."); continue

                        sl_price = round(sl_price, digits); tp_price = round(tp_price, digits)

                        print(f"[{now_utc}] Attempting to send {trade_type} order for {symbol}: Lots: {lot_size}, Entry (approx): {entry_price_raw:.{digits}f}, SL: {sl_price:.{digits}f}, TP: {tp_price:.{digits}f}")

                        request = {
                            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size, "type": order_type,
                            "price": tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid, # Use appropriate price for DEAL
                            "sl": sl_price, "tp": tp_price, "deviation": 10, "magic": 202401, 
                            "comment": "LondonBreakoutBot_v1", "type_time": mt5.ORDER_TIME_GTC, 
                            "type_filling": mt5.ORDER_FILLING_IOC, 
                        }
                        
                        order_result = mt5.order_send(request)

                        if order_result is None:
                            print(f"[{now_utc}] {symbol} order_send failed, error code: {mt5.last_error()}")
                        elif order_result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"[{now_utc}] {symbol} {trade_type} Order SENT successfully. Ticket: {order_result.order}")
                            daily_trades_taken_this_day[symbol] += 1
                            daily_total_risk_committed += risk_amount_for_this_trade 
                            open_trades_managed_by_bot.append({
                                'ticket': order_result.order, 'symbol': symbol, 'type': trade_type,
                                'entry_price': order_result.price, 'sl': sl_price, 'tp': tp_price,
                                'lot_size': lot_size, 'initial_risk_amount': risk_amount_for_this_trade,
                                'entry_time_utc': now_utc
                            })
                            print(f"[{now_utc}] {symbol} OPENED. Day Risk Used: {daily_total_risk_committed:.2f}/{max_total_daily_risk_value:.2f}")
                        else:
                            print(f"[{now_utc}] {symbol} Order FAILED, retcode={order_result.retcode}, comment={order_result.comment}")
                            print("Request details:", request); print("Result details:", order_result)
            
            # End of minute check block
            os_time.sleep(5) # Basic sleep after checks

        except Exception as e: # Catch errors in main loop
             print(f"{datetime.now(timezone.utc)} Error in main loop: {e}")
             print("Attempting to continue...")
             os_time.sleep(30) # Longer sleep after an error

# --- Main Execution ---
if __name__ == "__main__":
    try:
        live_trading_loop()
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    except Exception as e:
        print(f"An unrecoverable critical error occurred: {e}")
    finally:
        shutdown_mt5()