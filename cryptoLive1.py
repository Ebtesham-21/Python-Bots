import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone
import pytz # Still useful for server time awareness
import time
import traceback # For better error reporting

# --- Configuration ---
SYMBOLS_TO_TRADE = ["BTCUSD", "ETHUSD", "BTCXAU"] # Ensure these match Exness symbols
# SYMBOLS_TO_TRADE = ["XAUUSD", "EURUSD"]
# SYMBOLS_TO_TRADE = ["BTCUSD"]

TIMEFRAME_STR = "M30"
MAGIC_NUMBER = 123456 # CHOOSE A UNIQUE MAGIC NUMBER FOR THIS BOT

# --- Risk Management Configuration ---
RISK_PERCENT_PER_TRADE = 1.0 # Risk 1% OF CURRENT TOTAL ACCOUNT BALANCE per trade
DAILY_TOTAL_RISK_LIMIT_PERCENT = 2.0 # Maximum total risk for the day, as a % of balance at UTC day start

# Strategy Parameters
EMA_SHORT_LEN = 5
EMA_MEDIUM_LEN = 8
EMA_LONG_LEN = 13
RISK_REWARD_RATIO = 1.5
SL_BUFFER_PIPS = 5 # Multiplied by symbol_info.point

USE_ALTERNATIVE_EXIT = True

# --- MT5 Timeframe Mapping ---
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
}
MT5_TIMEFRAME = None

# --- Global Variables ---
INITIALIZED_SUCCESSFULLY = False
LAST_BAR_TIMES = {} # To track new bars for each symbol

# Daily risk management globals
balance_at_day_start_utc = 0.0
daily_risk_limit_amount_for_day = 0.0 # Max risk amount in account currency for the current UTC day
cumulative_risk_taken_today = 0.0 # Sum of (balance * RISK_PERCENT_PER_TRADE / 100) for trades taken today
current_trading_day_date = None # Stores the date (UTC) for which daily limits are set


# --- Helper function to determine volume precision for display ---
def get_volume_display_precision(volume_step):
    if volume_step == 0: # Should not happen for valid symbols, defensive
        return 8 # Default to a high precision if step is somehow 0
    # Convert to string, format to avoid float artifacts, then count decimal places
    # Using a reasonably high fixed precision for formatting before stripping zeros
    s_volume_step = "{:.8f}".format(volume_step).rstrip('0')
    if '.' in s_volume_step:
        # Get the part after the decimal point
        decimal_part = s_volume_step.split('.')[-1]
        return len(decimal_part)
    return 0 # Integer step (e.g., volume_step is 1.0, formatted as "1.", so 0 decimal places)


# --- MT5 Connection Functions ---
def initialize_mt5_connection():
    global MT5_TIMEFRAME, INITIALIZED_SUCCESSFULLY
    mt5_tf_val = TIMEFRAME_MAP.get(TIMEFRAME_STR)
    if not mt5_tf_val:
        print(f"Error: Invalid timeframe string '{TIMEFRAME_STR}' in config.")
        return False

    MT5_TIMEFRAME = mt5_tf_val

    if not mt5.initialize():
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False

    account_info = mt5.account_info()
    if account_info is None:
        print(f"Failed to get account info, error code = {mt5.last_error()}")
        mt5.shutdown()
        return False

    print(f"Connected to account: {account_info.login}, Server: {account_info.server}")
    print(f"Initial Total Balance: {account_info.balance:.2f} {account_info.currency}")
    print(f"Using Timeframe: {TIMEFRAME_STR}")
    INITIALIZED_SUCCESSFULLY = True
    return True

def shutdown_mt5():
    global INITIALIZED_SUCCESSFULLY
    if INITIALIZED_SUCCESSFULLY:
        mt5.shutdown()
        print("MetaTrader 5 Shutdown.")
        INITIALIZED_SUCCESSFULLY = False

# --- Data and Indicator Functions ---
def get_latest_bars(symbol, timeframe_mt5, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    return df

def calculate_indicators(df):
    if df.empty or len(df) < EMA_LONG_LEN:
        return pd.DataFrame()
    df.ta.ema(length=EMA_SHORT_LEN, append=True, col_names=(f'EMA_{EMA_SHORT_LEN}',))
    df.ta.ema(length=EMA_MEDIUM_LEN, append=True, col_names=(f'EMA_{EMA_MEDIUM_LEN}',))
    df.ta.ema(length=EMA_LONG_LEN, append=True, col_names=(f'EMA_{EMA_LONG_LEN}',))
    df.dropna(inplace=True)
    return df

# --- Position Sizing ---
def calculate_lot_size(balance_for_risk_calc, risk_percent, sl_distance_price_units, symbol_info):
    if sl_distance_price_units <= 1e-9: return 0.0
    if balance_for_risk_calc <= 0 :
        print(f"Warning: Balance for risk calculation ({balance_for_risk_calc:.2f}) is zero or negative. Cannot calculate lot size.")
        return 0.0

    risk_amount_account_currency = balance_for_risk_calc * (risk_percent / 100.0)
    
    if symbol_info.trade_tick_size == 0: # Should be point for FOREX/CFD, but trade_tick_size is used in MT5 docs
        print(f"Warning: symbol_info.trade_tick_size for {symbol_info.name} is zero. Cannot calculate value per unit.")
        return symbol_info.volume_min if risk_amount_account_currency > 0 else 0.0

    value_per_full_price_unit_per_lot = symbol_info.trade_tick_value / symbol_info.trade_tick_size
    if abs(value_per_full_price_unit_per_lot) < 1e-9:
        print(f"Warning: value_per_full_price_unit_per_lot for {symbol_info.name} is near zero ({value_per_full_price_unit_per_lot}). Using min volume if risk amount allows.")
        return symbol_info.volume_min if risk_amount_account_currency > 0 else 0.0
    
    sl_value_per_lot = sl_distance_price_units * value_per_full_price_unit_per_lot
    if abs(sl_value_per_lot) < 1e-9:
        print(f"Warning: sl_value_per_lot for {symbol_info.name} is near zero ({sl_value_per_lot}). Cannot calculate lot size effectively.")
        return 0.0
    
    raw_lot_size = risk_amount_account_currency / sl_value_per_lot
    
    if symbol_info.volume_step == 0: # Highly unlikely for a tradable symbol
        print(f"Warning: symbol_info.volume_step for {symbol_info.name} is zero. Rounding lot size to 8 decimal places.")
        rounded_lot_size = round(raw_lot_size, 8) 
    else:
        # Ensure lot size is a multiple of volume_step
        rounded_lot_size = (raw_lot_size // symbol_info.volume_step) * symbol_info.volume_step
        # Minor adjustment for floating point inaccuracies, round to precision of volume_step
        precision = get_volume_display_precision(symbol_info.volume_step)
        rounded_lot_size = round(rounded_lot_size, precision)

    final_lot_size = max(symbol_info.volume_min, rounded_lot_size)
    final_lot_size = min(symbol_info.volume_max, final_lot_size)
    
    if final_lot_size < symbol_info.volume_min:
        return 0.0 
    return final_lot_size

# --- Trade Execution ---
def place_trade(symbol, trade_type, lot_size, sl_price, tp_price, symbol_info):
    if trade_type == mt5.ORDER_TYPE_BUY:
        trade_type_str = "BUY"
    elif trade_type == mt5.ORDER_TYPE_SELL:
        trade_type_str = "SELL"
    else:
        trade_type_str = "UNKNOWN_TYPE"

    print(f"\nAttempting to place {trade_type_str} trade for {symbol}...")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick for {symbol}. Error: {mt5.last_error()}")
        return None

    price = tick.ask if trade_type == mt5.ORDER_TYPE_BUY else tick.bid
    deviation = 20

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size, # MT5 expects float here, will handle precision
        "type": trade_type,
        "price": price,
        "sl": round(sl_price, symbol_info.digits),
        "tp": round(tp_price, symbol_info.digits),
        "deviation": deviation,
        "magic": MAGIC_NUMBER,
        "comment": "EMA_Strategy_Bot_TotalBalRisk",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # Or mt5.ORDER_FILLING_FOK
    }

    result = mt5.order_send(request)
    if result is None:
        print(f"Order send failed for {symbol}, no result object. Error: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order send failed for {symbol}, retcode: {result.retcode} - {result.comment}")
        print(f"Request: Price={request['price']}, SL={request['sl']}, TP={request['tp']}, Volume={request['volume']}")
        print(f"Result details: {result}")
        return None
    
    lot_display_precision = get_volume_display_precision(symbol_info.volume_step)
    print(f"Trade {result.order} Placed for {symbol}: {trade_type_str} @ {result.price:.{symbol_info.digits}f}, SL: {sl_price:.{symbol_info.digits}f}, TP: {tp_price:.{symbol_info.digits}f}, Lots: {result.volume:.{lot_display_precision}f}")
    return result

def close_trade(position, symbol_info):
    print(f"\nAttempting to close position {position.ticket} for {symbol_info.name}...")
    tick = mt5.symbol_info_tick(symbol_info.name)
    if tick is None:
        print(f"Failed to get tick for {symbol_info.name} to close trade. Error: {mt5.last_error()}")
        return False

    close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
    trade_type_close = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    deviation = 20

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_info.name,
        "volume": position.volume,
        "type": trade_type_close,
        "position": position.ticket,
        "price": close_price,
        "deviation": deviation,
        "magic": MAGIC_NUMBER,
        "comment": "Alt Exit EMA Cross TotalBalRisk",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # Or mt5.ORDER_FILLING_FOK
    }
    result = mt5.order_send(request)
    if result is None:
        print(f"Close order send failed for position {position.ticket}, no result object. Error: {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close order send failed for position {position.ticket}, retcode: {result.retcode} - {result.comment}")
        print(f"Result details: {result}")
        return False
    
    print(f"Position {position.ticket} closed for {symbol_info.name} at {result.price:.{symbol_info.digits}f}")
    return True

# --- Main Trading Logic ---
def check_and_trade(symbol, symbol_info, current_total_balance_for_trade_calc):
    global LAST_BAR_TIMES, cumulative_risk_taken_today, daily_risk_limit_amount_for_day

    ema_s_col = f'EMA_{EMA_SHORT_LEN}'
    ema_m_col = f'EMA_{EMA_MEDIUM_LEN}'
    ema_l_col = f'EMA_{EMA_LONG_LEN}'
    sl_buffer_actual_price = SL_BUFFER_PIPS * symbol_info.point

    num_hist_bars = EMA_LONG_LEN + 5
    df_hist = get_latest_bars(symbol, MT5_TIMEFRAME, num_hist_bars)
    if df_hist is None or df_hist.empty or len(df_hist) < num_hist_bars -2: # Need at least 3 bars for prev_bar_2
        print(f"Not enough historical data for {symbol} (got {len(df_hist) if df_hist is not None else 0}, need > {EMA_LONG_LEN + 2}).")
        return

    current_bar_time = df_hist.index[-1]
    if symbol not in LAST_BAR_TIMES:
        LAST_BAR_TIMES[symbol] = current_bar_time
        print(f"Initialized last bar time for {symbol}: {current_bar_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return
    
    if current_bar_time <= LAST_BAR_TIMES[symbol]:
        return # Not a new bar
    
    LAST_BAR_TIMES[symbol] = current_bar_time
    print(f"New bar detected for {symbol} at {current_bar_time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Processing...")

    df_with_indicators = calculate_indicators(df_hist.copy())
    if df_with_indicators.empty or len(df_with_indicators) < 3: # Need at least 3 rows for prev_bar and prev_bar_2
        print(f"Not enough data after indicator calculation for {symbol} (got {len(df_with_indicators)}, need >= 3).")
        return

    prev_bar = df_with_indicators.iloc[-2]
    prev_bar_2 = df_with_indicators.iloc[-3]

    open_positions = mt5.positions_get(symbol=symbol)
    bot_position = None
    if open_positions:
        for pos in open_positions:
            if pos.magic == MAGIC_NUMBER:
                bot_position = pos
                break
    
    if bot_position:
        if USE_ALTERNATIVE_EXIT:
            exit_signal = False
            if bot_position.type == mt5.ORDER_TYPE_BUY: # Buying position, look for sell signal to exit
                if prev_bar[ema_s_col] < prev_bar[ema_m_col] and prev_bar_2[ema_s_col] >= prev_bar_2[ema_m_col]: # EMA Short crosses below Medium
                    print(f"Alt exit for BUY on {symbol}: EMA Short ({prev_bar[ema_s_col]:.5f}) crossed below Medium ({prev_bar[ema_m_col]:.5f}).")
                    exit_signal = True
            elif bot_position.type == mt5.ORDER_TYPE_SELL: # Selling position, look for buy signal to exit
                if prev_bar[ema_s_col] > prev_bar[ema_m_col] and prev_bar_2[ema_s_col] <= prev_bar_2[ema_m_col]: # EMA Short crosses above Medium
                    print(f"Alt exit for SELL on {symbol}: EMA Short ({prev_bar[ema_s_col]:.5f}) crossed above Medium ({prev_bar[ema_m_col]:.5f}).")
                    exit_signal = True
            
            if exit_signal:
                close_trade(bot_position, symbol_info)
                return # Exited a trade, don't look for new entry on this bar
        return # Has position, no alt exit, so do nothing more on this bar

    # No existing bot position, look for entry
    if bot_position is None:
        trade_type = None
        stop_loss_price = None
        take_profit_price = None
        risk_price_diff_abs = 0
        
        current_tick = mt5.symbol_info_tick(symbol)
        if not current_tick:
            print(f"Could not get current tick for {symbol} to evaluate entry.")
            return

        # Long Entry Condition
        long_ema_cross = prev_bar[ema_s_col] > prev_bar[ema_m_col] and prev_bar_2[ema_s_col] <= prev_bar_2[ema_m_col]
        long_price_above_emas = prev_bar['Close'] > prev_bar[ema_s_col] and prev_bar['Close'] > prev_bar[ema_m_col] and prev_bar['Close'] > prev_bar[ema_l_col]

        if long_ema_cross and long_price_above_emas:
            trade_type_to_place = mt5.ORDER_TYPE_BUY
            sl_candidate = prev_bar['Low'] - sl_buffer_actual_price
            if sl_candidate >= current_tick.bid : # SL too close or above current bid
                print(f"Long SL candidate {sl_candidate:.{symbol_info.digits}f} is too close or above current bid {current_tick.bid:.{symbol_info.digits}f}. Widening SL attempt or skipping.")
                sl_candidate = current_tick.bid - sl_buffer_actual_price - (symbol_info.point * 10) # Wider buffer
                if sl_candidate >= current_tick.bid:
                    print(f"Still too close for BUY on {symbol} after adjustment. Skipping trade.")
                    return

            risk_price_diff_abs = current_tick.ask - sl_candidate
            if risk_price_diff_abs > (symbol_info.point * 2): # Ensure SL is not extremely tight (e.g. at least 2 points away)
                stop_loss_price = sl_candidate
                take_profit_price = current_tick.ask + (risk_price_diff_abs * RISK_REWARD_RATIO)
                trade_type = trade_type_to_place
            else:
                print(f"Calculated risk distance too small for BUY on {symbol}. Diff: {risk_price_diff_abs:.{symbol_info.digits}f}")

        # Short Entry Condition (only if no long signal)
        short_ema_cross = prev_bar[ema_s_col] < prev_bar[ema_m_col] and prev_bar_2[ema_s_col] >= prev_bar_2[ema_m_col]
        short_price_below_emas = prev_bar['Close'] < prev_bar[ema_s_col] and prev_bar['Close'] < prev_bar[ema_m_col] and prev_bar['Close'] < prev_bar[ema_l_col]

        if trade_type is None and short_ema_cross and short_price_below_emas :
            trade_type_to_place = mt5.ORDER_TYPE_SELL
            sl_candidate = prev_bar['High'] + sl_buffer_actual_price
            if sl_candidate <= current_tick.ask : # SL too close or below current ask
                print(f"Short SL candidate {sl_candidate:.{symbol_info.digits}f} is too close or below current ask {current_tick.ask:.{symbol_info.digits}f}. Widening SL attempt or skipping.")
                sl_candidate = current_tick.ask + sl_buffer_actual_price + (symbol_info.point * 10) # Wider buffer
                if sl_candidate <= current_tick.ask:
                    print(f"Still too close for SELL on {symbol} after adjustment. Skipping trade.")
                    return
            
            risk_price_diff_abs = sl_candidate - current_tick.bid
            if risk_price_diff_abs > (symbol_info.point * 2): # Ensure SL is not extremely tight
                stop_loss_price = sl_candidate
                take_profit_price = current_tick.bid - (risk_price_diff_abs * RISK_REWARD_RATIO)
                trade_type = trade_type_to_place
            else:
                print(f"Calculated risk distance too small for SELL on {symbol}. Diff: {risk_price_diff_abs:.{symbol_info.digits}f}")

        if trade_type is not None and stop_loss_price is not None and take_profit_price is not None:
            if current_total_balance_for_trade_calc <= 0:
                print(f"Current total balance ({current_total_balance_for_trade_calc:.2f}) is zero or less. Cannot place trade for {symbol}.")
                return

            risk_amount_for_this_trade = current_total_balance_for_trade_calc * (RISK_PERCENT_PER_TRADE / 100.0)

            if (cumulative_risk_taken_today + risk_amount_for_this_trade) > (daily_risk_limit_amount_for_day + 1e-5): # Epsilon for float compare
                print(f"Daily risk limit would be EXCEEDED for {symbol}. "
                      f"Cumulative risk today: {cumulative_risk_taken_today:.2f}, "
                      f"Potential for this trade: {risk_amount_for_this_trade:.2f}, "
                      f"Daily Limit: {daily_risk_limit_amount_for_day:.2f}. Skipping trade.")
                return
            
            if risk_amount_for_this_trade <= 0:
                print(f"Calculated risk amount for this trade is {risk_amount_for_this_trade:.2f}. Skipping trade for {symbol}.")
                return

            calculated_lot = calculate_lot_size(current_total_balance_for_trade_calc,
                                                RISK_PERCENT_PER_TRADE,
                                                risk_price_diff_abs, symbol_info)
            
            lot_display_precision = get_volume_display_precision(symbol_info.volume_step) # For display

            if calculated_lot >= symbol_info.volume_min:
                trade_type_str = "BUY" if trade_type == mt5.ORDER_TYPE_BUY else "SELL"
                print(f"Signal for {symbol}: {trade_type_str}, SL: {stop_loss_price:.{symbol_info.digits}f}, TP: {take_profit_price:.{symbol_info.digits}f}, Potential Lots: {calculated_lot:.{lot_display_precision}f}")
                print(f"  Risking {RISK_PERCENT_PER_TRADE}% of current total balance ({current_total_balance_for_trade_calc:.2f}) = {risk_amount_for_this_trade:.2f} for this trade.")
                print(f"  Cumulative daily risk if trade taken: {cumulative_risk_taken_today + risk_amount_for_this_trade:.2f} / {daily_risk_limit_amount_for_day:.2f}")
                
                trade_result = place_trade(symbol, trade_type, calculated_lot, stop_loss_price, take_profit_price, symbol_info)
                if trade_result and trade_result.retcode == mt5.TRADE_RETCODE_DONE:
                    cumulative_risk_taken_today += risk_amount_for_this_trade 
                    print(f"  Trade successful. Updated cumulative risk taken today: {cumulative_risk_taken_today:.2f}")
            else:
                print(f"Calculated lot size {calculated_lot:.{lot_display_precision}f} is less than min volume {symbol_info.volume_min} for {symbol} (or zero due to risk constraints).")
                print(f"  (Based on risking {RISK_PERCENT_PER_TRADE}% of {current_total_balance_for_trade_calc:.2f} = {risk_amount_for_this_trade:.2f}) Skipping trade.")


# --- Main Execution Loop ---
if __name__ == "__main__":
    if not initialize_mt5_connection():
        exit()

    for sym in SYMBOLS_TO_TRADE:
        symbol_info_check = mt5.symbol_info(sym)
        if symbol_info_check is None:
            print(f"Symbol {sym} not found by MT5. Check spelling or broker availability. Exiting.")
            shutdown_mt5()
            exit()
        if not symbol_info_check.visible:
            print(f"Symbol {sym} not visible, enabling...")
            if not mt5.symbol_select(sym, True):
                print(f"Failed to enable {sym}. Error: {mt5.last_error()}. Exiting.")
                shutdown_mt5()
                exit()
            time.sleep(1) 
            print(f"{sym} enabled.")

    SYMBOLS_INFO = {}
    for sym in SYMBOLS_TO_TRADE:
        info = mt5.symbol_info(sym)
        if info: 
            SYMBOLS_INFO[sym] = info
            # Print some symbol details for verification
            print(f"Symbol: {info.name}, Digits: {info.digits}, Point: {info.point}, Volume Min: {info.volume_min}, Volume Max: {info.volume_max}, Volume Step: {info.volume_step}, Tick Value: {info.trade_tick_value}, Tick Size: {info.trade_tick_size}")
        else:
            print(f"Could not get symbol_info for {sym} post-enabling. Exiting.")
            shutdown_mt5()
            exit()
    
    print("\n--- Starting Live Trading Bot (Total Balance Risk Mode) ---")
    print(f"Trading Symbols: {', '.join(SYMBOLS_TO_TRADE)}")
    print(f"Timeframe: {TIMEFRAME_STR}")
    print(f"Magic Number: {MAGIC_NUMBER}")
    print(f"Risk per Trade: {RISK_PERCENT_PER_TRADE}% of CURRENT total balance.")
    print(f"Daily Total Risk Limit: {DAILY_TOTAL_RISK_LIMIT_PERCENT}% of total balance at UTC day start.")
    print("Press Ctrl+C to stop the bot.\n")

    sleep_duration = 30 

    try:
        while True:
            if not mt5.terminal_info(): 
                print("MT5 terminal connection lost. Attempting to reconnect...")
                shutdown_mt5()
                time.sleep(10)
                if not initialize_mt5_connection():
                    print("Reconnection failed. Waiting before next attempt...")
                    time.sleep(60)
                    continue 
                else:
                    print("Reconnected to MT5 successfully.")
                    current_trading_day_date = None 

            account_info = mt5.account_info()
            if account_info is None:
                print("Could not get account info. Possible connection issue. Waiting...")
                time.sleep(sleep_duration)
                continue
            
            current_total_balance = account_info.balance
            account_currency = account_info.currency

            today_utc_date = datetime.now(timezone.utc).date()

            if current_trading_day_date is None or current_trading_day_date != today_utc_date:
                current_trading_day_date = today_utc_date
                balance_at_day_start_utc = current_total_balance 
                daily_risk_limit_amount_for_day = balance_at_day_start_utc * (DAILY_TOTAL_RISK_LIMIT_PERCENT / 100.0)
                cumulative_risk_taken_today = 0.0
                
                print(f"\n--- NEW TRADING DAY (UTC): {current_trading_day_date.strftime('%Y-%m-%d')} ---")
                print(f"Balance at Day Start (UTC): {balance_at_day_start_utc:.2f} {account_currency}")
                print(f"Daily Total Risk Limit ({DAILY_TOTAL_RISK_LIMIT_PERCENT}% of start balance): {daily_risk_limit_amount_for_day:.2f} {account_currency}")
                print(f"Cumulative Risk Taken Today (Reset): {cumulative_risk_taken_today:.2f} {account_currency}\n")

            if daily_risk_limit_amount_for_day <= 0 and balance_at_day_start_utc > 0 : # Allow trading if balance was positive at day start but limit is small
                 print(f"Warning: Daily risk limit amount ({daily_risk_limit_amount_for_day:.2f}) is zero or negative, but day start balance was positive. Trades might be very small or skipped.")
            elif balance_at_day_start_utc <= 0:
                print(f"Balance at day start ({balance_at_day_start_utc:.2f}) is zero or negative. No new trades will be initiated today.")
                time.sleep(sleep_duration) # Wait before next check if no trading is possible
                continue


            for symbol_name in SYMBOLS_TO_TRADE:
                symbol_information = SYMBOLS_INFO.get(symbol_name)
                if not symbol_information:
                    print(f"Missing symbol_info for {symbol_name} in main loop. Skipping.")
                    continue
                
                check_and_trade(symbol_name, symbol_information, current_total_balance)
                time.sleep(0.2) 

            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\nBot stopping due to Ctrl+C...")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down MT5 connection...")
        shutdown_mt5()
        print("Bot stopped.")