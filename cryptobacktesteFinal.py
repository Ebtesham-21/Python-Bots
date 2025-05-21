import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone, time as dt_time # Import time for hour comparison
import pytz
import time
import traceback
import uuid # For unique trade IDs

# --- Configuration ---
SYMBOLS_TO_TRADE = ["BTCUSD", "ETHUSD", "BTCXAU"] # , "BTCXAU"
# SYMBOLS_TO_TRADE = ["XAUUSD", "EURUSD"]
# SYMBOLS_TO_TRADE = ["BTCUSD"]

TIMEFRAME_STR = "M30"
MAGIC_NUMBER = 123456

RISK_PERCENT_PER_TRADE = 1.0
DAILY_CAPITAL_ALLOCATION_PERCENT = 2.0 # Set to 100% for simpler backtesting if not testing this feature

# Strategy Parameters
EMA_SHORT_LEN = 5
EMA_MEDIUM_LEN = 8
EMA_LONG_LEN = 13 # Used for trading TF trend in reactivation
RISK_REWARD_RATIO = 2.0
SL_BUFFER_PIPS = 5 # This is used for initial SL as per the provided code
USE_ALTERNATIVE_EXIT = True

# Trailing Stop at 1R Configuration
USE_TRAILING_STOP_AT_1R = True
TRAILING_STOP_1R_TRIGGER_R = 1.0  # Price must move 1R in profit to trigger
TRAILING_STOP_1R_LEVEL_R = 0.2    # Move SL to +0.2R from entry price

# New Momentum Indicator Parameters
RSI_LEN = 14
MACD_FAST_LEN = 12
MACD_SLOW_LEN = 26
MACD_SIGNAL_LEN = 9

# Trading Time Limit (UTC)
TRADING_START_HOUR_UTC = 6
TRADING_END_HOUR_UTC = 20 

# --- Backtesting Specific Configuration ---
BACKTEST_START_DATE = datetime(2024, 11, 1, tzinfo=timezone.utc) 
BACKTEST_END_DATE = datetime(2025, 5, 20, tzinfo=timezone.utc)
INITIAL_BALANCE = 200.0
SPREAD_PIPS = 2

# --- Drawdown Control & Reactivation Configuration ---
DRAWDOWN_CONSECUTIVE_LOSSES_LIMIT = 5
DRAWDOWN_EQUITY_PERCENT_LIMIT = 20.0 
HTF_TIMEFRAME_STR = "H4" 
HTF_EMA_LEN = 20 
PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK = SYMBOLS_TO_TRADE[0] if SYMBOLS_TO_TRADE else None 

# --- MT5 Timeframe Mapping ---
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
}
MT5_TIMEFRAME = TIMEFRAME_MAP.get(TIMEFRAME_STR)
MT5_HTF_TIMEFRAME = TIMEFRAME_MAP.get(HTF_TIMEFRAME_STR)


# --- Global Variables for Backtesting ---
sim_balance = INITIAL_BALANCE
sim_equity = INITIAL_BALANCE
sim_open_positions = []
sim_trade_history = []
SYMBOLS_INFO = {}
all_htf_historical_data = {} 

daily_allocated_capital_for_trading = 0.0
current_trading_day_date_sim = None

# --- Drawdown Tracking & Control State ---
peak_equity_so_far = INITIAL_BALANCE
max_drawdown_percentage = 0.0

bot_deactivated = False
consecutive_losses = 0
trend_filter_flipped_post_deactivation = False
htf_confirmed_post_deactivation = False
trend_direction_at_flip = None 
last_known_trading_tf_trend_state_for_reactivation = None 


# --- MT5 Connection ---
def initialize_mt5_for_backtest():
    if not mt5.initialize():
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    print("MT5 Initialized for data download.")
    return True

def shutdown_mt5_after_backtest():
    mt5.shutdown()
    print("MetaTrader 5 Shutdown after data operations.")

# --- Data and Indicator Functions ---
def get_historical_data_for_backtest(symbol, timeframe_mt5, start_date, end_date, timeframe_str_log=""):
    print(f"Fetching {timeframe_str_log} historical data for {symbol} from {start_date} to {end_date}...")
    rates = mt5.copy_rates_range(symbol, timeframe_mt5, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"No {timeframe_str_log} data returned for {symbol} in the given range.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC') 
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    print(f"Fetched {len(df)} {timeframe_str_log} bars for {symbol}.")
    return df

def calculate_indicators(df, ema_short=EMA_SHORT_LEN, ema_medium=EMA_MEDIUM_LEN, ema_long=EMA_LONG_LEN, 
                         rsi_len=RSI_LEN, macd_fast=MACD_FAST_LEN, macd_slow=MACD_SLOW_LEN, macd_signal=MACD_SIGNAL_LEN,
                         htf_ema_len=HTF_EMA_LEN, is_htf=False):
    if df.empty: return pd.DataFrame()
    
    if is_htf:
        if len(df) < htf_ema_len: return pd.DataFrame() 
        df.ta.ema(length=htf_ema_len, append=True, col_names=(f'EMA_{htf_ema_len}',))
    else:
        min_len_needed_ema = max(ema_short, ema_medium, ema_long)
        min_len_needed_rsi = rsi_len
        min_len_needed_macd = macd_slow + macd_signal -1 
        
        min_len_overall = max(min_len_needed_ema, min_len_needed_rsi, min_len_needed_macd)
        if len(df) < min_len_overall:
            return pd.DataFrame()

        df.ta.ema(length=ema_short, append=True, col_names=(f'EMA_{ema_short}',))
        df.ta.ema(length=ema_medium, append=True, col_names=(f'EMA_{ema_medium}',))
        df.ta.ema(length=ema_long, append=True, col_names=(f'EMA_{ema_long}',))
        
        df.ta.rsi(length=rsi_len, append=True) 
        df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True) 
    
    df.dropna(inplace=True)
    return df

# --- Position Sizing ---
def calculate_lot_size(balance_for_risk_calc, risk_percent, sl_distance_price_units, symbol_info):
    if sl_distance_price_units <= 1e-9: return 0.0
    if balance_for_risk_calc <= 0 : return 0.0
    risk_amount_account_currency = balance_for_risk_calc * (risk_percent / 100.0)
    if symbol_info.trade_tick_size == 0: return symbol_info.volume_min if risk_amount_account_currency > 0 else 0.0
    value_per_full_price_unit_per_lot = symbol_info.trade_tick_value / symbol_info.trade_tick_size
    if abs(value_per_full_price_unit_per_lot) < 1e-9: return symbol_info.volume_min if risk_amount_account_currency > 0 else 0.0
    sl_value_per_lot = sl_distance_price_units * value_per_full_price_unit_per_lot
    if abs(sl_value_per_lot) < 1e-9: return 0.0
    raw_lot_size = risk_amount_account_currency / sl_value_per_lot
    rounded_lot_size = (raw_lot_size // symbol_info.volume_step) * symbol_info.volume_step if symbol_info.volume_step != 0 else round(raw_lot_size, 8)
    final_lot_size = max(symbol_info.volume_min, rounded_lot_size)
    final_lot_size = min(symbol_info.volume_max, final_lot_size)
    if final_lot_size < symbol_info.volume_min: return 0.0
    return final_lot_size

# --- Simulated Trade Execution ---
def sim_place_trade(symbol, trade_type, lot_size, entry_price, sl_price, tp_price, current_time, symbol_info, initial_risk_price_units):
    global sim_open_positions
    spread_amount = SPREAD_PIPS * symbol_info.point
    actual_entry_price = entry_price + (spread_amount / 2 if trade_type == mt5.ORDER_TYPE_BUY else -spread_amount / 2)
    position_id = str(uuid.uuid4())
    position = {"ticket": position_id, "symbol": symbol, "type": trade_type, "volume": lot_size,
                  "price_open": actual_entry_price, "sl": sl_price, "tp": tp_price, "time_open": current_time,
                  "magic": MAGIC_NUMBER, "comment": "EMA_Strategy_Backtest",
                  "initial_risk_price_units": initial_risk_price_units, # Store initial risk
                  "sl_trailed_to_profit_1R": False # Flag for this trailing stop
                  }
    sim_open_positions.append(position)
    trade_type_str = "BUY" if trade_type == mt5.ORDER_TYPE_BUY else "SELL"
    print(f"{current_time}: Placed {trade_type_str} {symbol} @ {actual_entry_price:.{symbol_info.digits}f}, SL={sl_price:.{symbol_info.digits}f}, TP={tp_price:.{symbol_info.digits}f}, Lots={lot_size}, Initial Risk: {initial_risk_price_units:.{symbol_info.digits}f}")

def sim_close_trade(position_ticket, close_price, close_time, reason="Manual Close"):
    global sim_balance, sim_open_positions, sim_trade_history, consecutive_losses, bot_deactivated, \
           trend_filter_flipped_post_deactivation, htf_confirmed_post_deactivation, trend_direction_at_flip, \
           last_known_trading_tf_trend_state_for_reactivation, peak_equity_so_far

    position_to_close_idx = -1
    for i, pos in enumerate(sim_open_positions):
        if pos["ticket"] == position_ticket:
            position_to_close_idx = i
            break
    if position_to_close_idx == -1:
        # This can happen if a trade is closed by SL hit, then attempted to be closed again by TP hit in same bar logic.
        # print(f"Error: Could not find position {position_ticket} to close.") 
        return
    
    position_to_close = sim_open_positions.pop(position_to_close_idx)
    symbol_info = SYMBOLS_INFO[position_to_close["symbol"]]
    spread_amount = SPREAD_PIPS * symbol_info.point
    actual_close_price = close_price - (spread_amount / 2 if position_to_close["type"] == mt5.ORDER_TYPE_BUY else -spread_amount / 2)
    price_diff = (actual_close_price - position_to_close["price_open"]) if position_to_close["type"] == mt5.ORDER_TYPE_BUY else (position_to_close["price_open"] - actual_close_price)
    
    pnl_per_unit = symbol_info.trade_tick_value / symbol_info.trade_tick_size if symbol_info.trade_tick_size != 0 else 0
    pnl = price_diff * pnl_per_unit * position_to_close["volume"]
    sim_balance += pnl

    closed_trade_info = {**position_to_close, "price_close": actual_close_price, "time_close": close_time, "pnl": pnl, "reason": reason}
    sim_trade_history.append(closed_trade_info)
    print(f"{close_time}: Closed {position_to_close['symbol']} Pos {position_to_close['ticket']} @ {actual_close_price:.{symbol_info.digits}f}. P&L: {pnl:.2f}. Reason: {reason}. Balance: {sim_balance:.2f}")

    if not bot_deactivated: 
        if pnl < 0:
            consecutive_losses += 1
            print(f"Consecutive losses: {consecutive_losses}")
            if consecutive_losses >= DRAWDOWN_CONSECUTIVE_LOSSES_LIMIT:
                bot_deactivated = True
                trend_filter_flipped_post_deactivation = False
                htf_confirmed_post_deactivation = False
                trend_direction_at_flip = None
                last_known_trading_tf_trend_state_for_reactivation = None
                print(f"*** BOT DEACTIVATED due to {consecutive_losses} consecutive losses. ***")
        else:
            consecutive_losses = 0
    return pnl


# --- Check SL/TP ---
def check_sl_tp_hits(symbol, current_bar_high, current_bar_low, current_bar_time):
    # Iterate over a copy of the list as sim_close_trade modifies the original list
    for pos in list(sim_open_positions): 
        if pos["symbol"] == symbol:
            closed_this_iteration = False
            if pos["type"] == mt5.ORDER_TYPE_BUY:
                if pos["sl"] is not None and current_bar_low <= pos["sl"]:
                    sim_close_trade(pos["ticket"], pos["sl"], current_bar_time, reason="SL Hit")
                    closed_this_iteration = True
                if not closed_this_iteration and pos["tp"] is not None and current_bar_high >= pos["tp"]: # Check TP only if not closed by SL
                    sim_close_trade(pos["ticket"], pos["tp"], current_bar_time, reason="TP Hit")
            elif pos["type"] == mt5.ORDER_TYPE_SELL:
                if pos["sl"] is not None and current_bar_high >= pos["sl"]:
                    sim_close_trade(pos["ticket"], pos["sl"], current_bar_time, reason="SL Hit")
                    closed_this_iteration = True
                if not closed_this_iteration and pos["tp"] is not None and current_bar_low <= pos["tp"]: # Check TP only if not closed by SL
                    sim_close_trade(pos["ticket"], pos["tp"], current_bar_time, reason="TP Hit")

# --- Manage Trailing Stop at 1R ---
def manage_trailing_stop_1R(symbol, current_bar_high, current_bar_low, current_event_time, symbol_info_obj):
    if not USE_TRAILING_STOP_AT_1R:
        return

    # Iterate over a copy of the list as sim_close_trade (if called from here) modifies the original list
    for pos_idx, pos in enumerate(list(sim_open_positions)): # Use enumerate if modifying original via index, or direct object mod
        if pos["symbol"] == symbol and not pos.get("sl_trailed_to_profit_1R", False):
            initial_risk = pos["initial_risk_price_units"]
            if initial_risk <= 1e-9 : continue # Skip if initial risk is zero or invalid

            target_reached = False
            if pos["type"] == mt5.ORDER_TYPE_BUY:
                profit_target_1R = pos["price_open"] + (initial_risk * TRAILING_STOP_1R_TRIGGER_R)
                if current_bar_high >= profit_target_1R:
                    target_reached = True
            elif pos["type"] == mt5.ORDER_TYPE_SELL:
                profit_target_1R = pos["price_open"] - (initial_risk * TRAILING_STOP_1R_TRIGGER_R)
                if current_bar_low <= profit_target_1R:
                    target_reached = True
            
            if target_reached:
                new_sl_level = 0.0
                if pos["type"] == mt5.ORDER_TYPE_BUY:
                    new_sl_level = pos["price_open"] + (initial_risk * TRAILING_STOP_1R_LEVEL_R)
                    # Ensure new SL is an improvement and valid
                    if new_sl_level > pos["sl"] and new_sl_level < current_bar_high: # Must be better than old SL and below current high
                        # Find the actual position in the original list to modify
                        original_pos_to_update = next((p for p in sim_open_positions if p["ticket"] == pos["ticket"]), None)
                        if original_pos_to_update:
                            original_pos_to_update["sl"] = new_sl_level
                            original_pos_to_update["sl_trailed_to_profit_1R"] = True
                            print(f"{current_event_time}: {symbol} BUY Pos {pos['ticket']} SL trailed to {new_sl_level:.{symbol_info_obj.digits}f} (+0.2R).")
                            # Check if newly trailed SL was hit by this bar's low
                            if current_bar_low <= new_sl_level:
                                sim_close_trade(pos["ticket"], new_sl_level, current_event_time, reason="Trailing SL Hit (same bar)")
                                # continue # To next position as this one is closed
                
                elif pos["type"] == mt5.ORDER_TYPE_SELL:
                    new_sl_level = pos["price_open"] - (initial_risk * TRAILING_STOP_1R_LEVEL_R)
                    # Ensure new SL is an improvement and valid
                    if new_sl_level < pos["sl"] and new_sl_level > current_bar_low: # Must be better than old SL and above current low
                        original_pos_to_update = next((p for p in sim_open_positions if p["ticket"] == pos["ticket"]), None)
                        if original_pos_to_update:
                            original_pos_to_update["sl"] = new_sl_level
                            original_pos_to_update["sl_trailed_to_profit_1R"] = True
                            print(f"{current_event_time}: {symbol} SELL Pos {pos['ticket']} SL trailed to {new_sl_level:.{symbol_info_obj.digits}f} (+0.2R).")
                            # Check if newly trailed SL was hit by this bar's high
                            if current_bar_high >= new_sl_level:
                                sim_close_trade(pos["ticket"], new_sl_level, current_event_time, reason="Trailing SL Hit (same bar)")
                                # continue # To next position as this one is closed


# --- Main Trading Logic ---
def backtest_check_and_trade(symbol, historical_df_slice, current_bar_data, next_bar_open, symbol_info, capital_for_day_risk_calc):
    global bot_deactivated 

    if bot_deactivated: 
        return 

    current_signal_time_utc = current_bar_data.name 
    current_hour_utc = current_signal_time_utc.hour

    if not (TRADING_START_HOUR_UTC <= current_hour_utc < TRADING_END_HOUR_UTC):
        return

    ema_s_col, ema_m_col, ema_l_col = f'EMA_{EMA_SHORT_LEN}', f'EMA_{EMA_MEDIUM_LEN}', f'EMA_{EMA_LONG_LEN}'
    rsi_col = f'RSI_{RSI_LEN}'
    macd_line_col = f'MACD_{MACD_FAST_LEN}_{MACD_SLOW_LEN}_{MACD_SIGNAL_LEN}'
    macd_signal_col = f'MACDs_{MACD_FAST_LEN}_{MACD_SLOW_LEN}_{MACD_SIGNAL_LEN}'

    sl_buffer_actual_price = SL_BUFFER_PIPS * symbol_info.point # Initial SL is based on this

    df_with_indicators = calculate_indicators(historical_df_slice.copy()) 
    
    if df_with_indicators.empty or len(df_with_indicators) < 2: 
        return

    signal_bar = df_with_indicators.iloc[-1]
    bar_before_signal = df_with_indicators.iloc[-2]
    bot_position = next((p for p in sim_open_positions if p["symbol"] == symbol and p["magic"] == MAGIC_NUMBER), None)

    minutes_delta = 30 
    if TIMEFRAME_STR == "M1": minutes_delta = 1
    # ... (other timeframes)
    elif TIMEFRAME_STR == "H4": minutes_delta = 240
    next_bar_event_time = current_signal_time_utc + pd.Timedelta(minutes=minutes_delta)

    if bot_position: # Manages existing positions (alternative exit)
        if USE_ALTERNATIVE_EXIT:
            exit_signal = False
            # ... (alternative exit logic)
            if exit_signal:
                sim_close_trade(bot_position["ticket"], next_bar_open, next_bar_event_time, reason="Alt Exit EMA Cross")
                return
        return # Return if position exists and no alt exit, or alt exit handled

    # --- New Trade Entry Logic ---
    trade_type_to_place, stop_loss_price, take_profit_price, risk_price_diff_abs = None, None, None, 0.0
    tentative_m30_trade_type = None
    tentative_sl, tentative_tp, tentative_risk_diff = None, None, 0.0

    m30_long_ema_cross = signal_bar[ema_s_col] > signal_bar[ema_m_col] and bar_before_signal[ema_s_col] <= bar_before_signal[ema_m_col]
    m30_long_price_above_emas = signal_bar['Close'] > signal_bar[ema_s_col] and signal_bar['Close'] > signal_bar[ema_m_col] and signal_bar['Close'] > signal_bar[ema_l_col]

    if m30_long_ema_cross and m30_long_price_above_emas:
        tentative_m30_trade_type = mt5.ORDER_TYPE_BUY
        sl_candidate_buy = signal_bar['Low'] - sl_buffer_actual_price # Initial SL based on pip buffer
        if sl_candidate_buy >= next_bar_open: sl_candidate_buy = next_bar_open - sl_buffer_actual_price - (symbol_info.point * 5)
        if sl_candidate_buy < next_bar_open:
            risk_diff_buy = next_bar_open - sl_candidate_buy
            if risk_diff_buy > symbol_info.point * 2: # Min risk threshold
                tentative_sl, tentative_tp, tentative_risk_diff = sl_candidate_buy, next_bar_open + (risk_diff_buy * RISK_REWARD_RATIO), risk_diff_buy
            else: tentative_m30_trade_type = None
        else: tentative_m30_trade_type = None

    if tentative_m30_trade_type is None:
        m30_short_ema_cross = signal_bar[ema_s_col] < signal_bar[ema_m_col] and bar_before_signal[ema_s_col] >= bar_before_signal[ema_m_col]
        m30_short_price_below_emas = signal_bar['Close'] < signal_bar[ema_s_col] and signal_bar['Close'] < signal_bar[ema_m_col] and signal_bar['Close'] < signal_bar[ema_l_col]
        if m30_short_ema_cross and m30_short_price_below_emas:
            tentative_m30_trade_type = mt5.ORDER_TYPE_SELL
            sl_candidate_sell = signal_bar['High'] + sl_buffer_actual_price # Initial SL based on pip buffer
            if sl_candidate_sell <= next_bar_open: sl_candidate_sell = next_bar_open + sl_buffer_actual_price + (symbol_info.point * 5)
            if sl_candidate_sell > next_bar_open:
                risk_diff_sell = sl_candidate_sell - next_bar_open
                if risk_diff_sell > symbol_info.point * 2: # Min risk threshold
                    tentative_sl, tentative_tp, tentative_risk_diff = sl_candidate_sell, next_bar_open - (risk_diff_sell * RISK_REWARD_RATIO), risk_diff_sell
                else: tentative_m30_trade_type = None
            else: tentative_m30_trade_type = None

    if tentative_m30_trade_type is None: return 

    h4_trend_direction = None
    # ... (H4 trend logic)
    if sym in all_htf_historical_data and not all_htf_historical_data[sym].empty:
        df_htf_sym = all_htf_historical_data[sym]
        aligned_htf_bars = df_htf_sym[df_htf_sym.index <= current_signal_time_utc] 
        if not aligned_htf_bars.empty:
            latest_htf_bar = aligned_htf_bars.iloc[-1]
            htf_ema_col_name = f'EMA_{HTF_EMA_LEN}'
            if htf_ema_col_name in latest_htf_bar.index and pd.notna(latest_htf_bar[htf_ema_col_name]):
                if latest_htf_bar['Close'] > latest_htf_bar[htf_ema_col_name]: h4_trend_direction = "bullish"
                elif latest_htf_bar['Close'] < latest_htf_bar[htf_ema_col_name]: h4_trend_direction = "bearish"


    m30_h4_aligned_trade_type = None
    # ... (M30 H4 alignment)
    if tentative_m30_trade_type == mt5.ORDER_TYPE_BUY and h4_trend_direction == "bullish":
        m30_h4_aligned_trade_type = mt5.ORDER_TYPE_BUY
    elif tentative_m30_trade_type == mt5.ORDER_TYPE_SELL and h4_trend_direction == "bearish":
        m30_h4_aligned_trade_type = mt5.ORDER_TYPE_SELL
    
    if m30_h4_aligned_trade_type is None: return 

    rsi_value = signal_bar.get(rsi_col)
    macd_line = signal_bar.get(macd_line_col)
    macd_signal_val = signal_bar.get(macd_signal_col)

    if rsi_value is None or macd_line is None or macd_signal_val is None: return 

    momentum_confirmed = False
    # ... (Momentum confirmation)
    if m30_h4_aligned_trade_type == mt5.ORDER_TYPE_BUY:
        if rsi_value > 50 and macd_line > macd_signal_val:
            momentum_confirmed = True
    elif m30_h4_aligned_trade_type == mt5.ORDER_TYPE_SELL:
        if rsi_value < 50 and macd_line < macd_signal_val:
            momentum_confirmed = True
            
    if not momentum_confirmed: return 

    trade_type_to_place = m30_h4_aligned_trade_type
    stop_loss_price = tentative_sl
    take_profit_price = tentative_tp
    risk_price_diff_abs = tentative_risk_diff # This is our 1R for the trade
    
    if trade_type_to_place is not None: 
        if capital_for_day_risk_calc <= 0 and DAILY_CAPITAL_ALLOCATION_PERCENT > 0: return
        balance_for_lot_calc = sim_balance if DAILY_CAPITAL_ALLOCATION_PERCENT == 100 else capital_for_day_risk_calc
        calculated_lot = calculate_lot_size(balance_for_lot_calc, RISK_PERCENT_PER_TRADE, risk_price_diff_abs, symbol_info)
        if calculated_lot >= symbol_info.volume_min:
            # Pass risk_price_diff_abs as initial_risk_price_units
            sim_place_trade(symbol, trade_type_to_place, calculated_lot, next_bar_open,
                            stop_loss_price, take_profit_price, next_bar_event_time, symbol_info, risk_price_diff_abs)


# --- Main Backtesting Loop ---
if __name__ == "__main__":
    # ... (Initialization as before)
    if not MT5_TIMEFRAME: print(f"Error: Invalid trading timeframe string '{TIMEFRAME_STR}'."); exit()
    if not MT5_HTF_TIMEFRAME: print(f"Error: Invalid HTF timeframe string '{HTF_TIMEFRAME_STR}'."); exit()
    if not PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK: print("Error: No primary symbol defined for reactivation checks."); exit()

    if not initialize_mt5_for_backtest(): exit()

    all_historical_data = {} 
    min_data_length = 0

    for sym in SYMBOLS_TO_TRADE:
        info = mt5.symbol_info(sym)
        if info is None: print(f"Symbol {sym} not found. Exiting."); shutdown_mt5_after_backtest(); exit()
        if not info.visible:
            if not mt5.symbol_select(sym, True): print(f"Failed to enable {sym}. Exiting."); shutdown_mt5_after_backtest(); exit()
            time.sleep(0.5)
        SYMBOLS_INFO[sym] = info
        df = get_historical_data_for_backtest(sym, MT5_TIMEFRAME, BACKTEST_START_DATE, BACKTEST_END_DATE, TIMEFRAME_STR)
        if df.empty: print(f"No {TIMEFRAME_STR} data for {sym}. Exiting."); shutdown_mt5_after_backtest(); exit()
        all_historical_data[sym] = df
        min_data_length = len(df) if min_data_length == 0 else min(min_data_length, len(df))

    for sym in SYMBOLS_TO_TRADE: 
        df_htf = get_historical_data_for_backtest(sym, MT5_HTF_TIMEFRAME, BACKTEST_START_DATE, BACKTEST_END_DATE, HTF_TIMEFRAME_STR)
        if df_htf.empty: 
            print(f"Warning: No {HTF_TIMEFRAME_STR} data for {sym}. MTFC for this symbol will not be possible.")
            all_htf_historical_data[sym] = pd.DataFrame() 
        else: 
            all_htf_historical_data[sym] = calculate_indicators(df_htf, htf_ema_len=HTF_EMA_LEN, is_htf=True)
            if all_htf_historical_data[sym].empty:
                 print(f"Warning: HTF data for {sym} processed but resulted in empty DataFrame. MTFC might be affected.")

    if PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK not in all_htf_historical_data or all_htf_historical_data[PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK].empty:
        print(f"CRITICAL: HTF data for primary reactivation symbol {PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK} is missing or empty. Reactivation will not work correctly.")

    shutdown_mt5_after_backtest()
    if not all_historical_data: print("No trading data loaded. Exiting."); exit()

    common_index = None
    for sym_df in all_historical_data.values():
        common_index = sym_df.index if common_index is None else common_index.intersection(sym_df.index)
    if common_index is None or common_index.empty: print("No common time index. Exiting."); exit()
    
    print(f"Found {len(common_index)} common {TIMEFRAME_STR} steps for backtesting.")
    for sym in SYMBOLS_TO_TRADE:
        all_historical_data[sym] = all_historical_data[sym].loc[common_index].sort_index()

    print("\n--- Starting Backtest ---")
    # ... (Print configurations)
    print(f"Symbols: {SYMBOLS_TO_TRADE}")
    print(f"Trading Timeframe: {TIMEFRAME_STR}, HTF for Confirmation: {HTF_TIMEFRAME_STR}")
    print(f"Trading Hours UTC: {TRADING_START_HOUR_UTC}:00 - {TRADING_END_HOUR_UTC-1}:59")
    print(f"Initial SL Buffer: {SL_BUFFER_PIPS} pips")
    if USE_TRAILING_STOP_AT_1R:
        print(f"Trailing Stop: Active. Trigger at {TRAILING_STOP_1R_TRIGGER_R}R, moves SL to +{TRAILING_STOP_1R_LEVEL_R}R.")
    else:
        print("Trailing Stop: Inactive.")
    print(f"Risk per trade: {RISK_PERCENT_PER_TRADE}%, Risk/Reward: 1:{RISK_REWARD_RATIO}")
    print(f"RSI({RSI_LEN}), MACD({MACD_FAST_LEN},{MACD_SLOW_LEN},{MACD_SIGNAL_LEN}) confirmations active.")
    print(f"Backtest Period: {BACKTEST_START_DATE.strftime('%Y-%m-%d')} to {BACKTEST_END_DATE.strftime('%Y-%m-%d')}")


    max_indicator_period = max(EMA_LONG_LEN, RSI_LEN, (MACD_SLOW_LEN + MACD_SIGNAL_LEN -1) )
    required_bars_for_signal_logic = max_indicator_period + 2 

    for i in range(required_bars_for_signal_logic, len(common_index)): 
        current_signal_bar_time = common_index[i-1] 
        next_bar_open_time = common_index[i] # This is the timestamp for current bar's H/L/C     

        today_utc_date_sim = current_signal_bar_time.date()
        # ... (Daily capital allocation)
        if current_trading_day_date_sim is None or current_trading_day_date_sim != today_utc_date_sim:
            current_trading_day_date_sim = today_utc_date_sim
            daily_allocated_capital_for_trading = sim_balance * (DAILY_CAPITAL_ALLOCATION_PERCENT / 100.0)


        if bot_deactivated: 
            # ... (Reactivation logic)
            primary_sym_df_trading = all_historical_data.get(PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK)
            primary_sym_df_htf = all_htf_historical_data.get(PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK)

            if primary_sym_df_trading is not None and not primary_sym_df_trading.empty and \
               primary_sym_df_htf is not None and not primary_sym_df_htf.empty:
                if not trend_filter_flipped_post_deactivation:
                    signal_bar_iloc_pos_primary = primary_sym_df_trading.index.get_loc(current_signal_bar_time)
                    start_iloc_pos_primary = max(0, signal_bar_iloc_pos_primary - required_bars_for_signal_logic + 1)
                    primary_slice_trading = primary_sym_df_trading.iloc[start_iloc_pos_primary : signal_bar_iloc_pos_primary + 1]
                    
                    primary_indicators_trading_react_ema = calculate_indicators(primary_slice_trading.copy(), ema_long=EMA_LONG_LEN, is_htf=False) 
                    
                    if not primary_indicators_trading_react_ema.empty:
                        latest_primary_bar_trading = primary_indicators_trading_react_ema.iloc[-1]
                        if f'EMA_{EMA_LONG_LEN}' in latest_primary_bar_trading:
                            current_primary_tf_trend = "bullish" if latest_primary_bar_trading['Close'] > latest_primary_bar_trading[f'EMA_{EMA_LONG_LEN}'] else "bearish"
                            if last_known_trading_tf_trend_state_for_reactivation is None: 
                                last_known_trading_tf_trend_state_for_reactivation = current_primary_tf_trend
                            elif current_primary_tf_trend != last_known_trading_tf_trend_state_for_reactivation:
                                trend_filter_flipped_post_deactivation = True
                                trend_direction_at_flip = current_primary_tf_trend
                                print(f"{current_signal_bar_time}: Trading TF Trend for {PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK} Flipped to {trend_direction_at_flip}. Awaiting HTF confirmation.")
                            last_known_trading_tf_trend_state_for_reactivation = current_primary_tf_trend
                if trend_filter_flipped_post_deactivation and not htf_confirmed_post_deactivation:
                    aligned_htf_bars = primary_sym_df_htf[primary_sym_df_htf.index <= current_signal_bar_time]
                    if not aligned_htf_bars.empty:
                        latest_htf_bar = aligned_htf_bars.iloc[-1]
                        htf_ema_col_name = f'EMA_{HTF_EMA_LEN}'
                        if htf_ema_col_name in latest_htf_bar and pd.notna(latest_htf_bar[htf_ema_col_name]): 
                            current_htf_trend = "bullish" if latest_htf_bar['Close'] > latest_htf_bar[htf_ema_col_name] else "bearish"
                            if current_htf_trend == trend_direction_at_flip:
                                htf_confirmed_post_deactivation = True
                                print(f"{current_signal_bar_time}: HTF Trend for {PRIMARY_SYMBOL_FOR_REACTIVATION_CHECK} Confirmed {current_htf_trend}.")
                if trend_filter_flipped_post_deactivation and htf_confirmed_post_deactivation:
                    bot_deactivated = False
                    consecutive_losses = 0 
                    peak_equity_so_far = sim_equity 
                    print(f"*** BOT REACTIVATED at {current_signal_bar_time}. Peak equity reset to {peak_equity_so_far:.2f} ***")


        for sym in SYMBOLS_TO_TRADE:
            symbol_info = SYMBOLS_INFO[sym]
            historical_df = all_historical_data[sym]

            if next_bar_open_time not in historical_df.index or current_signal_bar_time not in historical_df.index:
                continue

            current_bar_high_for_sltp = historical_df.loc[next_bar_open_time]['High']
            current_bar_low_for_sltp = historical_df.loc[next_bar_open_time]['Low']
            
            # Check initial SL/TP hits first for positions open at start of this bar
            check_sl_tp_hits(sym, current_bar_high_for_sltp, current_bar_low_for_sltp, next_bar_open_time)

            # Manage trailing stops for any open positions that qualify
            # This function might close trades if the new trailed SL is hit on the same bar
            manage_trailing_stop_1R(sym, current_bar_high_for_sltp, current_bar_low_for_sltp, next_bar_open_time, symbol_info)
            
            can_trade_today = daily_allocated_capital_for_trading > 0 or DAILY_CAPITAL_ALLOCATION_PERCENT == 0
            if not can_trade_today and not bot_deactivated : 
                continue

            signal_bar_iloc_pos = historical_df.index.get_loc(current_signal_bar_time)
            start_iloc_pos = max(0, signal_bar_iloc_pos - required_bars_for_signal_logic + 1) 
            df_slice_for_indicators = historical_df.iloc[start_iloc_pos : signal_bar_iloc_pos + 1]
            
            signal_generating_bar_data = historical_df.loc[current_signal_bar_time]
            entry_price_bar_open = historical_df.loc[next_bar_open_time]['Open']

            # This function attempts to open new trades
            backtest_check_and_trade(sym, df_slice_for_indicators, signal_generating_bar_data,
                                     entry_price_bar_open, symbol_info, daily_allocated_capital_for_trading)
        
        # --- Equity, Drawdown, and Deactivation by Equity Drop ---
        # ... (Equity and drawdown calculation as before)
        current_total_equity = sim_balance
        for pos in sim_open_positions:
            pos_symbol_info = SYMBOLS_INFO[pos["symbol"]]
            if next_bar_open_time in all_historical_data[pos["symbol"]].index:
                current_close_price_for_pos_sym = all_historical_data[pos["symbol"]].loc[next_bar_open_time]['Close']
                price_diff_ue = (current_close_price_for_pos_sym - pos["price_open"]) if pos["type"] == mt5.ORDER_TYPE_BUY else (pos["price_open"] - current_close_price_for_pos_sym)
                pnl_per_unit_ue = pos_symbol_info.trade_tick_value / pos_symbol_info.trade_tick_size if pos_symbol_info.trade_tick_size != 0 else 0
                current_total_equity += price_diff_ue * pnl_per_unit_ue * pos["volume"]
        sim_equity = current_total_equity

        if sim_equity > peak_equity_so_far: peak_equity_so_far = sim_equity
        
        current_drawdown = 0
        if peak_equity_so_far > 0 : 
            current_drawdown = (peak_equity_so_far - sim_equity) / peak_equity_so_far
        
        if current_drawdown > max_drawdown_percentage: max_drawdown_percentage = current_drawdown

        if not bot_deactivated and peak_equity_so_far > 0: 
            if current_drawdown >= (DRAWDOWN_EQUITY_PERCENT_LIMIT / 100.0):
                bot_deactivated = True
                trend_filter_flipped_post_deactivation = False
                htf_confirmed_post_deactivation = False
                trend_direction_at_flip = None
                last_known_trading_tf_trend_state_for_reactivation = None
                consecutive_losses = 0 
                print(f"*** BOT DEACTIVATED due to equity drawdown of {current_drawdown*100:.2f}% (limit {DRAWDOWN_EQUITY_PERCENT_LIMIT}%) at {current_signal_bar_time}. Peak Equity was {peak_equity_so_far:.2f}, Current Equity {sim_equity:.2f} ***")


    # --- End of Backtest Loop ---
    # ... (Closing remaining positions and printing stats as before)
    if sim_open_positions:
        print("\n--- Closing remaining open positions at end of backtest ---")
        last_bar_time = common_index[-1]
        for sym_open_pos in list(sim_open_positions): # Iterate over a copy
            # Check if historical data exists for the symbol up to the last bar
            if sym_open_pos["symbol"] in all_historical_data and not all_historical_data[sym_open_pos["symbol"]].empty:
                last_close_price = all_historical_data[sym_open_pos["symbol"]].iloc[-1]['Close']
                sim_close_trade(sym_open_pos["ticket"], last_close_price, last_bar_time, reason="End of Backtest")
            else:
                print(f"Warning: Could not find historical data for {sym_open_pos['symbol']} to close position at end of backtest.")
        
        sim_equity = sim_balance # Recalculate equity after closing all
        if sim_equity > peak_equity_so_far: peak_equity_so_far = sim_equity
        if peak_equity_so_far > 0: # Avoid division by zero
            final_drawdown_check = (peak_equity_so_far - sim_equity) / peak_equity_so_far
            if final_drawdown_check > max_drawdown_percentage: max_drawdown_percentage = final_drawdown_check


    print("\n--- Backtest Finished ---")
    print(f"Period: {common_index[0].strftime('%Y-%m-%d %H:%M')} to {common_index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"Initial Balance: {INITIAL_BALANCE:.2f}")
    print(f"Final Balance: {sim_balance:.2f}")
    print(f"Final Equity: {sim_equity:.2f}")
    print(f"Peak Equity During Backtest: {peak_equity_so_far:.2f}")
    print(f"Max Drawdown: {max_drawdown_percentage * 100:.2f}%")
    print(f"Total Trades: {len(sim_trade_history)}")

    if len(sim_trade_history) > 0:
        pnl_values = [trade['pnl'] for trade in sim_trade_history]
        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = sum(p for p in pnl_values if p < 0)
        net_profit = sum(pnl_values)
        winning_trades = sum(1 for p in pnl_values if p > 0)
        losing_trades = sum(1 for p in pnl_values if p < 0)
        win_rate = (winning_trades / len(sim_trade_history) * 100) if len(sim_trade_history) > 0 else 0
        print(f"Net Profit: {net_profit:.2f}")
        print(f"Gross Profit: {gross_profit:.2f}")
        print(f"Gross Loss: {abs(gross_loss):.2f}")
        print(f"Profit Factor: {(gross_profit / abs(gross_loss)) if gross_loss != 0 else 'inf'}")
        print(f"Win Rate: {win_rate:.2f}% ({winning_trades}/{len(sim_trade_history)})")
    else:
        print("No trades were executed during the backtest.")