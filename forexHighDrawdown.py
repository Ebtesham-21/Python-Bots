import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date, time as dt_time # Import time for hour checks

# --- Configuration ---
SYMBOLS_TO_TEST = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
ENTRY_TIMEFRAME_STR = "M5"
BOX_TREND_TIMEFRAME_STR = "H1"
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc) # User updated
END_DATE = datetime(2025, 1, 30, tzinfo=timezone.utc)   # User updated

STARTING_BALANCE = 200.00 # USD
RISK_PER_TRADE_PCT = 0.01  # 1% risk per trade (default)
JPY_RISK_PER_TRADE_PCT = 0.001 # 0.2% risk for JPY pairs
DAILY_MAX_LOSS_PCT = 0.02 # 2% max daily loss of starting daily balance


RISK_REWARD_RATIO = 2.0
SL_BUFFER_PIPS_DEFAULT = 2

EMA_PERIOD_1H = 50
LOCAL_STRUCTURE_LOOKBACK_5M = 15
PIN_BAR_BODY_RATIO = 0.33
PIN_BAR_WICK_MIN_RATIO = 2.0

# Trading Session Times (UTC)
SESSION_1_START_UTC = dt_time(7, 0)  # 07:00 UTC
SESSION_1_END_UTC = dt_time(11, 0) # 11:00 UTC
SESSION_2_START_UTC = dt_time(12, 0) # 12:00 UTC
SESSION_2_END_UTC = dt_time(16, 0) # 16:00 UTC


# --- MT5 Timeframe Mapping ---
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}
ENTRY_TIMEFRAME_MT5 = None
BOX_TREND_TIMEFRAME_MT5 = None

# --- Helper function for Pip Value and SL Buffer ---
def get_symbol_pip_info(symbol_str):
    symbol_info_obj = mt5.symbol_info(symbol_str)
    if symbol_info_obj is None:
        print(f"CRITICAL: Could not get symbol_info for {symbol_str}. MT5 Error: {mt5.last_error()}")
        return 0.0001, SL_BUFFER_PIPS_DEFAULT, 0.01 # Default pip size, buffer, lot_step
    point = symbol_info_obj.point
    digits = symbol_info_obj.digits
    lot_step = symbol_info_obj.volume_step

    # pip_size_for_calc here means the actual size of a pip (e.g., 0.0001 or 0.01)
    if "JPY" in symbol_str.upper():
        pip_size_for_calc = 0.01
        sl_buffer_pips = SL_BUFFER_PIPS_DEFAULT * 10 # JPY pairs usually have 2-3 decimal places for pips
    elif digits == 5 or digits == 3: # 5-digit (EURUSD 1.23456) or 3-digit (USDJPY 123.456)
        pip_size_for_calc = 10 * point
        sl_buffer_pips = SL_BUFFER_PIPS_DEFAULT
    elif digits == 4 or digits == 2: # 4-digit (EURUSD 1.2345) or 2-digit (USDJPY 123.45)
        pip_size_for_calc = point
        sl_buffer_pips = SL_BUFFER_PIPS_DEFAULT
    else: # Default fallback
        pip_size_for_calc = point
        sl_buffer_pips = SL_BUFFER_PIPS_DEFAULT
    return pip_size_for_calc, sl_buffer_pips, lot_step

# --- MT5 Connection Functions ---
def initialize_mt5_connection():
    global ENTRY_TIMEFRAME_MT5, BOX_TREND_TIMEFRAME_MT5
    entry_tf_mt5_val = TIMEFRAME_MAP.get(ENTRY_TIMEFRAME_STR)
    box_trend_tf_mt5_val = TIMEFRAME_MAP.get(BOX_TREND_TIMEFRAME_STR)
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error()); return False
    print("MetaTrader 5 Initialized Successfully")
    acct_info = mt5.account_info()
    if acct_info: print(f"Account: {acct_info.login}, Server: {acct_info.server}, Balance: {acct_info.balance} {acct_info.currency}")
    else: print("Could not get account info. Error:", mt5.last_error());
    if entry_tf_mt5_val is None or box_trend_tf_mt5_val is None:
        print("Error: Invalid timeframe string(s) in config."); mt5.shutdown(); return False
    ENTRY_TIMEFRAME_MT5 = entry_tf_mt5_val; BOX_TREND_TIMEFRAME_MT5 = box_trend_tf_mt5_val
    return True

def shutdown_mt5():
    mt5.shutdown(); print("MetaTrader 5 Shutdown")

# --- Data Fetching ---
def get_historical_data(symbol, timeframe_mt5, start_dt, end_dt, count_limit=200000):
    rates = mt5.copy_rates_range(symbol, timeframe_mt5, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        print(f"copy_rates_range for {symbol} (TF: {timeframe_mt5}) from {start_dt} to {end_dt} returned no data or failed. Error: {mt5.last_error()}. Trying copy_rates_from_pos.")
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count_limit)
        if rates is None or len(rates) == 0:
            print(f"CRITICAL: copy_rates_from_pos also failed for {symbol}. MT5 Error: {mt5.last_error()}")
            return pd.DataFrame()
    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s', utc=True)
    rates_df = rates_df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
    rates_df.set_index('time', inplace=True)
    # Filter again to ensure exact date range, especially if copy_rates_from_pos was used
    rates_df = rates_df[(rates_df.index >= start_dt) & (rates_df.index <= end_dt)]
    return rates_df

# --- Strategy Logic Helper Functions ---
def calculate_ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def get_1h_trend_bias(current_5m_time, rates_1h_df, ema_col_name='ema_1h'):
    # Get the 1H candle that was active or just closed before or at current_5m_time
    relevant_1h_candle_slice = rates_1h_df[rates_1h_df.index <= current_5m_time]
    if relevant_1h_candle_slice.empty: return "NEUTRAL"
    relevant_1h_candle = relevant_1h_candle_slice.iloc[-1:] # Last row is the relevant 1H candle
    if relevant_1h_candle.empty: return "NEUTRAL"

    close_1h = relevant_1h_candle['close'].iloc[0]
    ema_1h = relevant_1h_candle[ema_col_name].iloc[0]

    if pd.isna(ema_1h): return "NEUTRAL" # EMA might not be calculated for early candles
    if close_1h > ema_1h: return "BUY"
    elif close_1h < ema_1h: return "SELL"
    else: return "NEUTRAL"

def is_bullish_engulfing(prev_candle, curr_candle):
    if prev_candle is None or curr_candle is None: return False
    if prev_candle['open'] > prev_candle['close'] and curr_candle['open'] < curr_candle['close']: # Prev bearish, curr bullish
        if curr_candle['close'] > prev_candle['open'] and curr_candle['open'] < prev_candle['close']: return True # Curr engulfs prev body
    return False

def is_bearish_engulfing(prev_candle, curr_candle):
    if prev_candle is None or curr_candle is None: return False
    if prev_candle['open'] < prev_candle['close'] and curr_candle['open'] > curr_candle['close']: # Prev bullish, curr bearish
        if curr_candle['close'] < prev_candle['open'] and curr_candle['open'] > prev_candle['close']: return True # Curr engulfs prev body
    return False

def is_pin_bar(candle, is_bullish_check=True):
    if candle is None: return False
    body_size = abs(candle['open'] - candle['close'])
    total_range = candle['high'] - candle['low']
    if total_range < 0.00001 : return False # Avoid division by zero for zero-range candles
    if (body_size / total_range) > PIN_BAR_BODY_RATIO: return False

    if is_bullish_check: # Long lower wick
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        if body_size > 0.000001 : # Measurable body
            return lower_wick >= PIN_BAR_WICK_MIN_RATIO * body_size and lower_wick > upper_wick
        else: # Doji-like
            return lower_wick > upper_wick and lower_wick > (total_range * 0.6) # Lower wick dominates
    else: # Long upper wick
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        if body_size > 0.000001: # Measurable body
            return upper_wick >= PIN_BAR_WICK_MIN_RATIO * body_size and upper_wick > lower_wick
        else: # Doji-like
            return upper_wick > lower_wick and upper_wick > (total_range * 0.6) # Upper wick dominates

# --- Lot Size Calculation ---
def calculate_lot_size(equity, current_risk_pct_per_trade, sl_pips, pip_size_for_calc, contract_size, lot_step_val, min_lot_val):
    if sl_pips <= 0: return min_lot_val
    risk_amount_per_trade = equity * current_risk_pct_per_trade # Use current_risk_pct_per_trade

    # Value of 1 pip for 1 standard lot (e.g., for EURUSD, pip_size_for_calc=0.0001, contract_size=100,000, so $10)
    value_of_one_pip_per_lot = pip_size_for_calc * contract_size

    sl_monetary_value_for_one_lot = sl_pips * value_of_one_pip_per_lot
    if sl_monetary_value_for_one_lot <= 0: return min_lot_val

    calculated_lot = risk_amount_per_trade / sl_monetary_value_for_one_lot
    calculated_lot = max(min_lot_val, np.floor(calculated_lot / lot_step_val) * lot_step_val)
    return round(calculated_lot, int(-np.log10(lot_step_val)) if lot_step_val < 1 else 2)

# --- Trading Session Check ---
def is_within_trading_session(current_time_utc):
    time_now_utc = current_time_utc.time()
    session1 = (SESSION_1_START_UTC <= time_now_utc < SESSION_1_END_UTC)
    session2 = (SESSION_2_START_UTC <= time_now_utc < SESSION_2_END_UTC)
    return session1 or session2

# --- Main Backtesting Function ---
def run_single_symbol_backtest(symbol_to_test, initial_equity_for_symbol):
    print(f"\n{'='*20} Starting Backtest for {symbol_to_test} {'='*20}")
    pip_size_for_calc, current_sl_buffer_pips_val, lot_step = get_symbol_pip_info(symbol_to_test)
    symbol_info_obj = mt5.symbol_info(symbol_to_test)
    contract_size = symbol_info_obj.trade_contract_size
    min_lot = symbol_info_obj.volume_min

    # Determine symbol-specific risk percentage
    if "JPY" in symbol_to_test.upper():
        symbol_specific_risk_pct = JPY_RISK_PER_TRADE_PCT
        print(f"INFO ({symbol_to_test}): Using JPY-specific risk: {symbol_specific_risk_pct*100}%")
    else:
        symbol_specific_risk_pct = RISK_PER_TRADE_PCT

    ema_buffer_days = EMA_PERIOD_1H * 3
    rates_1h_start_date = START_DATE - timedelta(days=ema_buffer_days)
    rates_1h_df = get_historical_data(symbol_to_test, BOX_TREND_TIMEFRAME_MT5, rates_1h_start_date, END_DATE)
    rates_5m_df = get_historical_data(symbol_to_test, ENTRY_TIMEFRAME_MT5, START_DATE, END_DATE)

    if rates_1h_df.empty or rates_5m_df.empty:
        print(f"Could not fetch sufficient data for {symbol_to_test} (1H empty: {rates_1h_df.empty}, 5M empty: {rates_5m_df.empty}). Skipping.")
        return None, initial_equity_for_symbol

    rates_1h_df['ema_1h'] = calculate_ema(rates_1h_df, EMA_PERIOD_1H)

    trades = []
    active_trade = None
    current_equity = initial_equity_for_symbol
    peak_equity = initial_equity_for_symbol
    max_drawdown_pct = 0.0

    daily_pnl = {}
    start_of_day_equity = {}

    consecutive_daily_limit_hits = 0
    trading_halt_until_date = None
    last_processed_day_for_streak_check = None

    broken_level_buy = None; bos_index_buy_5m = -1; waiting_for_pullback_buy = False
    broken_level_sell = None; bos_index_sell_5m = -1; waiting_for_pullback_sell = False

    if rates_5m_df.empty:
        print(f"No M5 data for {symbol_to_test} in the specified range after initial fetch. Skipping.")
        return None, initial_equity_for_symbol

    for i in range(max(1, LOCAL_STRUCTURE_LOOKBACK_5M), len(rates_5m_df)):
        current_candle_5m = rates_5m_df.iloc[i]
        prev_candle_5m = rates_5m_df.iloc[i-1] # Sufficient lookback ensures i-1 is valid
        current_time_5m = rates_5m_df.index[i]
        current_date = current_time_5m.date()

        # --- Daily Loss Streak and Trading Halt Logic (processed once per day) ---
        if last_processed_day_for_streak_check is None or current_date > last_processed_day_for_streak_check:
            if last_processed_day_for_streak_check is not None:
                prev_day_for_streak = last_processed_day_for_streak_check
                if prev_day_for_streak in daily_pnl and prev_day_for_streak in start_of_day_equity:
                    sod_equity_prev = start_of_day_equity[prev_day_for_streak]
                    daily_loss_limit_amount_prev = sod_equity_prev * DAILY_MAX_LOSS_PCT
                    pnl_prev_day = daily_pnl[prev_day_for_streak]

                    day_hit_limit = False
                    if daily_loss_limit_amount_prev > 0 and pnl_prev_day <= -daily_loss_limit_amount_prev:
                        day_hit_limit = True

                    if day_hit_limit:
                        consecutive_daily_limit_hits += 1
                        # print(f"INFO ({symbol_to_test} | {prev_day_for_streak}): Daily loss limit HIT. PnL: {pnl_prev_day:.2f}. Consecutive hits: {consecutive_daily_limit_hits}.")
                    else:
                        if consecutive_daily_limit_hits > 0:
                            pass # print(f"INFO ({symbol_to_test} | {prev_day_for_streak}): Daily PnL: {pnl_prev_day:.2f}. Resetting consecutive hit count.")
                        consecutive_daily_limit_hits = 0

                    if consecutive_daily_limit_hits >= 3:
                        trading_halt_until_date = prev_day_for_streak + timedelta(days=4) # Halt for 3 full trading days after the 3rd loss day
                        print(f"HALT ({symbol_to_test}): 3 consecutive daily loss limits. Trading halted. Will check to resume on or after {trading_halt_until_date}.")
                        consecutive_daily_limit_hits = 0
            
            last_processed_day_for_streak_check = current_date

            if trading_halt_until_date is not None and current_date >= trading_halt_until_date:
                print(f"RESUME ({symbol_to_test}): Trading halt period ended. Resuming trading checks from {current_date}.")
                trading_halt_until_date = None
        
        if current_date not in daily_pnl: # Initialize for the current day
            daily_pnl[current_date] = 0.0
            start_of_day_equity[current_date] = current_equity

        if trading_halt_until_date is not None and current_date < trading_halt_until_date:
            continue # Skip all processing for this candle if in a trading halt for the symbol

        peak_equity = max(peak_equity, current_equity)

        # --- Active Trade Management ---
        if active_trade:
            exit_reason = None; exit_price = None; unrealized_pnl_trade = 0
            value_of_one_pip_for_one_lot = pip_size_for_calc * contract_size

            if active_trade['type'] == 'BUY':
                # PnL in currency: (current_price - entry_price) * lot_size_in_units
                unrealized_pnl_trade = (current_candle_5m['close'] - active_trade['entry_price']) * (contract_size * active_trade['lot_size'])
                if current_candle_5m['high'] >= active_trade['tp']: exit_price, exit_reason = active_trade['tp'], "TP_HIT"
                elif current_candle_5m['low'] <= active_trade['sl']: exit_price, exit_reason = active_trade['sl'], "SL_HIT"
            elif active_trade['type'] == 'SELL':
                unrealized_pnl_trade = (active_trade['entry_price'] - current_candle_5m['close']) * (contract_size * active_trade['lot_size'])
                if current_candle_5m['low'] <= active_trade['tp']: exit_price, exit_reason = active_trade['tp'], "TP_HIT"
                elif current_candle_5m['high'] >= active_trade['sl']: exit_price, exit_reason = active_trade['sl'], "SL_HIT"

            current_simulated_equity = initial_equity_for_symbol + sum(t['pnl_currency'] for t in trades) + unrealized_pnl_trade
            if peak_equity > 0:
                drawdown = (peak_equity - current_simulated_equity) / peak_equity
                max_drawdown_pct = max(max_drawdown_pct, drawdown)

            if exit_reason:
                pnl_pips_val = (exit_price - active_trade['entry_price']) / pip_size_for_calc if active_trade['type'] == 'BUY' else (active_trade['entry_price'] - exit_price) / pip_size_for_calc
                # Actual PnL calculation based on pips, pip value for lot size, and lot size
                pnl_currency_val = pnl_pips_val * value_of_one_pip_for_one_lot * active_trade['lot_size']

                current_equity += pnl_currency_val
                if current_date in daily_pnl: daily_pnl[current_date] += pnl_currency_val
                else: daily_pnl[current_date] = pnl_currency_val # Should be initialized

                peak_equity = max(peak_equity, current_equity)

                trades.append({
                    'symbol': symbol_to_test, 'entry_time': active_trade['entry_time'], 'exit_time': current_time_5m,
                    'type': active_trade['type'], 'lot_size': active_trade['lot_size'],
                    'entry_price': active_trade['entry_price'],'sl': active_trade['sl'], 'tp': active_trade['tp'],
                    'exit_price': exit_price, 'pnl_pips': pnl_pips_val,
                    'pnl_currency': pnl_currency_val, 'reason': exit_reason,
                    'equity_after_trade': current_equity
                })
                active_trade = None
                waiting_for_pullback_buy = False
                waiting_for_pullback_sell = False
            else:
                continue # Active trade, no exit, continue to next candle

        # --- Daily Loss Limit Check (for new trades) ---
        if start_of_day_equity.get(current_date, 0) > 0: # Ensure SOD equity is positive
            max_allowable_daily_loss_amount = start_of_day_equity[current_date] * DAILY_MAX_LOSS_PCT
            if max_allowable_daily_loss_amount > 0 and daily_pnl.get(current_date, 0) <= -max_allowable_daily_loss_amount :
                continue # Skip new trade checks if daily loss limit hit

        # --- Trading Session Check (for new trades) ---
        if not is_within_trading_session(current_time_5m):
            continue

        # --- New Trade Entry Logic ---
        if active_trade is None: # Only look for new trades if none active and limits/sessions permit
            trend_1h_bias = get_1h_trend_bias(current_time_5m, rates_1h_df)

            if trend_1h_bias == "BUY":
                waiting_for_pullback_sell = False # Reset sell state
                if not waiting_for_pullback_buy:
                    lookback_slice_5m = rates_5m_df.iloc[max(0, i - LOCAL_STRUCTURE_LOOKBACK_5M) : i]
                    if not lookback_slice_5m.empty:
                        local_resistance_5m = lookback_slice_5m['high'].max()
                        if current_candle_5m['close'] > local_resistance_5m: # BoS
                            broken_level_buy = local_resistance_5m; bos_index_buy_5m = i; waiting_for_pullback_buy = True
                elif waiting_for_pullback_buy and i > bos_index_buy_5m and broken_level_buy is not None:
                    # Check for pullback to the broken level (or slightly above for buffer)
                    if current_candle_5m['low'] <= broken_level_buy + (1 * pip_size_for_calc):
                        if is_bullish_engulfing(prev_candle_5m, current_candle_5m) or is_pin_bar(current_candle_5m, is_bullish_check=True):
                            entry_price = current_candle_5m['close']
                            sl_base = current_candle_5m['low'] # Default SL base
                            if is_bullish_engulfing(prev_candle_5m, current_candle_5m) and prev_candle_5m is not None:
                                sl_base = min(sl_base, prev_candle_5m['low']) # Consider engulfed candle's low for tighter SL
                            sl_price = sl_base - (current_sl_buffer_pips_val * pip_size_for_calc)
                            sl_pips = abs(entry_price - sl_price) / pip_size_for_calc

                            if sl_pips > 0: # Ensure SL is not zero
                                calculated_lot = calculate_lot_size(current_equity, symbol_specific_risk_pct, sl_pips, pip_size_for_calc, contract_size, lot_step, min_lot)
                                if calculated_lot >= min_lot :
                                    tp_price = entry_price + (sl_pips * pip_size_for_calc * RISK_REWARD_RATIO)
                                    active_trade = {'type': 'BUY', 'lot_size': calculated_lot, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price, 'entry_time': current_time_5m, 'entry_index_5m': i}
                                    waiting_for_pullback_buy = False; broken_level_buy = None
            
            elif trend_1h_bias == "SELL":
                waiting_for_pullback_buy = False # Reset buy state
                if not waiting_for_pullback_sell:
                    lookback_slice_5m = rates_5m_df.iloc[max(0, i - LOCAL_STRUCTURE_LOOKBACK_5M) : i]
                    if not lookback_slice_5m.empty:
                        local_support_5m = lookback_slice_5m['low'].min()
                        if current_candle_5m['close'] < local_support_5m: # BoS
                            broken_level_sell = local_support_5m; bos_index_sell_5m = i; waiting_for_pullback_sell = True
                elif waiting_for_pullback_sell and i > bos_index_sell_5m and broken_level_sell is not None:
                    # Check for pullback to the broken level (or slightly below for buffer)
                    if current_candle_5m['high'] >= broken_level_sell - (1 * pip_size_for_calc):
                        if is_bearish_engulfing(prev_candle_5m, current_candle_5m) or is_pin_bar(current_candle_5m, is_bullish_check=False):
                            entry_price = current_candle_5m['close']
                            sl_base = current_candle_5m['high'] # Default SL base
                            if is_bearish_engulfing(prev_candle_5m, current_candle_5m) and prev_candle_5m is not None:
                                sl_base = max(sl_base, prev_candle_5m['high']) # Consider engulfed candle's high for tighter SL
                            sl_price = sl_base + (current_sl_buffer_pips_val * pip_size_for_calc)
                            sl_pips = abs(sl_price - entry_price) / pip_size_for_calc

                            if sl_pips > 0: # Ensure SL is not zero
                                calculated_lot = calculate_lot_size(current_equity, symbol_specific_risk_pct, sl_pips, pip_size_for_calc, contract_size, lot_step, min_lot)
                                if calculated_lot >= min_lot:
                                    tp_price = entry_price - (sl_pips * pip_size_for_calc * RISK_REWARD_RATIO)
                                    active_trade = {'type': 'SELL','lot_size': calculated_lot, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price, 'entry_time': current_time_5m, 'entry_index_5m': i}
                                    waiting_for_pullback_sell = False; broken_level_sell = None
            else: # NEUTRAL trend
                waiting_for_pullback_buy = False
                waiting_for_pullback_sell = False

    final_equity_for_symbol = current_equity
    print(f"\n--- Backtest Results for {symbol_to_test} ---")
    if not trades:
        print("No trades were executed."); print(f"Starting Balance: {initial_equity_for_symbol:.2f} USD"); print(f"Final Equity: {final_equity_for_symbol:.2f} USD")
        return None, final_equity_for_symbol
    else:
        trades_df = pd.DataFrame(trades); total_trades = len(trades_df); wins_df = trades_df[trades_df['pnl_currency'] > 0]
        losses_df = trades_df[trades_df['pnl_currency'] <= 0]; num_wins = len(wins_df); num_losses = total_trades - num_wins
        win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = wins_df['pnl_currency'].sum(); gross_loss = abs(losses_df[losses_df['pnl_currency'] < 0]['pnl_currency'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        print(f"Starting Balance for Symbol Test: {initial_equity_for_symbol:.2f} USD"); print(f"Total Trades: {total_trades}"); print(f"Wins: {num_wins}, Losses: {num_losses}")
        print(f"Win Rate: {win_rate:.2f}%"); print(f"Gross Profit: {gross_profit:.2f} USD"); print(f"Gross Loss: {gross_loss:.2f} USD"); print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown for Symbol Test: {max_drawdown_pct*100:.2f}%"); print(f"Final Equity for Symbol Test: {final_equity_for_symbol:.2f} USD")
        print(f"Total Net PnL (Currency): {trades_df['pnl_currency'].sum():.2f} USD")
        return trades_df, final_equity_for_symbol

# --- Main Execution Block ---
if __name__ == "__main__":
    if not initialize_mt5_connection():
        print("CRITICAL: MT5 Initialization failed. Exiting program.")
    else:
        all_symbols_results_list = []
        for symbol_name in SYMBOLS_TO_TEST: # Renamed variable to avoid conflict
            symbol_info_mt5 = mt5.symbol_info(symbol_name)
            if symbol_info_mt5 is None: print(f"Symbol {symbol_name} not found by MT5. Skipping."); continue
            if not symbol_info_mt5.visible:
                print(f"Symbol {symbol_name} not visible. Attempting to select.")
                if not mt5.symbol_select(symbol_name, True):
                    print(f"Could not make {symbol_name} visible. Skipping. MT5 Error: {mt5.last_error()}"); continue
                mt5.sleep(1000) # Give MT5 time to make symbol available

            symbol_trades_df, _ = run_single_symbol_backtest(symbol_name, STARTING_BALANCE)
            if symbol_trades_df is not None and not symbol_trades_df.empty:
                all_symbols_results_list.append(symbol_trades_df)

        if all_symbols_results_list:
            combined_trades_df = pd.concat(all_symbols_results_list, ignore_index=True)
            if not combined_trades_df.empty:
                combined_trades_df.sort_values(by='exit_time', inplace=True) # Sort all trades chronologically

                # Calculate portfolio equity curve and max drawdown
                running_portfolio_equity = STARTING_BALANCE
                peak_portfolio_equity = STARTING_BALANCE
                max_portfolio_drawdown_pct = 0.0
                portfolio_equity_curve_values = [STARTING_BALANCE] # Re-initialize for actual curve

                for index, trade in combined_trades_df.iterrows():
                    running_portfolio_equity += trade['pnl_currency']
                    portfolio_equity_curve_values.append(running_portfolio_equity)
                    peak_portfolio_equity = max(peak_portfolio_equity, running_portfolio_equity)
                    if peak_portfolio_equity > 0: # Avoid division by zero if equity becomes zero or negative
                        current_drawdown = (peak_portfolio_equity - running_portfolio_equity) / peak_portfolio_equity
                        max_portfolio_drawdown_pct = max(max_portfolio_drawdown_pct, current_drawdown)

                final_portfolio_equity = portfolio_equity_curve_values[-1] if portfolio_equity_curve_values else STARTING_BALANCE

                print(f"\n{'='*20} Overall Combined Portfolio Results {'='*20}")
                print(f"Starting Balance: {STARTING_BALANCE:.2f} USD"); total_trades_combined = len(combined_trades_df)
                print(f"Total Trades (All Symbols): {total_trades_combined}")
                wins_combined_df = combined_trades_df[combined_trades_df['pnl_currency'] > 0]
                # For win rate, consider trades with PnL > 0 as wins. PnL <= 0 are losses or break-even.
                losses_combined_df = combined_trades_df[combined_trades_df['pnl_currency'] <= 0]
                num_wins_combined = len(wins_combined_df)
                num_losses_combined = total_trades_combined - num_wins_combined # All non-wins
                overall_win_rate = (num_wins_combined / total_trades_combined) * 100 if total_trades_combined > 0 else 0
                print(f"Total Wins: {num_wins_combined}, Total Losses: {num_losses_combined}"); print(f"Overall Win Rate: {overall_win_rate:.2f}%")

                gross_profit_combined = wins_combined_df['pnl_currency'].sum()
                # For gross loss, only sum trades with PnL < 0
                actual_losses_df = combined_trades_df[combined_trades_df['pnl_currency'] < 0]
                gross_loss_combined = abs(actual_losses_df['pnl_currency'].sum())

                profit_factor_combined = gross_profit_combined / gross_loss_combined if gross_loss_combined > 0 else float('inf') if gross_profit_combined > 0 else 0
                print(f"Gross Profit: {gross_profit_combined:.2f} USD"); print(f"Gross Loss: {gross_loss_combined:.2f} USD")
                print(f"Profit Factor: {profit_factor_combined:.2f}"); print(f"Max Portfolio Drawdown: {max_portfolio_drawdown_pct*100:.2f}%")
                print(f"Final Portfolio Equity: {final_portfolio_equity:.2f} USD")
                print(f"Total Net PnL (Currency): {combined_trades_df['pnl_currency'].sum():.2f} USD")
            else:
                print("\nNo trades executed across any symbols (after concat).")
                print(f"Starting Balance: {STARTING_BALANCE:.2f} USD")
                print(f"Final Equity: {STARTING_BALANCE:.2f} USD")
        else:
            print("\nNo trades executed across any symbols.")
            print(f"Starting Balance: {STARTING_BALANCE:.2f} USD")
            print(f"Final Equity: {STARTING_BALANCE:.2f} USD")

        shutdown_mt5()