import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date, time as dt_time # Import time for hour checks

# --- Configuration ---
SYMBOLS_TO_TEST = ["EURUSD", "GBPUSD",  "AUDUSD", "USDCHF",  "USDCAD", "USDJPY", "EURJPY", "GBPJPY"]
ENTRY_TIMEFRAME_STR = "M5"
BOX_TREND_TIMEFRAME_STR = "H1"
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 5, 20, tzinfo=timezone.utc)

STARTING_BALANCE = 200.00 # USD
RISK_PER_TRADE_PCT = 0.01
JPY_RISK_PER_TRADE_PCT = 0.001
DAILY_MAX_LOSS_PCT = 0.02 # APPLIES TO JPY PAIRS ONLY for same-day trading stop

# NEW: Configuration for severe daily drawdown halt
SEVERE_DAILY_DRAWDOWN_PCT_LIMIT = 0.20 # 20% daily drawdown limit
SEVERE_DAILY_DRAWDOWN_HALT_DAYS = 3    # Number of full days to halt trading after limit is hit

# NEW: JPY Per-Trade Max Drawdown Hard Cap
MAX_PER_TRADE_DRAWDOWN_JPY_HARD_CAP_PCT = 0.05 # 5% hard cap

MIN_SL_PIPS_JPY = 10
JPY_SL_BUFFER_PIPS = 5
RISK_REWARD_RATIO = 2.0
SL_BUFFER_PIPS_DEFAULT = 2

EMA_PERIOD_1H = 50
LOCAL_STRUCTURE_LOOKBACK_5M = 15
PIN_BAR_BODY_RATIO = 0.33
PIN_BAR_WICK_MIN_RATIO = 2.0

SESSION_1_START_UTC = dt_time(7, 0)
SESSION_1_END_UTC = dt_time(11, 0)
SESSION_2_START_UTC = dt_time(12, 0)
SESSION_2_END_UTC = dt_time(16, 0)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}
ENTRY_TIMEFRAME_MT5 = None
BOX_TREND_TIMEFRAME_MT5 = None
ACCOUNT_CURRENCY = "USD" # Default, will be updated

def get_symbol_pip_info(symbol_str):
    symbol_info_obj = mt5.symbol_info(symbol_str)
    if symbol_info_obj is None:
        print(f"CRITICAL: Could not get symbol_info for {symbol_str}. MT5 Error: {mt5.last_error()}")
        return 0.0001, SL_BUFFER_PIPS_DEFAULT, 0.01
    point = symbol_info_obj.point
    digits = symbol_info_obj.digits
    lot_step = symbol_info_obj.volume_step

    if "JPY" in symbol_str.upper():
        pip_size_for_calc = 0.01
        sl_buffer_pips = JPY_SL_BUFFER_PIPS
    elif digits == 5 or digits == 3:
        pip_size_for_calc = 10 * point
        sl_buffer_pips = SL_BUFFER_PIPS_DEFAULT
    elif digits == 4 or digits == 2:
        pip_size_for_calc = point
        sl_buffer_pips = SL_BUFFER_PIPS_DEFAULT
    else:
        pip_size_for_calc = point
        sl_buffer_pips = SL_BUFFER_PIPS_DEFAULT
    return pip_size_for_calc, sl_buffer_pips, lot_step

def initialize_mt5_connection():
    global ENTRY_TIMEFRAME_MT5, BOX_TREND_TIMEFRAME_MT5, ACCOUNT_CURRENCY
    entry_tf_mt5_val = TIMEFRAME_MAP.get(ENTRY_TIMEFRAME_STR)
    box_trend_tf_mt5_val = TIMEFRAME_MAP.get(BOX_TREND_TIMEFRAME_STR)
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error()); return False
    print("MetaTrader 5 Initialized Successfully")
    acct_info = mt5.account_info()
    if acct_info:
        ACCOUNT_CURRENCY = acct_info.currency
        print(f"Account: {acct_info.login}, Server: {acct_info.server}, Balance: {acct_info.balance} {ACCOUNT_CURRENCY}")
    else:
        print("Could not get account info. Error:", mt5.last_error());
    if entry_tf_mt5_val is None or box_trend_tf_mt5_val is None:
        print("Error: Invalid timeframe string(s) in config."); mt5.shutdown(); return False
    ENTRY_TIMEFRAME_MT5 = entry_tf_mt5_val; BOX_TREND_TIMEFRAME_MT5 = box_trend_tf_mt5_val
    return True

def shutdown_mt5():
    mt5.shutdown(); print("MetaTrader 5 Shutdown")

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
    rates_df = rates_df[(rates_df.index >= start_dt) & (rates_df.index <= end_dt)]
    return rates_df

def calculate_ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def get_1h_trend_bias(current_5m_time, rates_1h_df, ema_col_name='ema_1h'):
    relevant_1h_candle_slice = rates_1h_df[rates_1h_df.index <= current_5m_time]
    if relevant_1h_candle_slice.empty: return "NEUTRAL"
    relevant_1h_candle = relevant_1h_candle_slice.iloc[-1:]
    if relevant_1h_candle.empty: return "NEUTRAL"
    close_1h = relevant_1h_candle['close'].iloc[0]; ema_1h = relevant_1h_candle[ema_col_name].iloc[0]
    if pd.isna(ema_1h): return "NEUTRAL"
    if close_1h > ema_1h: return "BUY"
    elif close_1h < ema_1h: return "SELL"
    else: return "NEUTRAL"

def is_bullish_engulfing(prev_candle, curr_candle):
    if prev_candle is None or curr_candle is None: return False
    if prev_candle['open'] > prev_candle['close'] and curr_candle['open'] < curr_candle['close']:
        if curr_candle['close'] > prev_candle['open'] and curr_candle['open'] < prev_candle['close']: return True
    return False

def is_bearish_engulfing(prev_candle, curr_candle):
    if prev_candle is None or curr_candle is None: return False
    if prev_candle['open'] < prev_candle['close'] and curr_candle['open'] > curr_candle['close']:
        if curr_candle['close'] < prev_candle['open'] and curr_candle['open'] > prev_candle['close']: return True
    return False

def is_pin_bar(candle, is_bullish_check=True):
    if candle is None: return False
    body_size = abs(candle['open'] - candle['close']); total_range = candle['high'] - candle['low']
    if total_range < 0.00001 : return False
    if (body_size / total_range) > PIN_BAR_BODY_RATIO: return False
    if is_bullish_check:
        lower_wick = min(candle['open'], candle['close']) - candle['low']; upper_wick = candle['high'] - max(candle['open'], candle['close'])
        if body_size > 0.000001 : return lower_wick >= PIN_BAR_WICK_MIN_RATIO * body_size and lower_wick > upper_wick
        else: return lower_wick > upper_wick and lower_wick > (total_range * 0.6)
    else:
        upper_wick = candle['high'] - max(candle['open'], candle['close']); lower_wick = min(candle['open'], candle['close']) - candle['low']
        if body_size > 0.000001: return upper_wick >= PIN_BAR_WICK_MIN_RATIO * body_size and upper_wick > lower_wick
        else: return upper_wick > lower_wick and upper_wick > (total_range * 0.6)

def calculate_lot_size(equity, current_risk_pct_per_trade, sl_pips, pip_size_for_calc, contract_size, lot_step_val, min_lot_val, symbol_info_obj_for_lot_calc):
    # NOTE: For full accuracy, pip_value_in_account_currency should be used here.
    # The current implementation uses pip value in QUOTE currency.
    # pip_value_one_lot_quote_curr = pip_size_for_calc * contract_size
    # sl_monetary_value_for_one_lot_quote_curr = sl_pips * pip_value_one_lot_quote_curr
    # A more accurate approach:
    if symbol_info_obj_for_lot_calc.point == 0: return min_lot_val # Safety
    pip_value_one_lot_acc_curr = symbol_info_obj_for_lot_calc.trade_tick_value * (pip_size_for_calc / symbol_info_obj_for_lot_calc.point)

    if sl_pips <= 0: return min_lot_val
    risk_amount_per_trade = equity * current_risk_pct_per_trade # In account currency

    sl_monetary_value_for_one_lot_acc_curr = sl_pips * pip_value_one_lot_acc_curr # In account currency
    
    if sl_monetary_value_for_one_lot_acc_curr <= 0: return min_lot_val
    
    calculated_lot = risk_amount_per_trade / sl_monetary_value_for_one_lot_acc_curr
    calculated_lot = max(min_lot_val, np.floor(calculated_lot / lot_step_val) * lot_step_val)
    return round(calculated_lot, int(-np.log10(lot_step_val)) if lot_step_val < 1 else 2)


def is_within_trading_session(current_time_utc):
    time_now_utc = current_time_utc.time()
    session1 = (SESSION_1_START_UTC <= time_now_utc < SESSION_1_END_UTC)
    session2 = (SESSION_2_START_UTC <= time_now_utc < SESSION_2_END_UTC)
    return session1 or session2

def run_single_symbol_backtest(symbol_to_test, initial_equity_for_symbol):
    print(f"\n{'='*20} Starting Backtest for {symbol_to_test} {'='*20}")
    pip_size_for_calc, current_sl_buffer_pips_val, lot_step = get_symbol_pip_info(symbol_to_test)
    symbol_info_obj = mt5.symbol_info(symbol_to_test)
    if symbol_info_obj is None: # Should have been caught by get_symbol_pip_info but double check
        print(f"CRITICAL: Symbol info for {symbol_to_test} is None in run_single_symbol_backtest. Skipping.")
        return None, initial_equity_for_symbol
        
    contract_size = symbol_info_obj.trade_contract_size
    min_lot = symbol_info_obj.volume_min
    is_jpy_pair = "JPY" in symbol_to_test.upper()

    if is_jpy_pair:
        symbol_specific_risk_pct = JPY_RISK_PER_TRADE_PCT
        print(f"INFO ({symbol_to_test}): JPY Rules -> Risk: {symbol_specific_risk_pct*100:.3f}%, Min SL: {MIN_SL_PIPS_JPY} pips, SL Buffer (fixed fallback): {current_sl_buffer_pips_val} pips, Daily Max Loss (same-day stop): {DAILY_MAX_LOSS_PCT*100:.2f}%. JPY Max DD per trade: {MAX_PER_TRADE_DRAWDOWN_JPY_HARD_CAP_PCT*100:.0f}%. Dynamic ATR SL buffer will be used if available.")
    else:
        symbol_specific_risk_pct = RISK_PER_TRADE_PCT
        print(f"INFO ({symbol_to_test}): Non-JPY Rules -> Risk: {symbol_specific_risk_pct*100:.2f}%, SL Buffer: {current_sl_buffer_pips_val} pips. No JPY-specific daily max loss pct applied.")
    print(f"INFO ({symbol_to_test}): All Symbols -> Severe Daily DD Limit: {SEVERE_DAILY_DRAWDOWN_PCT_LIMIT*100:.0f}%, Halt Duration: {SEVERE_DAILY_DRAWDOWN_HALT_DAYS} days if limit hit.")

    ema_buffer_days = EMA_PERIOD_1H * 3
    rates_1h_start_date = START_DATE - timedelta(days=ema_buffer_days)
    rates_1h_df = get_historical_data(symbol_to_test, BOX_TREND_TIMEFRAME_MT5, rates_1h_start_date, END_DATE)
    rates_5m_df = get_historical_data(symbol_to_test, ENTRY_TIMEFRAME_MT5, START_DATE, END_DATE)

    if rates_1h_df.empty or rates_5m_df.empty:
        print(f"Could not fetch sufficient data for {symbol_to_test}. Skipping.")
        return None, initial_equity_for_symbol

    rates_1h_df['ema_1h'] = calculate_ema(rates_1h_df, EMA_PERIOD_1H)

    if not rates_5m_df.empty:
        rates_5m_df['atr_custom_range_calc'] = (rates_5m_df['high'].rolling(window=14).max() - rates_5m_df['low'].rolling(window=14).min())
    else:
        print(f"CRITICAL: rates_5m_df became empty for {symbol_to_test} before ATR calculation. Skipping.")
        return None, initial_equity_for_symbol

    trades = []
    active_trade = None
    current_equity = initial_equity_for_symbol
    peak_equity = initial_equity_for_symbol; max_drawdown_pct = 0.0
    daily_pnl = {}; start_of_day_equity = {}
    
    consecutive_daily_limit_hits = 0
    trading_halt_until_date_jpy_streak = None
    last_processed_day_for_jpy_streak_check = None

    severe_dd_halt_until_date = None
    last_processed_date_for_severe_dd_reset = None

    broken_level_buy = None; bos_index_buy_5m = -1; waiting_for_pullback_buy = False
    broken_level_sell = None; bos_index_sell_5m = -1; waiting_for_pullback_sell = False

    for i in range(max(1, LOCAL_STRUCTURE_LOOKBACK_5M), len(rates_5m_df)):
        current_candle_5m = rates_5m_df.iloc[i]; prev_candle_5m = rates_5m_df.iloc[i-1]
        current_time_5m = rates_5m_df.index[i]; current_date = current_time_5m.date()
        
        if last_processed_day_for_jpy_streak_check is None or current_date > last_processed_day_for_jpy_streak_check:
            if last_processed_day_for_jpy_streak_check is not None:
                prev_day_for_streak = last_processed_day_for_jpy_streak_check
                if prev_day_for_streak in daily_pnl and prev_day_for_streak in start_of_day_equity:
                    sod_equity_prev = start_of_day_equity[prev_day_for_streak]
                    pnl_prev_day = daily_pnl[prev_day_for_streak]
                    day_hit_jpy_limit = False
                    if is_jpy_pair:
                        daily_loss_limit_amount_prev_jpy = sod_equity_prev * DAILY_MAX_LOSS_PCT
                        if daily_loss_limit_amount_prev_jpy > 0 and pnl_prev_day <= -daily_loss_limit_amount_prev_jpy:
                            day_hit_jpy_limit = True
                    if day_hit_jpy_limit: consecutive_daily_limit_hits += 1
                    else: consecutive_daily_limit_hits = 0
                    if consecutive_daily_limit_hits >= 3:
                        trading_halt_until_date_jpy_streak = prev_day_for_streak + timedelta(days=4)
                        print(f"HALT_JPY_STREAK ({symbol_to_test}): 3 consec JPY daily losses. Halt until {trading_halt_until_date_jpy_streak}.")
                        consecutive_daily_limit_hits = 0
            if trading_halt_until_date_jpy_streak is not None and current_date >= trading_halt_until_date_jpy_streak:
                print(f"RESUME_JPY_STREAK ({symbol_to_test}): JPY streak halt ended. Resuming on {current_date}.")
                trading_halt_until_date_jpy_streak = None
            last_processed_day_for_jpy_streak_check = current_date

        if last_processed_date_for_severe_dd_reset is None or current_date > last_processed_date_for_severe_dd_reset:
            if severe_dd_halt_until_date is not None and current_date >= severe_dd_halt_until_date:
                print(f"RESUME_20DD ({symbol_to_test}): Severe daily drawdown halt ended. Resuming trading on {current_date}.")
                severe_dd_halt_until_date = None
            last_processed_date_for_severe_dd_reset = current_date
        
        if current_date not in daily_pnl:
            daily_pnl[current_date] = 0.0
            start_of_day_equity[current_date] = current_equity

        if severe_dd_halt_until_date is not None and current_date < severe_dd_halt_until_date: continue
        if trading_halt_until_date_jpy_streak is not None and current_date < trading_halt_until_date_jpy_streak: continue
        
        peak_equity = max(peak_equity, current_equity)

        if active_trade:
            exit_reason, exit_price, unrealized_pnl_trade = None, None, 0
            
            # Correct pip value in account currency for P&L calculation
            if symbol_info_obj.point == 0: 
                pip_value_one_lot_trade_curr = (pip_size_for_calc * contract_size) # Fallback to quote currency value
                print(f"WARNING ({symbol_to_test}): Point value zero during active trade PNL. PNL might be inaccurate.")
            else:
                pip_value_one_lot_trade_curr = symbol_info_obj.trade_tick_value * (pip_size_for_calc / symbol_info_obj.point)

            if active_trade['type'] == 'BUY':
                unrealized_pnl_trade = ((current_candle_5m['close'] - active_trade['entry_price']) / pip_size_for_calc) * pip_value_one_lot_trade_curr * active_trade['lot_size']
                if current_candle_5m['high'] >= active_trade['tp']: exit_price, exit_reason = active_trade['tp'], "TP_HIT"
                elif current_candle_5m['low'] <= active_trade['sl']: exit_price, exit_reason = active_trade['sl'], "SL_HIT"
            elif active_trade['type'] == 'SELL':
                unrealized_pnl_trade = ((active_trade['entry_price'] - current_candle_5m['close']) / pip_size_for_calc) * pip_value_one_lot_trade_curr * active_trade['lot_size']
                if current_candle_5m['low'] <= active_trade['tp']: exit_price, exit_reason = active_trade['tp'], "TP_HIT"
                elif current_candle_5m['high'] >= active_trade['sl']: exit_price, exit_reason = active_trade['sl'], "SL_HIT"
            
            current_simulated_equity = initial_equity_for_symbol + sum(t['pnl_currency'] for t in trades) + unrealized_pnl_trade
            if peak_equity > 0: max_drawdown_pct = max(max_drawdown_pct, (peak_equity - current_simulated_equity) / peak_equity)

            if exit_reason:
                pnl_pips_val = (exit_price - active_trade['entry_price']) / pip_size_for_calc if active_trade['type'] == 'BUY' else (active_trade['entry_price'] - exit_price) / pip_size_for_calc
                pnl_currency_val = pnl_pips_val * pip_value_one_lot_trade_curr * active_trade['lot_size']
                
                current_equity += pnl_currency_val
                daily_pnl[current_date] += pnl_currency_val
                peak_equity = max(peak_equity, current_equity)
                trades.append({'symbol': symbol_to_test, 'entry_time': active_trade['entry_time'], 'exit_time': current_time_5m,
                               'type': active_trade['type'], 'lot_size': active_trade['lot_size'],
                               'entry_price': active_trade['entry_price'],'sl': active_trade['sl'], 'tp': active_trade['tp'],
                               'exit_price': exit_price, 'pnl_pips': pnl_pips_val,
                               'pnl_currency': pnl_currency_val, 'reason': exit_reason, 'equity_after_trade': current_equity})
                
                if severe_dd_halt_until_date is None:
                    sod_equity_today = start_of_day_equity.get(current_date); todays_pnl = daily_pnl.get(current_date, 0.0)
                    if sod_equity_today and sod_equity_today > 0:
                        daily_loss_value = -todays_pnl
                        if daily_loss_value / sod_equity_today >= SEVERE_DAILY_DRAWDOWN_PCT_LIMIT:
                            severe_dd_halt_until_date = current_date + timedelta(days=SEVERE_DAILY_DRAWDOWN_HALT_DAYS + 1)
                            print(f"HALT_20DD ({symbol_to_test}): Daily drawdown >= {SEVERE_DAILY_DRAWDOWN_PCT_LIMIT*100:.0f}% ({(daily_loss_value / sod_equity_today)*100 :.2f}%) on {current_date}. PnL: {todays_pnl:.2f}, SoD Equity: {sod_equity_today:.2f}. Trading for this symbol halted until {severe_dd_halt_until_date}.")
                active_trade = None; waiting_for_pullback_buy = False; waiting_for_pullback_sell = False
            else:
                continue
        
        if is_jpy_pair and active_trade is None:
            sod_equity_today = start_of_day_equity.get(current_date, 0)
            if sod_equity_today > 0:
                current_pnl_today = daily_pnl.get(current_date, 0)
                daily_loss_limit_amount_jpy = sod_equity_today * DAILY_MAX_LOSS_PCT
                if daily_loss_limit_amount_jpy > 0 and current_pnl_today <= -daily_loss_limit_amount_jpy:
                    continue
        if severe_dd_halt_until_date is not None and current_date < severe_dd_halt_until_date and active_trade is None:
            continue
        if not is_within_trading_session(current_time_5m) and active_trade is None:
            continue

        if active_trade is None:
            trend_1h_bias = get_1h_trend_bias(current_time_5m, rates_1h_df)
            sl_pips, sl_price, entry_price, tp_price = 0, 0, 0, 0 
            calculated_lot = 0 # Initialize calculated_lot
            perform_trade_placement = True # Default to allow trade

            if trend_1h_bias == "BUY":
                waiting_for_pullback_sell = False
                if not waiting_for_pullback_buy:
                    lookback_slice_5m = rates_5m_df.iloc[max(0, i - LOCAL_STRUCTURE_LOOKBACK_5M) : i]
                    if not lookback_slice_5m.empty and current_candle_5m['close'] > lookback_slice_5m['high'].max():
                        broken_level_buy = lookback_slice_5m['high'].max(); bos_index_buy_5m = i; waiting_for_pullback_buy = True
                elif waiting_for_pullback_buy and i > bos_index_buy_5m and broken_level_buy is not None:
                    if current_candle_5m['low'] <= broken_level_buy + (1 * pip_size_for_calc):
                        if is_bullish_engulfing(prev_candle_5m, current_candle_5m) or is_pin_bar(current_candle_5m, is_bullish_check=True):
                            entry_price = current_candle_5m['close']
                            sl_base = current_candle_5m['low']
                            if is_bullish_engulfing(prev_candle_5m, current_candle_5m) and prev_candle_5m is not None:
                                sl_base = min(sl_base, prev_candle_5m['low'])
                            
                            active_buffer_price = current_sl_buffer_pips_val * pip_size_for_calc
                            if is_jpy_pair:
                                atr_val_prev_candle = rates_5m_df['atr_custom_range_calc'].iloc[i-1]
                                if pd.notna(atr_val_prev_candle) and atr_val_prev_candle > 0:
                                    active_buffer_price = atr_val_prev_candle * 0.3

                            temp_sl_price = sl_base - active_buffer_price
                            sl_price = temp_sl_price 
                            sl_pips = abs(entry_price - sl_price) / pip_size_for_calc if pip_size_for_calc > 0 else 0

                            if is_jpy_pair and sl_pips < MIN_SL_PIPS_JPY:
                                sl_pips = MIN_SL_PIPS_JPY
                                sl_price = entry_price - (sl_pips * pip_size_for_calc) 
                            
                            if sl_pips > 0:
                                calculated_lot = calculate_lot_size(current_equity, symbol_specific_risk_pct, sl_pips, pip_size_for_calc, contract_size, lot_step, min_lot, symbol_info_obj)
                                
                                # NEW: JPY Per-Trade Max Drawdown Check
                                if is_jpy_pair and calculated_lot > 0 :
                                    if symbol_info_obj.point == 0:
                                        print(f"WARNING ({symbol_to_test} @ {current_time_5m}): Point value is zero. Skipping JPY hard cap DD check.")
                                    elif current_equity <= 0:
                                        print(f"WARNING ({symbol_to_test} @ {current_time_5m}): Equity {current_equity:.2f} non-positive. Skipping JPY hard cap DD check.")
                                    else:
                                        pip_value_one_lot_acc_curr = symbol_info_obj.trade_tick_value * (pip_size_for_calc / symbol_info_obj.point)
                                        potential_loss_this_trade_acc_curr = sl_pips * pip_value_one_lot_acc_curr * calculated_lot
                                        potential_drawdown_this_trade_pct = potential_loss_this_trade_acc_curr / current_equity
                                        
                                        if potential_drawdown_this_trade_pct > MAX_PER_TRADE_DRAWDOWN_JPY_HARD_CAP_PCT:
                                            print(f"JPY TRADE SKIPPED ({symbol_to_test} @ {current_time_5m}): Potential DD {potential_drawdown_this_trade_pct*100:.2f}% "
                                                  f"(Loss: {potential_loss_this_trade_acc_curr:.2f} {ACCOUNT_CURRENCY} on Equity: {current_equity:.2f} {ACCOUNT_CURRENCY} "
                                                  f"for SL {sl_pips:.1f} pips, Lot {calculated_lot}) "
                                                  f"exceeds hard cap of {MAX_PER_TRADE_DRAWDOWN_JPY_HARD_CAP_PCT*100:.2f}%.")
                                            perform_trade_placement = False
                                
                                if perform_trade_placement and calculated_lot >= min_lot:
                                    tp_price = entry_price + (sl_pips * pip_size_for_calc * RISK_REWARD_RATIO)
                                    active_trade = {'type': 'BUY', 'lot_size': calculated_lot, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price, 'entry_time': current_time_5m}
                                    waiting_for_pullback_buy = False; broken_level_buy = None
            
            elif trend_1h_bias == "SELL":
                waiting_for_pullback_buy = False
                if not waiting_for_pullback_sell:
                    lookback_slice_5m = rates_5m_df.iloc[max(0, i - LOCAL_STRUCTURE_LOOKBACK_5M) : i]
                    if not lookback_slice_5m.empty and current_candle_5m['close'] < lookback_slice_5m['low'].min():
                        broken_level_sell = lookback_slice_5m['low'].min(); bos_index_sell_5m = i; waiting_for_pullback_sell = True
                elif waiting_for_pullback_sell and i > bos_index_sell_5m and broken_level_sell is not None:
                    if current_candle_5m['high'] >= broken_level_sell - (1 * pip_size_for_calc):
                        if is_bearish_engulfing(prev_candle_5m, current_candle_5m) or is_pin_bar(current_candle_5m, is_bullish_check=False):
                            entry_price = current_candle_5m['close']
                            sl_base = current_candle_5m['high']
                            if is_bearish_engulfing(prev_candle_5m, current_candle_5m) and prev_candle_5m is not None:
                                sl_base = max(sl_base, prev_candle_5m['high'])

                            active_buffer_price = current_sl_buffer_pips_val * pip_size_for_calc
                            if is_jpy_pair:
                                atr_val_prev_candle = rates_5m_df['atr_custom_range_calc'].iloc[i-1]
                                if pd.notna(atr_val_prev_candle) and atr_val_prev_candle > 0:
                                    active_buffer_price = atr_val_prev_candle * 0.3

                            temp_sl_price = sl_base + active_buffer_price
                            sl_price = temp_sl_price
                            sl_pips = abs(temp_sl_price - entry_price) / pip_size_for_calc if pip_size_for_calc > 0 else 0

                            if is_jpy_pair and sl_pips < MIN_SL_PIPS_JPY:
                                sl_pips = MIN_SL_PIPS_JPY
                                sl_price = entry_price + (sl_pips * pip_size_for_calc)

                            if sl_pips > 0:
                                calculated_lot = calculate_lot_size(current_equity, symbol_specific_risk_pct, sl_pips, pip_size_for_calc, contract_size, lot_step, min_lot, symbol_info_obj)

                                # NEW: JPY Per-Trade Max Drawdown Check
                                if is_jpy_pair and calculated_lot > 0 :
                                    if symbol_info_obj.point == 0:
                                        print(f"WARNING ({symbol_to_test} @ {current_time_5m}): Point value is zero. Skipping JPY hard cap DD check.")
                                    elif current_equity <= 0:
                                        print(f"WARNING ({symbol_to_test} @ {current_time_5m}): Equity {current_equity:.2f} non-positive. Skipping JPY hard cap DD check.")
                                    else:
                                        pip_value_one_lot_acc_curr = symbol_info_obj.trade_tick_value * (pip_size_for_calc / symbol_info_obj.point)
                                        potential_loss_this_trade_acc_curr = sl_pips * pip_value_one_lot_acc_curr * calculated_lot
                                        potential_drawdown_this_trade_pct = potential_loss_this_trade_acc_curr / current_equity

                                        if potential_drawdown_this_trade_pct > MAX_PER_TRADE_DRAWDOWN_JPY_HARD_CAP_PCT:
                                            print(f"JPY TRADE SKIPPED ({symbol_to_test} @ {current_time_5m}): Potential DD {potential_drawdown_this_trade_pct*100:.2f}% "
                                                  f"(Loss: {potential_loss_this_trade_acc_curr:.2f} {ACCOUNT_CURRENCY} on Equity: {current_equity:.2f} {ACCOUNT_CURRENCY} "
                                                  f"for SL {sl_pips:.1f} pips, Lot {calculated_lot}) "
                                                  f"exceeds hard cap of {MAX_PER_TRADE_DRAWDOWN_JPY_HARD_CAP_PCT*100:.2f}%.")
                                            perform_trade_placement = False

                                if perform_trade_placement and calculated_lot >= min_lot:
                                    tp_price = entry_price - (sl_pips * pip_size_for_calc * RISK_REWARD_RATIO)
                                    active_trade = {'type': 'SELL', 'lot_size': calculated_lot, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price, 'entry_time': current_time_5m}
                                    waiting_for_pullback_sell = False; broken_level_sell = None
            else: 
                waiting_for_pullback_buy = False; waiting_for_pullback_sell = False

    final_equity_for_symbol = current_equity
    print(f"\n--- Backtest Results for {symbol_to_test} ---")
    if not trades:
        print(f"No trades executed. Start: {initial_equity_for_symbol:.2f}, End: {final_equity_for_symbol:.2f}")
        return None, final_equity_for_symbol
    else:
        trades_df = pd.DataFrame(trades); total_trades = len(trades_df)
        wins_df = trades_df[trades_df['pnl_currency'] > 0]; num_wins = len(wins_df)
        num_losses = total_trades - num_wins
        win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = wins_df['pnl_currency'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_currency'] < 0]['pnl_currency'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
        print(f"Start Bal: {initial_equity_for_symbol:.2f}, Trades: {total_trades}, Wins: {num_wins}, Losses: {num_losses}")
        print(f"Win Rate: {win_rate:.2f}%, Gross Profit: {gross_profit:.2f}, Gross Loss: {gross_loss:.2f}, PF: {profit_factor:.2f}")
        print(f"Max DD: {max_drawdown_pct*100:.2f}%, Final Equity: {final_equity_for_symbol:.2f}, Net PnL: {trades_df['pnl_currency'].sum():.2f}")
        return trades_df, final_equity_for_symbol

# --- Main Execution Block ---
if __name__ == "__main__":
    if not initialize_mt5_connection():
        print("CRITICAL: MT5 Init failed. Exiting.")
    else:
        all_symbols_results_list = []
        for symbol_name in SYMBOLS_TO_TEST:
            symbol_info_mt5 = mt5.symbol_info(symbol_name)
            if symbol_info_mt5 is None: print(f"Symbol {symbol_name} not found. Skipping."); continue
            if not symbol_info_mt5.visible:
                if not mt5.symbol_select(symbol_name, True): print(f"Could not make {symbol_name} visible. Skip."); continue
                mt5.sleep(1000) # Allow time for symbol to be available
            symbol_trades_df, _ = run_single_symbol_backtest(symbol_name, STARTING_BALANCE)
            if symbol_trades_df is not None and not symbol_trades_df.empty:
                all_symbols_results_list.append(symbol_trades_df)

        if all_symbols_results_list:
            combined_trades_df = pd.concat(all_symbols_results_list, ignore_index=True)
            if not combined_trades_df.empty:
                combined_trades_df.sort_values(by='exit_time', inplace=True)
                
                temp_equity = STARTING_BALANCE
                equity_curve = [STARTING_BALANCE]
                # Ensure 'pnl_currency' column exists and has numeric data
                if 'pnl_currency' in combined_trades_df.columns:
                    for pnl in combined_trades_df['pnl_currency'].fillna(0): # Handle potential NaNs
                        temp_equity += pnl
                        equity_curve.append(temp_equity)
                
                if len(equity_curve) > 1 : # if any trades happened
                    peak_port_eq = equity_curve[0]
                    max_port_dd = 0.0
                    for eq_val in equity_curve:
                        peak_port_eq = max(peak_port_eq, eq_val)
                        if peak_port_eq > 0: # Avoid division by zero if peak is zero or negative
                            drawdown = (peak_port_eq - eq_val) / peak_port_eq
                            max_port_dd = max(max_port_dd, drawdown)
                    
                    final_portfolio_equity = equity_curve[-1]
                    max_portfolio_drawdown_pct = max_port_dd

                    print(f"\n{'='*20} Overall Combined Portfolio Results {'='*20}")
                    total_trades_combined = len(combined_trades_df)
                    wins_combined_df = combined_trades_df[combined_trades_df['pnl_currency'] > 0]
                    num_wins_combined = len(wins_combined_df); num_losses_combined = total_trades_combined - num_wins_combined
                    overall_win_rate = (num_wins_combined / total_trades_combined) * 100 if total_trades_combined > 0 else 0
                    gross_profit_combined = wins_combined_df['pnl_currency'].sum()
                    gross_loss_combined = abs(combined_trades_df[combined_trades_df['pnl_currency'] < 0]['pnl_currency'].sum())
                    profit_factor_combined = gross_profit_combined / gross_loss_combined if gross_loss_combined > 0 else (float('inf') if gross_profit_combined > 0 else 0)
                    print(f"Start Bal: {STARTING_BALANCE:.2f} {ACCOUNT_CURRENCY}, Total Trades: {total_trades_combined}")
                    print(f"Wins: {num_wins_combined}, Losses: {num_losses_combined}, Win Rate: {overall_win_rate:.2f}%")
                    print(f"Gross Profit: {gross_profit_combined:.2f} {ACCOUNT_CURRENCY}, Gross Loss: {gross_loss_combined:.2f} {ACCOUNT_CURRENCY}, PF: {profit_factor_combined:.2f}")
                    print(f"Max Portfolio DD: {max_portfolio_drawdown_pct*100:.2f}%, Final Equity: {final_portfolio_equity:.2f} {ACCOUNT_CURRENCY}")
                    print(f"Total Net PnL: {combined_trades_df['pnl_currency'].sum():.2f} {ACCOUNT_CURRENCY}")
                else: # No trades with PNL data
                     print(f"\nNo trades with PNL data (after concat). Start: {STARTING_BALANCE:.2f} {ACCOUNT_CURRENCY}, End: {STARTING_BALANCE:.2f} {ACCOUNT_CURRENCY}")
            else: print(f"\nNo trades (after concat). Start: {STARTING_BALANCE:.2f} {ACCOUNT_CURRENCY}, End: {STARTING_BALANCE:.2f} {ACCOUNT_CURRENCY}")
        else: print(f"\nNo trades. Start: {STARTING_BALANCE:.2f} {ACCOUNT_CURRENCY}, End: {STARTING_BALANCE:.2f} {ACCOUNT_CURRENCY}")
        shutdown_mt5()