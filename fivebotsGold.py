import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# --- MT5 Connection Variables ---
mt5_connection_active = False
ACCOUNT_CURRENCY = "USD"

# --- Your MT5 Connection Functions ---
def initialize_mt5_connection_bt():
    global mt5_connection_active, ACCOUNT_CURRENCY
    if mt5_connection_active: return True
    if not mt5.initialize():
        print(f"Initialize failed, error code = {mt5.last_error()}")
        return False
    print("MetaTrader 5 Initialized Successfully")
    mt5_connection_active = True
    terminal_info = mt5.terminal_info()
    if terminal_info:
        if hasattr(terminal_info, 'currency'):
             ACCOUNT_CURRENCY = terminal_info.currency
             print(f"Account Currency detected: {ACCOUNT_CURRENCY}")
        else:
             print(f"Could not detect account currency from terminal_info. Assuming: {ACCOUNT_CURRENCY}")
    else:
        print(f"Could not get terminal info. Assuming Account Currency: {ACCOUNT_CURRENCY}")
    return True

def shutdown_mt5_bt():
    global mt5_connection_active
    if mt5_connection_active:
        mt5.shutdown()
        mt5_connection_active = False
        print("MetaTrader 5 Shutdown")

# --- Configuration ---
SYMBOL = "XAUUSD"
TIMEFRAME_H4 = mt5.TIMEFRAME_H4
TIMEFRAME_H1 = mt5.TIMEFRAME_H1
TIMEFRAME_M5 = mt5.TIMEFRAME_M5

BACKTEST_START_DATE = datetime(2025, 1, 1)
BACKTEST_END_DATE = datetime(2025, 5, 15)

EMA_PERIOD = 200
H4_EMA_SEPARATION_THRESHOLD = 0.001    # REVERTED: Stricter H4 trend confirmation
H1_KEY_LEVEL_LOOKBACK = 10             # CHANGED: Shorter lookback for H1 Key Levels
H1_RETEST_PROXIMITY_PERCENT = 0.001    # REVERTED: Stricter H1 Retest proximity
FVG_PATTERN_CANDLE_COUNT = 3
MIN_FVG_HEIGHT_PRICE_UNITS = 0.0      # CHANGED: FVG height filter disabled
RISK_REWARD_RATIO = 2.0               # KEEPING
STOP_LOSS_OFFSET_PRICE_UNITS_GOLD = 0.50

# --- P&L Configuration ---
STARTING_BALANCE_USD = 200.00
FIXED_LOT_SIZE = 0.01
POINT_VALUE_PER_STANDARD_LOT_GOLD = 100

# --- Helper Functions (Keep as they are) ---
def get_mt5_data(symbol, timeframe, start_date, end_date):
    print(f"Fetching {symbol} data for timeframe {timeframe_to_string(timeframe)} from {start_date} to {end_date}...")
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"No data returned from MT5 for {symbol}, timeframe {timeframe_to_string(timeframe)}. Error: {mt5.last_error()}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['Timestamp'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('Timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df[f'ema_{EMA_PERIOD}'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    print(f"Loaded {len(df)} rows for {symbol} - {timeframe_to_string(timeframe)}.")
    return df

def timeframe_to_string(tf_int):
    mapping = {
        mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1"
    }
    return mapping.get(tf_int, str(tf_int))

def get_higher_tf_context(df_higher, current_m5_timestamp):
    context = df_higher[df_higher.index <= current_m5_timestamp]
    if not context.empty:
        return context.iloc[-1]
    return None

def identify_fvg(m5_df, current_eval_candle_idx, pattern_candle_count=3):
    if current_eval_candle_idx < pattern_candle_count: return None, None, None, None
    idx_c3 = current_eval_candle_idx - 1
    idx_c2 = current_eval_candle_idx - 2
    idx_c1 = current_eval_candle_idx - 3
    if idx_c1 < 0: return None, None, None, None
    candle1_data = m5_df.iloc[idx_c1]
    candle2_data = m5_df.iloc[idx_c2]
    candle3_data = m5_df.iloc[idx_c3]
    if candle3_data['Low'] > candle1_data['High']:
        fvg_low_boundary = candle1_data['High']
        fvg_high_boundary = candle3_data['Low']
        fvg_height = abs(fvg_high_boundary - fvg_low_boundary)
        return fvg_low_boundary, fvg_high_boundary, 'bullish', fvg_height
    if candle3_data['High'] < candle1_data['Low']:
        fvg_low_boundary = candle3_data['High']
        fvg_high_boundary = candle1_data['Low']
        fvg_height = abs(fvg_high_boundary - fvg_low_boundary)
        return fvg_low_boundary, fvg_high_boundary, 'bearish', fvg_height
    return None, None, None, None

def find_recent_h1_key_level(h1_df, current_h1_timestamp, lookback, is_bullish_bias):
    relevant_h1_data = h1_df[h1_df.index < current_h1_timestamp].tail(lookback)
    if len(relevant_h1_data) < 3: return None
    data_for_swing = relevant_h1_data[:-2] # Use a portion of the lookback to find the swing
    if data_for_swing.empty: return None
    if is_bullish_bias: return data_for_swing['High'].max()
    else: return data_for_swing['Low'].min()
    # This return None is technically not needed if the above logic always returns or handles empty.
    # However, it's good practice for functions to have a clear return path.
    return None


# --- Main Script ---
if not initialize_mt5_connection_bt():
    print("Exiting due to MT5 initialization failure."); exit()

current_balance_usd = STARTING_BALANCE_USD
gross_profit_usd = 0
gross_loss_usd = 0

try:
    df_h4 = get_mt5_data(SYMBOL, TIMEFRAME_H4, BACKTEST_START_DATE, BACKTEST_END_DATE)
    df_h1 = get_mt5_data(SYMBOL, TIMEFRAME_H1, BACKTEST_START_DATE, BACKTEST_END_DATE)
    df_m5 = get_mt5_data(SYMBOL, TIMEFRAME_M5, BACKTEST_START_DATE, BACKTEST_END_DATE)
    if df_h4.empty or df_h1.empty or df_m5.empty:
        print("Exiting: Could not load all necessary data."); shutdown_mt5_bt(); exit()

    trades = []; open_trade = None
    min_start_time_all_tfs = max(df_h4.index.min(), df_h1.index.min(), df_m5.index.min())
    df_m5_filtered = df_m5[df_m5.index >= min_start_time_all_tfs].copy()
    h1_key_level_identified_price = None; h1_key_level_broken = False; h1_key_level_retested = False; h1_key_level_type = None
    loop_start_index = max(FVG_PATTERN_CANDLE_COUNT, H1_KEY_LEVEL_LOOKBACK)

    for i in range(loop_start_index, len(df_m5_filtered)):
        current_m5_candle = df_m5_filtered.iloc[i]; current_m5_timestamp = df_m5_filtered.index[i]
        h4_context = get_higher_tf_context(df_h4, current_m5_timestamp)
        h1_context = get_higher_tf_context(df_h1, current_m5_timestamp)
        if h4_context is None or h1_context is None: continue

        if open_trade:
            pnl_this_trade_usd = 0; trade_closed = False; pnl_points_trade = 0; status = ''; exit_price = 0
            if open_trade['type'] == 'long':
                if current_m5_candle['Low'] <= open_trade['sl']: pnl_points_trade = (open_trade['sl'] - open_trade['entry_price']); trade_closed = True; status = 'sl'; exit_price = open_trade['sl']
                elif current_m5_candle['High'] >= open_trade['tp']: pnl_points_trade = (open_trade['tp'] - open_trade['entry_price']); trade_closed = True; status = 'tp'; exit_price = open_trade['tp']
            elif open_trade['type'] == 'short':
                if current_m5_candle['High'] >= open_trade['sl']: pnl_points_trade = (open_trade['entry_price'] - open_trade['sl']); trade_closed = True; status = 'sl'; exit_price = open_trade['sl']
                elif current_m5_candle['Low'] <= open_trade['tp']: pnl_points_trade = (open_trade['entry_price'] - open_trade['tp']); trade_closed = True; status = 'tp'; exit_price = open_trade['tp']
            if trade_closed:
                pnl_this_trade_usd = pnl_points_trade * POINT_VALUE_PER_STANDARD_LOT_GOLD * FIXED_LOT_SIZE
                current_balance_usd += pnl_this_trade_usd
                if pnl_this_trade_usd > 0: gross_profit_usd += pnl_this_trade_usd
                else: gross_loss_usd += abs(pnl_this_trade_usd)
                trades.append({'timestamp': current_m5_timestamp, 'type': open_trade['type'], 'status': status, 'pnl_points': pnl_points_trade, 'pnl_usd': pnl_this_trade_usd, 'entry': open_trade['entry_price'], 'exit': exit_price, 'balance_after_trade': current_balance_usd})
                open_trade = None
            if open_trade: continue

        h4_price = h4_context['Close']; h4_ema = h4_context[f'ema_{EMA_PERIOD}']
        is_h4_bullish = h4_price > h4_ema and (h4_price - h4_ema) / h4_ema > H4_EMA_SEPARATION_THRESHOLD
        is_h4_bearish = h4_price < h4_ema and (h4_ema - h4_price) / h4_ema > H4_EMA_SEPARATION_THRESHOLD
        h4_bias = 'bullish' if is_h4_bullish else ('bearish' if is_h4_bearish else None)
        if not h4_bias:
            h1_key_level_identified_price = None; h1_key_level_broken = False; h1_key_level_retested = False; continue

        h1_current_price_of_context_candle = h1_context['Close']
        if not h1_key_level_identified_price:
            h1_key_level_identified_price = find_recent_h1_key_level(df_h1, h1_context.name, H1_KEY_LEVEL_LOOKBACK, h4_bias == 'bullish')
            h1_key_level_broken = False; h1_key_level_retested = False
            if h1_key_level_identified_price: h1_key_level_type = 'resistance' if h4_bias == 'bullish' else 'support'
        if h1_key_level_identified_price and not h1_key_level_broken:
            if (h4_bias == 'bullish' and h1_current_price_of_context_candle > h1_key_level_identified_price) or \
               (h4_bias == 'bearish' and h1_current_price_of_context_candle < h1_key_level_identified_price):
                h1_key_level_broken = True
        if h1_key_level_broken and not h1_key_level_retested:
            distance_to_key_level = abs(current_m5_candle['Close'] - h1_key_level_identified_price)
            proximity_threshold = h1_key_level_identified_price * H1_RETEST_PROXIMITY_PERCENT
            if distance_to_key_level < proximity_threshold: h1_key_level_retested = True
        if not (h1_key_level_identified_price and h1_key_level_broken and h1_key_level_retested): continue

        fvg_low_b, fvg_high_b, fvg_type, fvg_height = identify_fvg(df_m5_filtered, i, pattern_candle_count=FVG_PATTERN_CANDLE_COUNT)

        if fvg_type and not open_trade:
            if fvg_height is not None and fvg_height >= MIN_FVG_HEIGHT_PRICE_UNITS:
                entry_price = None; sl_price = None; tp_price = None; trade_type_to_open = None; risk_amount_points = 0
                idx_c1_fvg = i - FVG_PATTERN_CANDLE_COUNT
                idx_c2_fvg = i - (FVG_PATTERN_CANDLE_COUNT - 1)
                idx_c3_fvg = i - (FVG_PATTERN_CANDLE_COUNT - 2)

                if h4_bias == 'bullish' and fvg_type == 'bullish':
                    if current_m5_candle['Low'] <= fvg_high_b:
                        entry_price = fvg_high_b
                        sl_structure_low = min(df_m5_filtered.iloc[idx_c1_fvg]['Low'], df_m5_filtered.iloc[idx_c2_fvg]['Low'], df_m5_filtered.iloc[idx_c3_fvg]['Low'])
                        sl_price = min(sl_structure_low, h1_key_level_identified_price) - STOP_LOSS_OFFSET_PRICE_UNITS_GOLD
                        if entry_price > sl_price:
                            risk_amount_points = entry_price - sl_price
                            if risk_amount_points > 0.0001 :
                                tp_price = entry_price + (risk_amount_points * RISK_REWARD_RATIO); trade_type_to_open = 'long'
                elif h4_bias == 'bearish' and fvg_type == 'bearish':
                    if current_m5_candle['High'] >= fvg_low_b:
                        entry_price = fvg_low_b
                        sl_structure_high = max(df_m5_filtered.iloc[idx_c1_fvg]['High'], df_m5_filtered.iloc[idx_c2_fvg]['High'], df_m5_filtered.iloc[idx_c3_fvg]['High'])
                        sl_price = max(sl_structure_high, h1_key_level_identified_price) + STOP_LOSS_OFFSET_PRICE_UNITS_GOLD
                        if sl_price > entry_price:
                            risk_amount_points = sl_price - entry_price
                            if risk_amount_points > 0.0001:
                                tp_price = entry_price - (risk_amount_points * RISK_REWARD_RATIO); trade_type_to_open = 'short'

                if trade_type_to_open and entry_price and sl_price and tp_price:
                    open_trade = {'type': trade_type_to_open, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price, 'entry_time': current_m5_timestamp, 'h1_key_level_at_entry': h1_key_level_identified_price}
                    print(f"{current_m5_timestamp}: Opened {trade_type_to_open} @ {entry_price:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f} (FVG low: {fvg_low_b:.2f}-FVG high: {fvg_high_b:.2f}, Height: {fvg_height:.2f})")
                    h1_key_level_identified_price = None; h1_key_level_broken = False; h1_key_level_retested = False

    print("\n--- Backtest Results ---")
    print(f"Symbol: {SYMBOL}"); print(f"Backtest Period: {BACKTEST_START_DATE.date()} to {BACKTEST_END_DATE.date()}")
    print(f"Starting Balance (USD): {STARTING_BALANCE_USD:.2f}")
    if not trades: print("No trades were executed."); print(f"Ending Balance (USD): {current_balance_usd:.2f}")
    else:
        results_df = pd.DataFrame(trades)
        if not results_df.empty:
            results_df.set_index('timestamp', inplace=True)
            wins_df = results_df[results_df['status'] == 'tp']; losses_df = results_df[results_df['status'] == 'sl']
            win_rate = (len(wins_df) / len(results_df)) * 100 if len(results_df) > 0 else 0
            total_pnl_usd_val = results_df['pnl_usd'].sum()
            profit_factor = (gross_profit_usd / gross_loss_usd) if gross_loss_usd != 0 else (float('inf') if gross_profit_usd > 0 else 0)
            print(f"Total Trades: {len(results_df)}"); print(f"Wins: {len(wins_df)}"); print(f"Losses: {len(losses_df)}")
            print(f"Win Rate: {win_rate:.2f}%"); print(f"Total P&L (USD): {total_pnl_usd_val:.2f}")
            print(f"Gross Profit (USD): {gross_profit_usd:.2f}"); print(f"Gross Loss (USD): {gross_loss_usd:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}"); print(f"Ending Balance (USD): {current_balance_usd:.2f}")
            print("\nIndividual Trades (USD P&L):"); print(results_df[['type', 'status', 'entry', 'exit', 'pnl_usd', 'balance_after_trade']])
        else: print("No trades recorded in results_df."); print(f"Ending Balance (USD): {current_balance_usd:.2f}")
finally:
    shutdown_mt5_bt() 