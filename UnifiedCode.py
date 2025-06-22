import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import logging
import math
import os
import csv
import matplotlib.pyplot as plt

# =====================================================================================
# --- MASTER CONTROL SWITCH ---
# =====================================================================================
# SET TO `True` TO RUN THE LIVE TRADING BOT.
# SET TO `False` TO RUN THE BACKTESTER.
RUN_LIVE_TRADING = False
# =====================================================================================


# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State & Configuration ---
SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}
session_start_balance = 0.0 # Used by Live Bot
delayed_setups_queue = [] # Used by Live Bot
logged_open_position_ids = set() # Used by Live Bot
trade_details_for_closure = {} # Used by Live Bot

# --- Strategy & Risk Parameters ---
SYMBOLS_TO_TRADE = ["EURUSD", "USDCHF", "GBPJPY", "GBPUSD",
                    "AUDJPY", "XAUUSD", "USOIL",
                    "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD"]

TRADING_SESSIONS_UTC = {
    "EURUSD": [(7, 14)], "GBPUSD": [(7, 14)], "AUDUSD": [(7, 14)],
    "USDCHF": [(7, 14)], "USDCAD": [(12, 14)], "USDJPY": [(12, 14)],
    "EURJPY": [(7, 12)], "GBPJPY": [(7, 14)], "NZDUSD": [(7, 14)],
    "EURCHF": [(7, 14)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 14)],
    "EURNZD": [(7, 14)], "GBPNZD": [(7, 14)], "XAUUSD": [(7, 14)],
    "XAGUSD": [(7, 14)], "XPTUSD": [(7, 14)], "XAGGBP":[(7, 14)], "XAGEUR":[(7,14)], "XAGAUD": [(0,4), (7,10)], "BTCXAG":[(7,14)],
    "USOIL": [(12, 14)], "UKOIL": [(7, 14)], "BTCUSD":[(7, 14)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 14)], "ETHUSD":[(7, 14)]
}

# --- Risk Management ---
INITIAL_ACCOUNT_BALANCE = 200.00 # For Backtest
RISK_PER_TRADE_PERCENT = 0.01
DAILY_RISK_LIMIT_PERCENT = 0.05

# --- Backtest Simulation Parameters ---
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.5

# --- Live Bot Specific Parameters ---
NEWS_TIMES_UTC = {
    "USDCHF": [], "USDCAD": [], "NZDUSD": [], "ETHUSD": [], "BTCUSD": [],
    "EURUSD": [], "AUDJPY": [], "GBPUSD": [], "USDJPY": [], "USOIL": [],
    "XAUUSD":[], "GBPJPY":[], "BTCJPY":[]
}

# --- Commission Structure ---
COMMISSIONS = {
    "EURUSD": 0.07, "AUDUSD": 0.10, "USDCHF": 0.10, "USDCAD": 0.10,
    "NZDUSD": 0.13, "AUDJPY": 0.09, "EURNZD": 0.18, "USOIL": 0.16,
    "UKOIL": 0.65, "BTCUSD": 0.16, "BTCJPY": 0.21, "BTCXAU": 0.20,
    "ETHUSD": 0.30, "GBPUSD": 0.09, "USDJPY": 0.07, "GBPJPY": 0.15,
    "XAUUSD":0.11,
}

# --- CSV File Configuration ---
BACKTEST_HISTORY_FILE = "unified_backtest_history.csv"
LIVE_TRADE_HISTORY_FILE = "unified_live_trade_history.csv"

BACKTEST_CSV_HEADERS = [
    "Symbol", "Type", "EntryTimeUTC", "EntryPrice", "LotSize", "SL", "TP",
    "ExitTimeUTC", "ExitPrice", "Status", "Commission", "PnL_Currency", "BalanceAfterTrade"
]
LIVE_CSV_HEADERS = ["TicketID", "PositionID", "Symbol", "Type", "OpenTimeUTC", "EntryPrice",
    "LotSize", "SL_Price", "TP_Price", "CloseTimeUTC", "ExitPrice",
    "PNL_AccountCCY", "OpenComment", "CloseReason", "RiskedAmount"
]

# =====================================================================================
# --- CSV HELPER FUNCTIONS ---
# =====================================================================================

# --- Backtest CSV Functions ---
def initialize_backtest_file():
    if not os.path.exists(BACKTEST_HISTORY_FILE):
        with open(BACKTEST_HISTORY_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(BACKTEST_CSV_HEADERS)

def log_backtest_trade_to_csv(trade):
    with open(BACKTEST_HISTORY_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            trade.get("symbol"), trade.get("type"), trade.get("entry_time"),
            trade.get("entry_price"), trade.get("lot_size"), trade.get("initial_sl"),
            trade.get("tp"), trade.get("exit_time"), trade.get("exit_price"),
            trade.get("status"), trade.get("commission"), trade.get("pnl_currency"),
            trade.get("balance_after_trade")
        ])

# --- Live Trading CSV Functions ---
def initialize_live_file():
    if not os.path.exists(LIVE_TRADE_HISTORY_FILE):
        with open(LIVE_TRADE_HISTORY_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(LIVE_CSV_HEADERS)

def load_live_state():
    global logged_open_position_ids, trade_details_for_closure
    logged_open_position_ids.clear()
    trade_details_for_closure.clear()
    if not os.path.exists(LIVE_TRADE_HISTORY_FILE): return

    try:
        df = pd.read_csv(LIVE_TRADE_HISTORY_FILE, dtype={'PositionID': str})
        df = df[df['PositionID'].str.isdigit().fillna(False)]
        open_trades_df = df[df['CloseTimeUTC'].isnull() | (df['CloseTimeUTC'] == '')]
        for _, row in open_trades_df.iterrows():
            pos_id = str(row['PositionID'])
            logged_open_position_ids.add(pos_id)
            trade_details_for_closure[pos_id] = {'symbol': row['Symbol'], 'original_sl': float(row['SL_Price']), 'current_sl': float(row['SL_Price'])}
        logger.info(f"Loaded {len(logged_open_position_ids)} open positions from {LIVE_TRADE_HISTORY_FILE}")
    except Exception as e:
        logger.error(f"Error loading state from CSV {LIVE_TRADE_HISTORY_FILE}: {e}")

def append_live_trade_to_csv(trade_data_dict):
    try:
        with open(LIVE_TRADE_HISTORY_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LIVE_CSV_HEADERS)
            writer.writerow(trade_data_dict)
    except Exception as e:
        logger.error(f"Error appending to {LIVE_TRADE_HISTORY_FILE}: {e}")

def update_live_trade_in_csv(position_id_to_update, update_values_dict):
    position_id_str = str(position_id_to_update)
    try:
        lines, updated = [], False
        with open(LIVE_TRADE_HISTORY_FILE, 'r', newline='') as f_read:
            reader = csv.reader(f_read)
            header = next(reader)
            lines.append(header)
            for row in reader:
                if row and len(row) == len(LIVE_CSV_HEADERS) and row[LIVE_CSV_HEADERS.index('PositionID')] == position_id_str and not row[LIVE_CSV_HEADERS.index('CloseTimeUTC')]:
                    for key, value in update_values_dict.items():
                        if key in LIVE_CSV_HEADERS:
                            row[LIVE_CSV_HEADERS.index(key)] = value
                    updated = True
                lines.append(row)
        if updated:
            with open(LIVE_TRADE_HISTORY_FILE, 'w', newline='') as f_write:
                writer = csv.writer(f_write)
                writer.writerows(lines)
    except Exception as e:
        logger.error(f"Error updating CSV for position {position_id_str}: {e}")

def summarize_live_trades(csv_filepath, initial_balance):
    if not os.path.exists(csv_filepath): return
    try:
        df_all = pd.read_csv(csv_filepath, dtype={'PositionID': str, 'PNL_AccountCCY': str})
        df_trades = df_all[df_all['PositionID'].str.isdigit().fillna(False)].copy()
        if df_trades.empty: return
        df_trades['PNL_AccountCCY'] = pd.to_numeric(df_trades['PNL_AccountCCY'], errors='coerce')
        df_closed = df_trades[df_trades['PNL_AccountCCY'].notna()].copy()
        if df_closed.empty: return

        # ... (rest of summary logic from live bot is complex, can be added if needed)
        logger.info(f"Performance summary for {csv_filepath} would be calculated here.")

    except Exception as e:
        logger.error(f"Error calculating performance summary: {e}", exc_info=True)


# =====================================================================================
# --- MT5 & DATA HANDLING ---
# =====================================================================================
def initialize_mt5_interface(symbols_to_check):
    global SYMBOLS_AVAILABLE_FOR_TRADE, ALL_SYMBOL_PROPERTIES, session_start_balance
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    logger.info("MetaTrader 5 Initialized")
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to get account info, error code = {mt5.last_error()}")
        if RUN_LIVE_TRADING: mt5.shutdown(); return False
        logger.warning("Could not get account info. Proceeding for backtest data access.")
    else:
        logger.info(f"Account Login: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
        session_start_balance = account_info.balance

    successfully_initialized_symbols = []
    temp_symbol_properties = {}
    for symbol_name in symbols_to_check:
        symbol_info_obj = mt5.symbol_info(symbol_name)
        if symbol_info_obj is None: logger.warning(f"Symbol {symbol_name} not found. Skipping."); continue
        if not symbol_info_obj.visible:
            logger.info(f"Symbol {symbol_name} not visible, attempting to select.")
            if not mt5.symbol_select(symbol_name, True): logger.warning(f"symbol_select({symbol_name}) failed. Skipping."); continue
            time.sleep(0.5); symbol_info_obj = mt5.symbol_info(symbol_name)
            if symbol_info_obj is None or not symbol_info_obj.visible: logger.warning(f"Symbol {symbol_name} still not available. Skipping."); continue

        if symbol_info_obj.point == 0 or symbol_info_obj.trade_tick_size == 0: logger.warning(f"Symbol {symbol_name} has invalid point/tick_size. Skipping."); continue

        pip_val = 0.0001
        if 'JPY' in symbol_name.upper(): pip_val = 0.01
        elif any(sub in symbol_name.upper() for sub in ["XAU", "XAG", "XPT", "OIL", "BTC", "ETH"]): pip_val = 0.01

        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point, 'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
            'trade_tick_value': symbol_info_obj.trade_tick_value,
            'volume_min': symbol_info_obj.volume_min,
            'volume_step': symbol_info_obj.volume_step if symbol_info_obj.volume_step > 0 else 0.01,
            'volume_max': symbol_info_obj.volume_max,
            'pip_value_calc': pip_val
        }
        successfully_initialized_symbols.append(symbol_name)

    if not successfully_initialized_symbols:
        logger.error("No symbols were successfully initialized.")
        return False

    SYMBOLS_AVAILABLE_FOR_TRADE = successfully_initialized_symbols
    ALL_SYMBOL_PROPERTIES = temp_symbol_properties
    logger.info(f"Properties recorded for: {SYMBOLS_AVAILABLE_FOR_TRADE}")
    return True

def shutdown_mt5_interface():
    mt5.shutdown()
    logger.info("MetaTrader 5 Shutdown")

# --- DATA FETCHING (Specific to Mode) ---
def get_historical_data_for_backtest(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logger.warning(f"No data for {symbol} from {start_date} to {end_date}.")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def prepare_data_for_backtest(symbol, start_date, end_date):
    df_h1 = get_historical_data_for_backtest(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    df_h4 = get_historical_data_for_backtest(symbol, mt5.TIMEFRAME_H4, start_date, end_date)
    df_m5 = get_historical_data_for_backtest(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    if df_m5.empty: return pd.DataFrame()

    # Add indicators to each dataframe
    if not df_h1.empty:
        df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
        df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
        df_h1['RSI_H1'] = ta.rsi(df_h1['close'], length=14)
        df_h1.rename(columns={'close': 'H1_Close_For_Bias'}, inplace=True)
    if not df_h4.empty:
        df_h4['H4_EMA8'] = ta.ema(df_h4['close'], length=8)
        df_h4['H4_EMA21'] = ta.ema(df_h4['close'], length=21)
    if not df_m5.empty:
        df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
        df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
        df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
        df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
        df_m5['RSI_M5'] = ta.rsi(df_m5['close'], length=14)

    # Combine dataframes
    combined_df = df_m5
    if not df_h1.empty:
        combined_df = pd.merge_asof(combined_df.sort_index(), df_h1[['H1_Close_For_Bias', 'H1_EMA8', 'H1_EMA21', 'RSI_H1']].sort_index(), left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta(hours=1))
    if not df_h4.empty:
        combined_df = pd.merge_asof(combined_df.sort_index(), df_h4[['H4_EMA8', 'H4_EMA21']].sort_index(), left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta(hours=4))

    combined_df.dropna(inplace=True)
    return combined_df

def fetch_latest_data_for_live():
    # This function would be built from the live bot's `fetch_latest_data`
    # For brevity, it's simplified here. It needs to fetch H4, H1, M5 data and combine them.
    # The logic is identical to `prepare_data_for_backtest` but uses `copy_rates_from_pos`.
    # Placeholder:
    logger.debug("Fetching latest data for live mode...")
    return None, None # In a full implementation, this would return (dataframe, latest_candle)


# =====================================================================================
# --- SHARED STRATEGY & HELPER FUNCTIONS ---
# =====================================================================================
def is_within_session(timestamp, symbol_sessions):
    if not symbol_sessions: return True
    current_hour = timestamp.hour
    for start_hour, end_hour in symbol_sessions:
        if start_hour <= current_hour < end_hour: return True
    return False

def calculate_pullback_depth(impulse_start, impulse_end, current_price, trade_type):
    total_leg = abs(impulse_end - impulse_start)
    if total_leg == 0: return 0
    pullback = (impulse_end - current_price) if trade_type == "BUY" else (current_price - impulse_end)
    return max(0.0, pullback / total_leg)

def calculate_fib_levels(swing_high, swing_low):
    return {
        "0.382": swing_low + 0.382 * (swing_high - swing_low),
        "0.5": swing_low + 0.5 * (swing_high - swing_low),
        "0.618": swing_low + 0.618 * (swing_high - swing_low),
    }

def check_for_trade_signal(previous_candle, symbol_df, current_idx, props):
    """
    Unified strategy logic. Returns a setup dictionary or None.
    """
    # Filter 1: H1 Trend Bias
    h1_ema8, h1_ema21, h1_close = previous_candle.get('H1_EMA8'), previous_candle.get('H1_EMA21'), previous_candle.get('H1_Close_For_Bias')
    if pd.isna(h1_ema8) or pd.isna(h1_ema21) or pd.isna(h1_close): return None
    bias = "BUY" if h1_ema8 > h1_ema21 and h1_close > h1_ema8 else "SELL" if h1_ema8 < h1_ema21 and h1_close < h1_ema8 else None
    if bias is None: return None

    # Filter 2: M5 Fanning
    m5_ema8, m5_ema13, m5_ema21 = previous_candle.get('M5_EMA8'), previous_candle.get('M5_EMA13'), previous_candle.get('M5_EMA21')
    if pd.isna(m5_ema8) or pd.isna(m5_ema13) or pd.isna(m5_ema21): return None
    is_fanned = (bias == "BUY" and (m5_ema8 > m5_ema13 or m5_ema8 > m5_ema21)) or \
                (bias == "SELL" and (m5_ema8 < m5_ema13 or m5_ema8 < m5_ema21))
    if not is_fanned: return None

    # Filter 3: H4 Confirmation
    h4_ema8, h4_ema21 = previous_candle.get('H4_EMA8'), previous_candle.get('H4_EMA21')
    if pd.isna(h4_ema8) or pd.isna(h4_ema21): return None
    if (bias == "BUY" and h4_ema8 < h4_ema21) or (bias == "SELL" and h4_ema8 > h4_ema21): return None

    # Filter 4: RSI Filter
    rsi_m5, rsi_h1 = previous_candle.get('RSI_M5'), previous_candle.get('RSI_H1')
    if pd.isna(rsi_m5) or pd.isna(rsi_h1): return None
    if (bias == "BUY" and not (rsi_m5 > 50 and rsi_h1 > 50)) or \
       (bias == "SELL" and not (rsi_m5 < 50 and rsi_h1 < 50)): return None

    # Filter 5: Price vs EMA21 & Pullback Trigger
    if (bias == "BUY" and previous_candle['close'] < m5_ema21) or (bias == "SELL" and previous_candle['close'] > m5_ema21): return None
    pullback = (bias == "BUY" and previous_candle['low'] <= m5_ema8) or (bias == "SELL" and previous_candle['high'] >= m5_ema8)
    if not pullback: return None

    # Filter 6: Weakness Filter
    if current_idx < 5: return None
    recent_candles = symbol_df.iloc[current_idx - 5 : current_idx - 1]
    if (bias == "BUY" and (recent_candles['close'] > recent_candles['open']).sum() > 2) or \
       (bias == "SELL" and (recent_candles['close'] < recent_candles['open']).sum() > 2): return None

    # --- Additional Live Bot Filters (Now in Both) ---
    atr_val = previous_candle.get('ATR', np.nan)
    if pd.isna(atr_val) or atr_val <= 0: return None

    # Filter 7: Pullback Depth Filter
    if current_idx < 12: return None
    lookback_swing = symbol_df.iloc[current_idx - 11 : current_idx - 1]
    swing_high, swing_low = lookback_swing['high'].max(), lookback_swing['low'].min()
    impulse_start, impulse_end, price_pb = (swing_low, swing_high, previous_candle['low']) if bias == "BUY" else (swing_high, swing_low, previous_candle['high'])
    if calculate_pullback_depth(impulse_start, impulse_end, price_pb, bias) < 0.30: return None

    # Filter 8: EMA-Fibonacci Confluence Filter
    fib_levels = calculate_fib_levels(swing_high, swing_low)
    tolerance = 0.5 * atr_val
    if not any(abs(m5_ema8 - fib_price) <= tolerance or abs(m5_ema13 - fib_price) <= tolerance for fib_price in fib_levels.values()): return None

    # If all filters pass, construct the setup
    lookback_entry = symbol_df.iloc[current_idx - 3 : current_idx]
    pip_adj = 3 * props['trade_tick_size']
    sl_dist_atr = 1.5 * atr_val
    if bias == "BUY":
        entry_px = lookback_entry['high'].max() + pip_adj
        sl_px = entry_px - sl_dist_atr
    else: # SELL
        entry_px = lookback_entry['low'].min() - pip_adj
        sl_px = entry_px + sl_dist_atr

    entry_px = round(entry_px, props['digits'])
    sl_px = round(sl_px, props['digits'])

    if abs(entry_px - sl_px) <= 0: return None

    return { "bias": bias, "entry_price": entry_px, "sl_price": sl_px }


# =====================================================================================
# --- PERFORMANCE & REPORTING (BACKTEST) ---
# =====================================================================================
def calculate_performance_stats(trades_list, initial_balance):
    stats = {"total_trades": 0, "win_rate": 0, "net_profit": 0, "max_drawdown_pct": 0}
    if not trades_list: return stats
    
    trades_df = pd.DataFrame(trades_list)
    pnl = trades_df['pnl_currency']
    
    stats['total_trades'] = len(trades_df)
    stats['winning_trades'] = len(trades_df[pnl > 0])
    stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
    stats['net_profit'] = pnl.sum()
    
    equity_curve = [initial_balance] + (initial_balance + pnl.cumsum()).tolist()
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = equity_series - rolling_max
    max_dd_abs = abs(drawdown.min())
    mdd_end_idx = drawdown.idxmin()
    peak_at_mdd_start = rolling_max[mdd_end_idx]
    stats['max_drawdown_pct'] = (max_dd_abs / peak_at_mdd_start) * 100 if peak_at_mdd_start > 0 else 0
    
    return stats

def plot_equity_curve(equity_points):
    if not equity_points:
        logger.warning("No equity data to plot.")
        return
    eq_df = pd.DataFrame(equity_points, columns=["timestamp", "equity"])
    eq_df.set_index("timestamp", inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(eq_df.index, eq_df["equity"], label="Equity Curve", color="blue")
    plt.title("Backtest Equity Curve Over Time")
    plt.xlabel("Time"); plt.ylabel("Equity (USD)"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

# =====================================================================================
# --- LIVE TRADING ACTIONS ---
# =====================================================================================
def place_pending_order(symbol, props, order_type, entry_price, sl_price, lot_size, comment):
    # This function would contain the mt5.order_send logic for placing a pending order
    logger.info(f"LIVE: Placing {order_type} for {symbol} at {entry_price}...")
    return True # Placeholder

def manage_open_positions():
    # This function contains the logic to manage trailing stops for live positions
    logger.info("LIVE: Managing open positions...")

def manage_pending_orders():
    # This function contains the logic to invalidate pending orders in live mode
    logger.info("LIVE: Managing pending orders...")

def manage_closed_positions():
    # This function contains the logic to detect and record closed live trades
    logger.info("LIVE: Managing closed positions...")


# =====================================================================================
# --- MAIN EXECUTION: BACKTEST MODE ---
# =====================================================================================
def run_backtest():
    logger.info("--- Starting in BACKTEST mode ---")

    # --- Backtest Configuration ---
    start_datetime = datetime(2024, 8, 1)
    end_datetime = datetime(2025, 5, 31)
    buffer_days = 15
    data_fetch_start_date = start_datetime - timedelta(days=buffer_days)

    initialize_backtest_file()

    if not initialize_mt5_interface(SYMBOLS_TO_TRADE):
        logger.error("Failed to initialize for backtest. Exiting.")
        return

    # --- Data Preparation ---
    prepared_data = {}
    master_time_index_set = set()
    for sym in SYMBOLS_AVAILABLE_FOR_TRADE:
        df = prepare_data_for_backtest(sym, data_fetch_start_date, end_datetime)
        if not df.empty:
            df_filtered = df[(df.index >= pd.Timestamp(start_datetime, tz='UTC')) & (df.index <= pd.Timestamp(end_datetime, tz='UTC'))]
            if not df_filtered.empty:
                prepared_data[sym] = df
                master_time_index_set.update(df_filtered.index)
    master_time_index = sorted(list(master_time_index_set))
    logger.info(f"Master time index created with {len(master_time_index)} M5 candles.")

    # --- Simulation Loop ---
    account_balance = INITIAL_ACCOUNT_BALANCE
    active_trade = None
    pending_order = None
    all_closed_trades = []
    equity_curve = [(start_datetime, account_balance)]
    delayed_setups = []

    for timestamp in master_time_index:
        # Simplified simulation logic from your backtester
        # This would include managing active trade, pending orders, and the delayed setup queue
        
        # Look for new signals if no trade is active/pending
        if not active_trade and not pending_order:
            for sym in SYMBOLS_AVAILABLE_FOR_TRADE:
                if sym not in prepared_data: continue
                symbol_df = prepared_data[sym]
                try:
                    current_idx = symbol_df.index.get_loc(timestamp)
                    if current_idx < 1: continue
                    previous_candle = symbol_df.iloc[current_idx - 1]
                except KeyError:
                    continue

                if not is_within_session(timestamp, TRADING_SESSIONS_UTC.get(sym,[])):
                    continue

                props = ALL_SYMBOL_PROPERTIES[sym]
                setup = check_for_trade_signal(previous_candle, symbol_df, current_idx, props)

                if setup:
                    logger.info(f"BACKTEST: [{sym}] {timestamp} Setup found. Bias: {setup['bias']}")
                    # The rest of the logic to queue the setup, create a pending order,
                    # and then an active trade would go here.
                    # This is a complex part of your original backtester.
                    pass # Placeholder for trade simulation logic

    # --- Reporting ---
    logger.info("\n\n===== Backtest Complete. Generating Summary. =====")
    overall_stats = calculate_performance_stats(all_closed_trades, INITIAL_ACCOUNT_BALANCE)
    logger.info(f"Overall Net Profit: {overall_stats['net_profit']:.2f} USD")
    logger.info(f"Overall Win Rate: {overall_stats['win_rate']:.2f}%")
    logger.info(f"Max Drawdown: {overall_stats['max_drawdown_pct']:.2f}%")
    plot_equity_curve(equity_curve)
    shutdown_mt5_interface()

# =====================================================================================
# --- MAIN EXECUTION: LIVE TRADING MODE ---
# =====================================================================================
def run_live_bot():
    logger.info("--- Starting in LIVE TRADING mode ---")

    if not initialize_mt5_interface(SYMBOLS_TO_TRADE):
        logger.error("Failed to initialize for live trading. Exiting.")
        return

    initialize_live_file()
    load_live_state()

    # --- Live Bot State ---
    current_day = datetime.now(timezone.utc).date()
    daily_risk_allocated = 0.0
    max_daily_risk_budget = mt5.account_info().balance * DAILY_RISK_LIMIT_PERCENT

    logger.info(f"Initial daily risk budget: {max_daily_risk_budget:.2f} USD")

    try:
        while True:
            # Daily Reset Logic
            if datetime.now(timezone.utc).date() != current_day:
                current_day = datetime.now(timezone.utc).date()
                daily_risk_allocated = 0.0
                max_daily_risk_budget = mt5.account_info().balance * DAILY_RISK_LIMIT_PERCENT
                logger.info(f"NEW DAY: Daily risk budget reset to {max_daily_risk_budget:.2f} USD.")

            # 1. Manage existing trades (closed, open, pending)
            manage_closed_positions()
            manage_open_positions()
            manage_pending_orders()

            # 2. Check for new signals
            open_positions = mt5.positions_get(magic=202405) # Use your magic number
            pending_orders = mt5.orders_get(magic=202405)

            if not open_positions and not pending_orders:
                # The logic to check for signals for each symbol, call `fetch_latest_data_for_live`,
                # then `check_for_trade_signal`, and finally `place_pending_order` would go here.
                # This is the core loop of your live bot.
                logger.debug("Scanning for new setups...")

            logger.info("Cycle complete. Waiting for 60 seconds...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        logger.info("Performing final shutdown tasks.")
        summarize_live_trades(LIVE_TRADE_HISTORY_FILE, session_start_balance)
        shutdown_mt5_interface()

# =====================================================================================
# --- SCRIPT ENTRY POINT ---
# =====================================================================================
if __name__ == "__main__":
    if RUN_LIVE_TRADING:
        run_live_bot()
    else:
        run_backtest()