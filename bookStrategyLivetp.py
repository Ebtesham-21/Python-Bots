# --- START OF FILE bookStrategyLivetp.py ---

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta  # For EMAs, ATR, and ADX
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import logging
import math
import os    # For file operations
import csv   # For CSV writing
import io    # For in-memory file operations
from scipy.signal import argrelextrema # For dynamic TP calculation

# --- Logger Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables & State ---

SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}

# --- Bot State (populated at runtime) ---

logged_open_position_ids = set()
trade_details_for_closure = {}  # Holds details like original SL for management, and pending TP info
delayed_setups_queue = []  # List of setups waiting for confirmation
session_start_balance = 0.0 # Will be set on initialization

# --- Strategy & Risk Parameters ---

SYMBOLS_TO_TRADE =  ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                           "AUDJPY",   "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF", "CADCHF", "CADJPY", "EURCAD",
                           "GBPCAD", "NZDCAD", "GBPAUD", "GBPNZD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD", "EURAUD",
                       "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD","AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500",
                       "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C",

                             ]
TRADING_SESSIONS_UTC =  { # (start_hour_inclusive, end_hour_exclusive)
                           "EURUSD":[(0, 17)], "USDCHF":[(0, 17)],   "GBPJPY": [ (0, 17)], "GBPUSD": [ (0, 17)],
                           "AUDJPY":[(0, 12)],  "XAUUSD": [(0, 17)], "XAGUSD": [(0, 17)], "EURNZD": [(0, 17)], "NZDUSD": [(0, 17)], "AUDUSD": [ (0, 17)], "USDCAD": [(0, 17)],"USDJPY":[(0,17)], "EURJPY": [ (0, 17)],"EURCHF": [(0, 17)], "CADCHF": [  (0, 17)], "CADJPY": [ (0,17)], "EURCAD":[(0, 17)],
                           "GBPCAD": [(0, 17)], "NZDCAD":[(0,17)], "GBPAUD":[(0, 17)], "GBPNZD":[(0,17)], "GBPCHF":[(0,17)], "AUDCAD":[(0,17)], "AUDCHF":[(0,17)], "AUDNZD":[(0,12)], "EURAUD":[(0,17)],
                           "AAPL": [(10, 17)] , "MSFT": [(10, 17)], "GOOGL": [(10, 17)], "AMZN": [(10, 17)], "NVDA": [(10, 17)], "META": [(10, 17)], "TSLA": [(10, 17)], "AMD": [(10, 17)], "NFLX": [(10, 17)], "US500": [(10, 17)],
                           "USTEC": [(10, 17)],"INTC":[(10, 17)], "MO":[(10, 17)], "BABA":[(10, 17)], "ABT":[(10, 17)], "LI":[(10, 17)], "TME":[(10, 17)], "ADBE":[(10, 17)], "MMM":[(10, 17)], "WMT":[(10, 17)], "PFE":[(10, 17)], "EQIX":[(10, 17)], "F":[(10, 17)], "ORCL":[(10, 17)], "BA":[(10, 17)], "NKE":[(10, 17)], "C":[(10, 17)],

}

TRADING_SESSIONS_UTC["USOIL"] = [(0, 17)]
TRADING_SESSIONS_UTC["UKOIL"] = [(0, 17)]
CRYPTO_SESSIONS_USER = {"BTCUSD":[(0, 17)], "BTCJPY":[(0, 17)], "BTCXAU":[(0, 17)], "ETHUSD":[(0, 17)]}
for crypto_sym, sess_val in CRYPTO_SESSIONS_USER.items():
    TRADING_SESSIONS_UTC[crypto_sym] = sess_val

CRYPTO_SYMBOLS = list(CRYPTO_SESSIONS_USER.keys())

# --- NEW: Define a list of stock symbols for the volume filter ---
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500",
    "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM",
    "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C"
]

RISK_PER_TRADE_PERCENT = 0.01  # Risk 1% of current balance per trade
DAILY_RISK_LIMIT_PERCENT = 0.05 # Daily risk limit of 5% of balance at start of day
# --- ADD THIS NEW SECTION AT THE TOP OF THE SCRIPT ---

# --- NEW: Entry Price Strategy Configuration ---
# Choose one: 'BREAKOUT', 'ATR_BUFFER', 'FIXED_BUFFER', 'NO_BUFFER'
# 'BREAKOUT' is the logic that is perfectly synchronized with your backtester.
ENTRY_PRICE_STRATEGY = 'BREAKOUT' 

# --- Commission Structure ---
COMMISSIONS = {
    "EURUSD":0.07, "USDCHF":0.10,   "GBPJPY":0.15, "GBPUSD":0.09,
    "AUDJPY":0.09,   "EURNZD":0.18, "NZDUSD":0.13, "AUDUSD":0.10, "USDCAD":0.10,"USDJPY":0.07, "EURJPY":0.11,"EURCHF":0.17, "CADCHF":0.09, "CADJPY":0.15, "EURCAD":0.13,
    "GBPCAD":0.20, "NZDCAD":0.10, "GBPAUD":0.13, "GBPNZD":0.19, "GBPCHF":0.17, "AUDCAD":0.10, "AUDCHF":0.09, "AUDNZD":0.08, "EURAUD":0.13,
    "USOIL":0.16, "UKOIL":0.65, "XAUUSD":0.11, "XAGUSD":1.05,
    "BTCUSD":0.16, "BTCJPY":0.25, "BTCXAU":0.20, "ETHUSD":0.30,"AAPL":0.05, "MSFT":0.17,
    "AAPL": 0.05, "MSFT":0.17, "GOOGL": 0.11, "AMZN": 0.05, "NVDA": 0.08, "META": 0.33, "TSLA": 0.07,
    "AMD":0.14, "NFLX":0.98 , "US500":0.03 ,
    "USTEC":0.03,
}

# --- News Filter Times (User Input) ---
NEWS_TIMES_UTC = {
"EURUSD":[ ], "USDCHF":[],   "GBPJPY":[], "GBPUSD":[ ],
                           "AUDJPY":[],   "EURNZD":[], "NZDUSD":[], "AUDUSD":[ ], "USDCAD":[ ], "USDJPY":[ ], "EURJPY":[],"EURCHF":[], "CADCHF":[], "CADJPY":[], "EURCAD":[],
                           "GBPCAD":[], "NZDCAD":[], "GBPAUD":[], "GBPNZD":[], "GBPCHF":[], "AUDCAD":[], "AUDCHF":[], "AUDNZD":[], "EURAUD":[],
                       "USOIL":[], "UKOIL":[], "XAUUSD":[], "XAGUSD":[],
                       "BTCUSD":[], "BTCJPY":[], "BTCXAU":[], "ETHUSD":[],"AAPL":[], "MSFT":[], "GOOGL":[], "AMZN":[], "NVDA":[], "META":[], "TSLA":[], "AMD":[], "NFLX":[], "US500":[],
                       "USTEC":[], "INTC":[], "MO":[], "BABA":[], "ABT":[], "LI":[], "TME":[], "ADBE":[], "MMM":[], "WMT":[], "PFE":[], "EQIX":[], "F":[], "ORCL":[], "BA":[], "NKE":[], "C":[],

}

# --- CSV File Recording Configuration ---
TRADE_HISTORY_FILE = "bookStrategy_trade_history.csv"
CSV_HEADERS = ["TicketID", "PositionID", "Symbol", "Type", "OpenTimeUTC", "EntryPrice",
"LotSize", "SL_Price", "TP_Price", "CloseTimeUTC", "ExitPrice",
"PNL_AccountCCY", "OpenComment", "CloseReason", "RiskedAmount"]

# --- CSV Helper Functions (unchanged from your version) ---

def initialize_trade_history_file():
    if not os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
            logger.info(f"{TRADE_HISTORY_FILE} created with headers.")
        except IOError as e:
            logger.error(f"Error creating CSV file {TRADE_HISTORY_FILE}: {e}")

def load_state_from_csv():
    global logged_open_position_ids, trade_details_for_closure
    logged_open_position_ids.clear()
    trade_details_for_closure.clear()
    trade_data_lines = []
    try:
        with open(TRADE_HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
            for line in f:
                if "--- Performance Summary ---" in line:
                    break
                trade_data_lines.append(line)
    except FileNotFoundError:
        logger.info(f"{TRADE_HISTORY_FILE} not found. Starting with empty state.")
        return
    except Exception as e:
        logger.error(f"Error reading {TRADE_HISTORY_FILE} for state loading: {e}")
        return
    if len(trade_data_lines) <= 1:
        logger.info("No valid trade data in history file to load state from.")
        return
    csv_content_buffer = io.StringIO("".join(trade_data_lines))
    try:
        df = pd.read_csv(csv_content_buffer, dtype={'PositionID': str})
        df = df[df['PositionID'].str.isdigit().fillna(False)]
        open_trades_df = df[df['CloseTimeUTC'].isnull() | (df['CloseTimeUTC'] == '')]
        for _, row in open_trades_df.iterrows():
            pos_id = str(row['PositionID'])
            logged_open_position_ids.add(pos_id)
            trade_details_for_closure[pos_id] = {
                'symbol': row['Symbol'],
                'original_sl': float(row['SL_Price']),
                'current_sl': float(row['SL_Price']),
                 # --- NEW STATE: Initialize for loaded trades ---
                'trailing_active': False, # Assume offensive TSL is not active on load
                'defensive_tsl_active': False,
                'invalid_signal_streak': 0,
            }
        logger.info(f"Loaded {len(logged_open_position_ids)} open positions' IDs from {TRADE_HISTORY_FILE}")
    except pd.errors.EmptyDataError:
        logger.info(f"{TRADE_HISTORY_FILE} is empty. Starting with empty state.")
    except Exception as e:
        logger.error(f"Error loading state from CSV {TRADE_HISTORY_FILE}: {e}")

def append_trade_to_csv(trade_data_dict):
    try:
        trade_data_dict['PositionID'] = str(trade_data_dict['PositionID'])
        with open(TRADE_HISTORY_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(trade_data_dict)
        logger.info(f"Appended open trade (PosID: {trade_data_dict['PositionID']}) to {TRADE_HISTORY_FILE}.")
    except Exception as e:
        logger.error(f"Unexpected error appending to CSV {TRADE_HISTORY_FILE}: {e}")

def update_closed_trade_in_csv(position_id_to_update, update_values_dict):
    position_id_to_update_str = str(position_id_to_update)
    try:
        lines = []
        updated = False
        with open(TRADE_HISTORY_FILE, 'r', newline='') as f_read:
            reader = csv.reader(f_read)
            header = next(reader)
            lines.append(header)
            for row in reader:
                if row and len(row) == len(CSV_HEADERS):
                    if row[CSV_HEADERS.index('PositionID')] == position_id_to_update_str and \
                       (not row[CSV_HEADERS.index('CloseTimeUTC')] or row[CSV_HEADERS.index('CloseTimeUTC')] == ''):
                        for key, value in update_values_dict.items():
                            if key in CSV_HEADERS:
                                row[CSV_HEADERS.index(key)] = value
                        updated = True
                lines.append(row)
        if updated:
            with open(TRADE_HISTORY_FILE, 'w', newline='') as f_write:
                writer = csv.writer(f_write)
                writer.writerows(lines)
            logger.info(f"Updated closed trade (PosID: {position_id_to_update_str}) in {TRADE_HISTORY_FILE}.")
        else:
            logger.warning(f"Could not find open trade with PositionID {position_id_to_update_str} in {TRADE_HISTORY_FILE} to update.")
    except Exception as e:
        logger.error(f"Error updating CSV for position {position_id_to_update_str}: {e}")

def calculate_and_append_performance_summary(csv_filepath, session_initial_balance):
    logger.info(f"Calculating CUMULATIVE performance summary for all trades in {csv_filepath}.")
    if not os.path.exists(csv_filepath):
        logger.warning(f"Trade history file {csv_filepath} not found. Cannot calculate performance summary.")
        return
    trade_data_lines = []
    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as f_read:
            reader = csv.reader(f_read)
            for row in reader:
                if row and "--- Performance Summary ---" in row[0]: break
                trade_data_lines.append(row)
        while trade_data_lines and not any(field for field in trade_data_lines[-1]): trade_data_lines.pop()
    except Exception as e:
        logger.error(f"Error reading {csv_filepath} to remove old summary: {e}")
        return
    if len(trade_data_lines) <= 1:
        logger.info("No trade data in history file to summarize.")
        try:
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_write:
                writer = csv.writer(f_write)
                writer.writerows(trade_data_lines)
        except Exception as e:
            logger.error(f"Error cleaning up empty/summary-only file {csv_filepath}: {e}")
        return
    csv_content_buffer = io.StringIO()
    csv_writer = csv.writer(csv_content_buffer)
    csv_writer.writerows(trade_data_lines)
    csv_content_buffer.seek(0)
    try:
        df_all = pd.read_csv(csv_content_buffer, dtype={'PositionID': str, 'PNL_AccountCCY': str})
        df_trades_only = df_all[df_all['PositionID'].str.isdigit().fillna(False)].copy()
        if df_trades_only.empty:
            logger.info("No valid trades found in history file to summarize.")
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_write:
                writer = csv.writer(f_write)
                writer.writerows(trade_data_lines)
            return
        df_trades_only['PNL_AccountCCY'] = pd.to_numeric(df_trades_only['PNL_AccountCCY'], errors='coerce')
        df_closed = df_trades_only[df_trades_only['PNL_AccountCCY'].notna()].copy()
        if df_closed.empty:
            logger.info("No closed trades found to summarize.")
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_write:
                writer = csv.writer(f_write)
                writer.writerows(trade_data_lines)
            return
        df_closed['CloseTimeUTC_dt'] = pd.to_datetime(df_closed['CloseTimeUTC'], errors='coerce', utc=True)
        df_closed = df_closed.sort_values(by='CloseTimeUTC_dt').reset_index(drop=True)
        total_pnl = df_closed['PNL_AccountCCY'].sum()
        equity_curve = [session_initial_balance] + (session_initial_balance + df_closed['PNL_AccountCCY'].cumsum()).tolist()
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = equity_series - rolling_max
        max_drawdown_usd = abs(drawdown.min())
        mdd_end_idx = drawdown.idxmin()
        peak_at_mdd_start = rolling_max[mdd_end_idx]
        max_drawdown_percent = (max_drawdown_usd / peak_at_mdd_start) * 100 if peak_at_mdd_start > 0 else 0.0
        summary_data = [
            ["Total Closed Trades", len(df_closed)],
            ["Winning Trades", len(df_closed[df_closed['PNL_AccountCCY'] > 0])],
            ["Losing Trades", len(df_closed[df_closed['PNL_AccountCCY'] < 0])],
            ["Total PNL (Account CCY)", f"{total_pnl:.2f}"],
            ["Max Drawdown (USD)", f"{max_drawdown_usd:.2f}"],
            ["Max Drawdown (%)", f"{max_drawdown_percent:.2f}%"]
        ]
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_write:
            writer = csv.writer(f_write)
            writer.writerows(trade_data_lines)
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as f_append:
            writer = csv.writer(f_append)
            writer.writerow([])
            writer.writerow(["--- Performance Summary ---", f"Generated: {datetime.now(timezone.utc).isoformat()} ---"])
            writer.writerow(["Metric", "Value"])
            writer.writerows(summary_data)
        logger.info(f"CUMULATIVE performance summary updated in {csv_filepath}")
    except Exception as e:
        logger.error(f"Error calculating or writing cumulative performance summary: {e}", exc_info=True)


# --- MT5 Initialization and Shutdown (unchanged) ---

def initialize_mt5_interface(symbols_to_check):
    global SYMBOLS_AVAILABLE_FOR_TRADE, ALL_SYMBOL_PROPERTIES, session_start_balance
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    logger.info("MetaTrader 5 Initialized")
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to get account info, error code = {mt5.last_error()}")
        mt5.shutdown()
        return False
    else:
        logger.info(f"Account Login: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
        session_start_balance = account_info.balance
    successfully_initialized_symbols = []
    temp_symbol_properties = {}
    for symbol_name in symbols_to_check:
        symbol_info_obj = mt5.symbol_info(symbol_name)
        if symbol_info_obj is None: logger.warning(f"Symbol {symbol_name} not found in MT5. Skipping."); continue
        if not symbol_info_obj.visible:
            logger.info(f"Symbol {symbol_name} not visible, attempting to select.")
            if not mt5.symbol_select(symbol_name, True): logger.warning(f"symbol_select({symbol_name}) failed. Skipping."); continue
            time.sleep(0.5); symbol_info_obj = mt5.symbol_info(symbol_name)
            if symbol_info_obj is None or not symbol_info_obj.visible: logger.warning(f"Symbol {symbol_name} still not available. Skipping."); continue
        if symbol_info_obj.point == 0 or symbol_info_obj.trade_tick_size == 0:
            logger.warning(f"Symbol {symbol_name} has invalid point/tick_size. Skipping.")
            continue
        current_pip_value = 0.0001
        if 'JPY' in symbol_name.upper(): current_pip_value = 0.01
        elif any(sub in symbol_name.upper() for sub in ["XAU", "XAG", "XPT", "OIL", "BTC", "ETH"]): current_pip_value = 0.01
        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point, 'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
            'trade_tick_value': symbol_info_obj.trade_tick_value,
            'volume_min': symbol_info_obj.volume_min,
            'volume_step': symbol_info_obj.volume_step if symbol_info_obj.volume_step > 0 else 0.01,
            'volume_max': symbol_info_obj.volume_max,
            'pip_value_calc': current_pip_value
        }
        successfully_initialized_symbols.append(symbol_name)
    if not successfully_initialized_symbols:
        logger.error("No symbols were successfully initialized from the target list.")
        return False
    SYMBOLS_AVAILABLE_FOR_TRADE = successfully_initialized_symbols
    ALL_SYMBOL_PROPERTIES = temp_symbol_properties
    logger.info(f"Properties recorded for: {SYMBOLS_AVAILABLE_FOR_TRADE}")
    return True

def shutdown_mt5_interface():
    mt5.shutdown()
    logger.info("MetaTrader 5 Shutdown")

# --- Live Bot Helper Functions (unchanged) ---

def is_weekend_utc():
    now_utc = datetime.now(timezone.utc)
    return now_utc.weekday() >= 5 # 5=Saturday, 6=Sunday

def get_reference_clock_symbol():
    if is_weekend_utc():
        if "BTCUSD" in SYMBOLS_AVAILABLE_FOR_TRADE: return "BTCUSD"
        if "ETHUSD" in SYMBOLS_AVAILABLE_FOR_TRADE: return "ETHUSD"
        return None
    else:
        if "EURUSD" in SYMBOLS_AVAILABLE_FOR_TRADE: return "EURUSD"
        if "BTCUSD" in SYMBOLS_AVAILABLE_FOR_TRADE: return "BTCUSD"
        if "ETHUSD" in SYMBOLS_AVAILABLE_FOR_TRADE: return "ETHUSD"
        if SYMBOLS_AVAILABLE_FOR_TRADE: return SYMBOLS_AVAILABLE_FOR_TRADE[0]
        return None

def is_within_session(symbol_sessions):
    if not symbol_sessions: return True
    candle_hour = datetime.now(timezone.utc).hour
    for start_hour, end_hour in symbol_sessions:
        if start_hour <= candle_hour < end_hour: return True
    return False

def is_outside_news_blackout(symbol: str, news_times_map: dict) -> bool:
    symbol_news_times = news_times_map.get(symbol)
    if not symbol_news_times: return True
    now_utc = datetime.now(timezone.utc)
    current_minutes_since_midnight = now_utc.hour * 60 + now_utc.minute
    for news_time_str in symbol_news_times:
        try:
            parts = news_time_str.split(':')
            news_hour, news_minute = int(parts[0]), int(parts[1])
            news_event_minutes_since_midnight = news_hour * 60 + news_minute
            blackout_start = news_event_minutes_since_midnight - 30
            blackout_end = news_event_minutes_since_midnight + 5
            if blackout_start <= current_minutes_since_midnight <= blackout_end:
                logger.warning(f"NEWS BLACKOUT: Current time {now_utc.strftime('%H:%M')} is in news window for {symbol} (Event at {news_time_str} UTC)")
                return False
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing news time '{news_time_str}' for symbol {symbol}: {e}.")
            continue
    return True

# --- NEW: Helper functions for Dynamic Take Profit (unchanged) ---
def get_swing_points(df, order=5):
    """
    Finds swing high and low points in a dataframe.
    'order' determines how many points on each side must be lower/higher.
    """
    high_indices = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
    low_indices = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    return df['high'].iloc[high_indices], df['low'].iloc[low_indices]

def get_dynamic_tp(entry_price, sl_price, trade_type, swing_levels):
    """
    Calculates the first valid TP that is at least 2.0R away.
    """
    initial_risk_dollars = abs(entry_price - sl_price)
    if initial_risk_dollars == 0:
        return None, None

    if trade_type == 'BUY':
        potential_targets = sorted([level for level in swing_levels if level > entry_price])
    else: # SELL
        potential_targets = sorted([level for level in swing_levels if level < entry_price], reverse=True)

    for target_price in potential_targets:
        reward_dollars = abs(target_price - entry_price)
        r_multiple = reward_dollars / initial_risk_dollars
        
        if r_multiple >= 2.0:
            return target_price, r_multiple

    return None, None
# --- End of New Helpers ---




def get_live_data(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        logger.warning(f"No live data for {symbol} on {timeframe}. Err: {mt5.last_error()}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def get_latest_m5_candle_time(reference_symbol):
    try:
        rates = mt5.copy_rates_from_pos(reference_symbol, mt5.TIMEFRAME_M5, 0, 1)
        if rates is None or len(rates) == 0:
            logger.warning(f"Could not fetch latest candle time for clock '{reference_symbol}'. Last error: {mt5.last_error()}")
            return None
        return rates[0]['time']
    except Exception as e:
        logger.error(f"Error in get_latest_m5_candle_time for '{reference_symbol}': {e}")
        return None

# --- MODIFIED: fetch_latest_data now returns the full M5 df as well ---
def fetch_latest_data(symbol):
    df_h4 = get_live_data(symbol, mt5.TIMEFRAME_H4, 100)
    df_h1 = get_live_data(symbol, mt5.TIMEFRAME_H1, 100)
    df_m5 = get_live_data(symbol, mt5.TIMEFRAME_M5, 100)
    if df_h4.empty or df_h1.empty or df_m5.empty or len(df_m5) < 2:
        logger.warning(f"Could not fetch complete data for {symbol} or not enough M5 candles. Skipping analysis.")
        return None, None, None, None

    # H4 Indicators
    df_h4['H4_EMA8'] = ta.ema(df_h4['close'], length=8)
    df_h4['H4_EMA21'] = ta.ema(df_h4['close'], length=21)
    df_h4['RSI_H4'] = ta.rsi(df_h4['close'], length=14)

    # H1 Indicators
    df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
    df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
    df_h1['RSI_H1'] = ta.rsi(df_h1['close'], length=14)
    # Note: H1_ADX is calculated here but not used by the backtest logic, so we leave it but it has no effect.
    adx_h1 = ta.adx(df_h1['high'], df_h1['low'], df_h1['close'], length=14)
    if adx_h1 is not None and not adx_h1.empty:
        df_h1['H1_ADX'] = adx_h1['ADX_14']
    else:
        df_h1['H1_ADX'] = np.nan
    df_h1.rename(columns={'close': 'H1_Close_For_Bias'}, inplace=True)

    # M5 Indicators
    df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
    df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
    df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
    df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
    df_m5['RSI_M5'] = ta.rsi(df_m5['close'], length=14)
    if 'ATR' in df_m5.columns and len(df_m5) >= 34:
        df_m5['ATR_SMA20'] = ta.sma(df_m5['ATR'], length=20)
    else:
        df_m5['ATR_SMA20'] = np.nan
        
    # ✅ FIX 3: Standardize ADX column name to 'ADX_14' to match backtest
    adx_m5 = ta.adx(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
    if adx_m5 is not None and not adx_m5.empty:
        df_m5['ADX_14'] = adx_m5['ADX_14'] # Changed from M5_ADX
    else:
        df_m5['ADX_14'] = np.nan

    if 'tick_volume' in df_m5.columns and len(df_m5) >= 20:
        df_m5['volume_MA20'] = ta.sma(df_m5['tick_volume'], length=20)
    else:
        df_m5['volume_MA20'] = np.nan

    # Combine dataframes
    combined_df = pd.merge_asof(df_m5.sort_index(), df_h1[['H1_Close_For_Bias', 'H1_EMA8', 'H1_EMA21', 'RSI_H1', 'H1_ADX']].sort_index(),
                                left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta(hours=1))
    combined_df = pd.merge_asof(combined_df.sort_index(), df_h4[['H4_EMA8', 'H4_EMA21', 'RSI_H4']].sort_index(),
                                left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta(hours=4))
    combined_df.dropna(inplace=True)

    if combined_df.empty or len(combined_df) < 2:
        logger.warning(f"Not enough data for {symbol} after combining and cleaning. Skipping analysis.")
        return None, None, None, None

    return combined_df, combined_df.iloc[-2], df_m5, df_h1

def calculate_pullback_depth(impulse_start, impulse_end, current_price, trade_type):
    total_leg = abs(impulse_end - impulse_start)
    if total_leg == 0: return 0
    pullback = (impulse_end - current_price) if trade_type == "BUY" else (current_price - impulse_end)
    return max(0.0, pullback / total_leg)

def calculate_fib_levels(swing_high, swing_low):
    return {"0.382": swing_low + 0.382 * (swing_high - swing_low),"0.5": swing_low + 0.5 * (swing_high - swing_low),"0.618": swing_low + 0.618 * (swing_high - swing_low)}

def place_pending_order(symbol, props, order_type, entry_price, sl_price, lot_size, comment):
    trade_type = mt5.ORDER_TYPE_BUY_STOP if order_type == "BUY_STOP" else mt5.ORDER_TYPE_SELL_STOP
    # Place pending order with NO TP. TP will be set once the position is active.
    request = {"action": mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": lot_size, "type": trade_type, "price": entry_price, "sl": sl_price, "tp": 0.0, "magic": 202405, "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,}
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"[{symbol}] Order Send FAILED. Retcode: {result.retcode if result else 'N/A'}, Comment: {result.comment if result else 'N/A'}")
        return None
    logger.info(f"[{symbol}] PENDING ORDER PLACED. Ticket: {result.order}, Type: {order_type}, Price: {entry_price}, Lot: {lot_size}")
    return result.order

def modify_position_sltp(position, new_sl, new_tp):
    request = {"action": mt5.TRADE_ACTION_SLTP, "position": position.ticket, "sl": new_sl, "tp": new_tp}
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[{position.symbol}] Position {position.ticket} MODIFIED. New SL: {new_sl}, New TP: {new_tp}")
        return True
    else:
        logger.error(f"[{position.symbol}] Position {position.ticket} MODIFY FAILED. Retcode: {result.retcode if result else 'N/A'}, Error: {mt5.last_error()}")
        return False

def cancel_pending_order(order_ticket):
    request = {"action": mt5.TRADE_ACTION_REMOVE, "order": order_ticket}
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Pending order {order_ticket} CANCELLED successfully.")
        return True
    else:
        logger.error(f"Pending order {order_ticket} CANCELLATION FAILED. Retcode: {result.retcode if result else 'N/A'}")
        return False

def manage_closed_positions():
    live_position_ids = {str(p.ticket) for p in mt5.positions_get() if p.magic == 202405}
    closed_position_ids = logged_open_position_ids - live_position_ids
    for pos_id in closed_position_ids:
        logger.info(f"Position {pos_id} detected as closed. Fetching history...")
        time.sleep(2)
        deals = mt5.history_deals_get(position=int(pos_id))
        if not deals:
            logger.warning(f"Could not find deals for closed position {pos_id}. Will retry on next cycle.")
            continue
        deals_df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        entry_deals_df = deals_df[deals_df['entry'] == mt5.DEAL_ENTRY_IN]; exit_deals_df = deals_df[deals_df['entry'] == mt5.DEAL_ENTRY_OUT]
        if exit_deals_df.empty:
            logger.warning(f"Position {pos_id} is closed, but no 'OUT' deal found in history yet. Retrying next cycle.")
            continue
        if entry_deals_df.empty:
            logger.error(f"CRITICAL: Position {pos_id} has an exit deal but no entry deal. Skipping update.")
            if pos_id in logged_open_position_ids: logged_open_position_ids.remove(pos_id)
            if pos_id in trade_details_for_closure: del trade_details_for_closure[pos_id]
            continue
        entry_deal = entry_deals_df.iloc[0]; exit_deal = exit_deals_df.iloc[-1]
        commission_cost = COMMISSIONS.get(exit_deal.symbol, 0.0)
        net_pnl = exit_deal.profit + exit_deal.commission + exit_deal.swap - commission_cost
        update_values = {'CloseTimeUTC': pd.to_datetime(exit_deal.time, unit='s', utc=True).isoformat(), 'ExitPrice': exit_deal.price, 'PNL_AccountCCY': f"{net_pnl:.2f}", 'CloseReason': f"Closed by broker: {exit_deal.comment}"}
        update_closed_trade_in_csv(pos_id, update_values)
        logged_open_position_ids.remove(pos_id)
        if pos_id in trade_details_for_closure: del trade_details_for_closure[pos_id]

# +++ NEW HELPER FUNCTION FOR DEFENSIVE TSL +++
def is_trade_setup_still_valid(symbol, trade_type, last_closed_candle, full_m5_df):
    if last_closed_candle is None or full_m5_df is None: return False

    # ✅ FIX 1: H1 Trend Bias Check (Stricter, matches backtest)
    h1_ema8 = last_closed_candle['H1_EMA8']; h1_ema21 = last_closed_candle['H1_EMA21']
    h1_close = last_closed_candle['H1_Close_For_Bias']
    current_h1_bias = None
    if h1_ema8 > h1_ema21 and h1_close > h1_ema8 and h1_close > h1_ema21:
        current_h1_bias = "BUY"
    elif h1_ema8 < h1_ema21 and h1_close < h1_ema8 and h1_close < h1_ema21:
        current_h1_bias = "SELL"
        
    if trade_type != current_h1_bias:
        logger.debug(f"[{symbol}] Defensive Check FAIL: H1 bias ({current_h1_bias}) no longer matches trade type ({trade_type}).")
        return False

    # M5 Fanned EMAs Check (This was already correct)
    m5_fanned_buy = last_closed_candle['M5_EMA8'] > last_closed_candle['M5_EMA13'] or last_closed_candle['M5_EMA8'] > last_closed_candle['M5_EMA21']
    m5_fanned_sell = last_closed_candle['M5_EMA8'] < last_closed_candle['M5_EMA13'] or last_closed_candle['M5_EMA8'] < last_closed_candle['M5_EMA21']
    if not ((trade_type == "BUY" and m5_fanned_buy) or (trade_type == "SELL" and m5_fanned_sell)):
        logger.debug(f"[{symbol}] Defensive Check FAIL: M5 EMAs are no longer fanned for {trade_type}.")
        return False
    
    # ADX Check (non-crypto)
    is_crypto = symbol in CRYPTO_SYMBOLS
    if not is_crypto:
        # ✅ FIX 3: Use harmonized column name 'ADX_14'
        adx_value = last_closed_candle.get('ADX_14', 0) 
        if pd.isna(adx_value) or adx_value < 20:
            logger.debug(f"[{symbol}] Defensive Check FAIL: M5 ADX ({adx_value:.2f}) fell below 20.")
            return False

    # ✅ FIX 2: RSI Check (Corrected logic, matches backtest's valid function)
    rsi_m5 = last_closed_candle.get('RSI_M5', 50)
    rsi_h1 = last_closed_candle.get('RSI_H1', 50)
    if trade_type == "BUY":
        if not (rsi_m5 > 50 and rsi_h1 > 50):
            logger.debug(f"[{symbol}] Defensive Check FAIL: RSI conditions no longer met for BUY.")
            return False
    elif trade_type == "SELL":
        if not (rsi_m5 < 50 and rsi_h1 < 50):
            logger.debug(f"[{symbol}] Defensive Check FAIL: RSI conditions no longer met for SELL.")
            return False

    # Price vs M5 EMA21 (This was already correct)
    if (trade_type == "BUY" and last_closed_candle['close'] < last_closed_candle['M5_EMA21']) or \
       (trade_type == "SELL" and last_closed_candle['close'] > last_closed_candle['M5_EMA21']):
        logger.debug(f"[{symbol}] Defensive Check FAIL: Price crossed the M5_EMA21 against the trend.")
        return False

    return True

# +++ FULLY REVISED manage_open_positions FUNCTION +++

# +++ FULLY REVISED manage_open_positions FUNCTION +++

def manage_open_positions():
    open_positions = mt5.positions_get(magic=202405)
    if not open_positions: return

    for position in open_positions:
        pos_id_str = str(position.ticket)

        # --- BLOCK 1: NEW POSITION INITIALIZATION (No changes here, remains the same) ---
        if pos_id_str not in logged_open_position_ids:
            props = ALL_SYMBOL_PROPERTIES[position.symbol]
            tp_price = 0.0

            pending_details = trade_details_for_closure.get(pos_id_str)
            if pending_details and pending_details.get("is_pending"):
                tp_price = pending_details['tp_price']
                logger.info(f"[{position.symbol}] New position {pos_id_str} detected. Applying dynamic TP: {tp_price}")
            else:
                logger.warning(f"[{position.symbol}] New position {pos_id_str} detected, but no pending TP details found. Applying fallback 4R TP.")
                risk_val_diff = abs(position.price_open - position.sl)
                tp_price = round(position.price_open + (4 * risk_val_diff) if position.type == mt5.ORDER_TYPE_BUY else position.price_open - (4 * risk_val_diff), props['digits'])

            modify_position_sltp(position, position.sl, tp_price)

            risk_amount = (abs(position.price_open - position.sl) / props['trade_tick_size']) * props['trade_tick_value'] * position.volume if props['trade_tick_size'] > 0 else 0
            trade_data = {
                "TicketID": position.ticket, "PositionID": position.ticket, "Symbol": position.symbol,
                "Type": "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL",
                "OpenTimeUTC": pd.to_datetime(position.time, unit='s', utc=True).isoformat(),
                "EntryPrice": position.price_open, "LotSize": position.volume,
                "SL_Price": position.sl, "TP_Price": tp_price, "CloseTimeUTC": "", "ExitPrice": "",
                "PNL_AccountCCY": "", "OpenComment": position.comment, "CloseReason": "",
                "RiskedAmount": f"{risk_amount:.2f}"
            }
            append_trade_to_csv(trade_data)
            logged_open_position_ids.add(pos_id_str)
            
            trade_details_for_closure[pos_id_str] = {
                'symbol': position.symbol,
                'original_sl': position.sl,
                'current_sl': position.sl,
                'trailing_active': False,
                'defensive_tsl_active': False,
                'invalid_signal_streak': 0,
            }
            continue

        # --- BLOCK 2: ONGOING POSITION MANAGEMENT ---
        details = trade_details_for_closure.get(pos_id_str)
        if not details: continue
        
        _, last_closed_candle, full_m5_df, _ = fetch_latest_data(position.symbol)
        if last_closed_candle is None:
            logger.warning(f"Could not get data for TSL on position {position.ticket}. Skipping.")
            continue
        
        # If the Defensive brain is active, it takes priority. Skip the Offensive brain.
        if details['defensive_tsl_active']:
            # --- "Defensive Brain" logic is handled below ---
            pass
        else:
            # --- "Offensive Brain": Volatility-Adjusted TSL (SYNCHRONIZED WITH BACKTESTER) ---
            current_atr = last_closed_candle.get('ATR')
            average_atr = last_closed_candle.get('ATR_SMA20')

            # Guard clause to ensure we have valid data for the adaptive logic
            if pd.notna(current_atr) and pd.notna(average_atr) and average_atr > 0 and current_atr > 0:
                
                # --- START OF SYNCHRONIZED LOGIC (from backtester) ---
                volatility_ratio = current_atr / average_atr
                
                # Set TSL parameters based on the volatility regime
                if volatility_ratio >= 2.0:         # Very High Volatility
                    TRAIL_ACTIVATION_ATR, TRAIL_DISTANCE_ATR = 2.0, 3.0
                elif volatility_ratio >= 1.25:      # High Volatility
                    TRAIL_ACTIVATION_ATR, TRAIL_DISTANCE_ATR = 1.5, 2.5
                elif volatility_ratio >= 0.75:      # Normal Volatility
                    TRAIL_ACTIVATION_ATR, TRAIL_DISTANCE_ATR = 1.0, 2.0
                else:                               # Low Volatility
                    TRAIL_ACTIVATION_ATR, TRAIL_DISTANCE_ATR = 0.5, 1.5
                # --- END OF SYNCHRONIZED LOGIC ---

                initial_risk_price_diff = abs(position.price_open - details['original_sl'])
                move_from_entry_price = (last_closed_candle['high'] - position.price_open) if position.type == mt5.ORDER_TYPE_BUY else (position.price_open - last_closed_candle['low'])
                r_multiple_achieved = move_from_entry_price / initial_risk_price_diff if initial_risk_price_diff > 0 else 0

                if not details['trailing_active'] and r_multiple_achieved >= TRAIL_ACTIVATION_ATR:
                    details['trailing_active'] = True
                    logger.info(f"[{position.symbol}] Vol Ratio: {volatility_ratio:.2f}. OFFENSIVE TSL ACTIVATED for pos {position.ticket} at {r_multiple_achieved:.2f}R.")

                if details['trailing_active']:
                    new_sl_price = 0
                    if position.type == mt5.ORDER_TYPE_BUY:
                        potential_new_sl = last_closed_candle['high'] - (TRAIL_DISTANCE_ATR * current_atr)
                        if potential_new_sl > details['current_sl']:
                            new_sl_price = potential_new_sl
                    else: # SELL trade
                        potential_new_sl = last_closed_candle['low'] + (TRAIL_DISTANCE_ATR * current_atr)
                        if potential_new_sl < details['current_sl']:
                            new_sl_price = potential_new_sl

                    if new_sl_price > 0:
                        props = ALL_SYMBOL_PROPERTIES[position.symbol]
                        rounded_new_sl = round(new_sl_price, props['digits'])
                        if rounded_new_sl != details['current_sl']:
                            logger.info(f"[{position.symbol}] Offensive TSL Update: Moving SL for {position.ticket} to {rounded_new_sl}")
                            if modify_position_sltp(position, rounded_new_sl, position.tp):
                                details['current_sl'] = rounded_new_sl
                    
                    continue # If offensive TSL is active and ran, skip the defensive check for this cycle

        # --- "Defensive Brain": Signal-Degradation SL Tightening (No changes here, remains the same) ---
        trade_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
        is_still_valid = is_trade_setup_still_valid(position.symbol, trade_type, last_closed_candle, full_m5_df)

        if is_still_valid:
            if details['invalid_signal_streak'] > 0:
                logger.info(f"[{position.symbol}] Signal for position {position.ticket} has become VALID again. Resetting defensive measures.")
                details['invalid_signal_streak'] = 0
                details['defensive_tsl_active'] = False
        else:
            details['invalid_signal_streak'] += 1
            logger.warning(f"[{position.symbol}] Signal for position {position.ticket} is INVALID. Streak: {details['invalid_signal_streak']}/4.")

            if not details['defensive_tsl_active'] and details['invalid_signal_streak'] >= 4:
                details['defensive_tsl_active'] = True
                logger.warning(f"[{position.symbol}] DEFENSIVE TSL ACTIVATED for position {position.ticket} after 4 consecutive invalid signals.")

            if details['defensive_tsl_active']:
                current_atr = last_closed_candle.get('ATR')
                average_atr = last_closed_candle.get('ATR_SMA20')
                tighten_percentage = 0.01 

                if pd.notna(current_atr) and pd.notna(average_atr) and average_atr > 0:
                    if current_atr > (average_atr * 1.5):
                        tighten_percentage = 0.003
                    elif current_atr > average_atr:
                        tighten_percentage = 0.005
                    else:
                        tighten_percentage = 0.01
                
                initial_risk_dist = abs(position.price_open - details['original_sl'])
                tighten_amount = initial_risk_dist * tighten_percentage

                new_sl_price = 0
                if position.type == mt5.ORDER_TYPE_BUY:
                    potential_new_sl = details['current_sl'] + tighten_amount
                    if potential_new_sl > details['current_sl'] and potential_new_sl < position.price_current:
                        new_sl_price = potential_new_sl
                else: # SELL
                    potential_new_sl = details['current_sl'] - tighten_amount
                    if potential_new_sl < details['current_sl'] and potential_new_sl > position.price_current:
                        new_sl_price = potential_new_sl
                
                if new_sl_price > 0:
                    props = ALL_SYMBOL_PROPERTIES[position.symbol]
                    rounded_new_sl = round(new_sl_price, props['digits'])
                    if rounded_new_sl != details['current_sl']:
                        logger.warning(f"[{position.symbol}] Defensive TSL Update: Adaptively tightening SL for {position.ticket} to {rounded_new_sl} (Step: {tighten_percentage*100:.1f}%)")
                        if modify_position_sltp(position, rounded_new_sl, position.tp):
                            details['current_sl'] = rounded_new_sl

def manage_pending_orders():
    pending_orders = mt5.orders_get(magic=202405)
    if not pending_orders: return
    for order in pending_orders:
        _, last_closed_candle, _, _ = fetch_latest_data(order.symbol)
        if last_closed_candle is None:
            logger.warning(f"Could not get valid closed candle data for managing pending order {order.ticket}. Skipping this cycle.")
            continue
        setup_bias = "BUY" if order.type == mt5.ORDER_TYPE_BUY_STOP else "SELL"; m5_ema21 = last_closed_candle['M5_EMA21']
        if (setup_bias == "BUY" and last_closed_candle['close'] < m5_ema21) or (setup_bias == "SELL" and last_closed_candle['close'] > m5_ema21):
            logger.info(f"[{order.symbol}] PENDING order {order.ticket} invalidated (Close vs M5_EMA21). Cancelling...")
            cancel_pending_order(order.ticket)

# --- MODIFIED: check_for_new_signals to call the updated fetch_latest_data ---
# --- IN bookStrategyLivetp.py, REPLACE THE ENTIRE check_for_new_signals FUNCTION ---

def check_for_new_signals(daily_risk_allocated, max_daily_risk):
    global delayed_setups_queue
    new_queue, order_placed_this_cycle = [], False
    
    # Process the delayed queue (this logic is fine)
    for setup in delayed_setups_queue:
        setup['confirm_count'] += 1
        if setup['confirm_count'] < 2: 
            new_queue.append(setup)
            continue
        _, _, _, _ = fetch_latest_data(setup['symbol'])
        if _ is None:
            new_queue.append(setup)
            continue
        if daily_risk_allocated + setup["risk_amt"] > max_daily_risk:
            logger.warning(f"[{setup['symbol']}] Delayed setup confirmed, but would exceed daily risk limit. Discarding.")
            continue
        props = ALL_SYMBOL_PROPERTIES[setup['symbol']]
        order_ticket = place_pending_order(setup['symbol'], props, f"{setup['bias']}_STOP", setup['entry_price'], setup['sl_price'], setup['lot_size'], "LiveBot_v1_Delayed")
        if order_ticket:
            trade_details_for_closure[str(order_ticket)] = {"is_pending": True, "tp_price": setup['tp_price']}
            daily_risk_allocated += setup["risk_amt"]
            order_placed_this_cycle = True
            break
        else: 
            new_queue.append(setup)
            
    delayed_setups_queue = new_queue
    if order_placed_this_cycle: 
        return daily_risk_allocated

    # Scan for new signals
    for symbol in SYMBOLS_AVAILABLE_FOR_TRADE:
        if is_weekend_utc() and symbol not in CRYPTO_SYMBOLS: continue
        # Make sure you have synchronized the TRADING_SESSIONS_UTC dictionary!
        if not is_within_session(TRADING_SESSIONS_UTC.get(symbol, [])): continue
        if not is_outside_news_blackout(symbol, NEWS_TIMES_UTC): continue

        df, last_closed_candle, _, df_h1 = fetch_latest_data(symbol)
        if df is None or last_closed_candle is None or df_h1 is None: continue
        
        props = ALL_SYMBOL_PROPERTIES[symbol]
        
        # --- STRATEGY CONDITIONS ---

        if symbol in STOCK_SYMBOLS:
            current_volume = last_closed_candle.get('tick_volume', 0); avg_volume = last_closed_candle.get('volume_MA20', 0)
            if pd.notna(current_volume) and pd.notna(avg_volume) and avg_volume > 0:
                if current_volume < (1.5 * avg_volume): continue
        
        # 1. H1 Trend Filter (This is now correct)
        h1_ema8 = last_closed_candle['H1_EMA8']; h1_ema21 = last_closed_candle['H1_EMA21']
        h1_close = last_closed_candle['H1_Close_For_Bias']
        h1_trend_bias = None
        if h1_ema8 > h1_ema21 and h1_close > h1_ema8 and h1_close > h1_ema21:
            h1_trend_bias = "BUY"
        elif h1_ema8 < h1_ema21 and h1_close < h1_ema8 and h1_close < h1_ema21:
            h1_trend_bias = "SELL"
        if not h1_trend_bias: continue

        # 2. M5 EMA Filter (This was always correct)
        m5_fanned_buy = last_closed_candle['M5_EMA8'] > last_closed_candle['M5_EMA13'] or last_closed_candle['M5_EMA8'] > last_closed_candle['M5_EMA21']
        m5_fanned_sell = last_closed_candle['M5_EMA8'] < last_closed_candle['M5_EMA13'] or last_closed_candle['M5_EMA8'] < last_closed_candle['M5_EMA21']
        if not ((h1_trend_bias == "BUY" and m5_fanned_buy) or (h1_trend_bias == "SELL" and m5_fanned_sell)): continue
        
        # 3. ADX Filter (This is now correct)
        is_crypto = symbol in CRYPTO_SYMBOLS
        if not is_crypto:
            adx_value = last_closed_candle.get('ADX_14', 0)
            if pd.isna(adx_value) or adx_value < 20: continue

        # ✅ 4. CORRECTED RSI Filter Logic
        rsi_m5 = last_closed_candle.get('RSI_M5', 50)
        rsi_h1 = last_closed_candle.get('RSI_H1', 50)
        if h1_trend_bias == "BUY":
            if not (rsi_m5 > 50 and rsi_h1 > 50):
                continue
        elif h1_trend_bias == "SELL":
            if not (rsi_m5 < 50 and rsi_h1 < 50):
                continue

        # 5. Price vs M5 EMA21 Filter (This was always correct)
        if (h1_trend_bias == "BUY" and last_closed_candle['close'] < last_closed_candle['M5_EMA21']) or \
           (h1_trend_bias == "SELL" and last_closed_candle['close'] > last_closed_candle['M5_EMA21']): continue
        
        # 6. Pullback Filter (This was always correct)
        pullback_found = (h1_trend_bias == "BUY" and last_closed_candle['low'] <= last_closed_candle['M5_EMA8']) or \
                         (h1_trend_bias == "SELL" and last_closed_candle['high'] >= last_closed_candle['M5_EMA8']) \
                         if is_crypto else \
                         (h1_trend_bias == "BUY" and last_closed_candle['close'] <= last_closed_candle['M5_EMA8']) or \
                         (h1_trend_bias == "SELL" and last_closed_candle['close'] >= last_closed_candle['M5_EMA8'])
        if not pullback_found: continue
        
        # --- The rest of the logic was already synchronized and is correct ---
        recent_candles = df.iloc[-6:-2]; bullish_count = (recent_candles['close'] > recent_candles['open']).sum(); bearish_count = (recent_candles['close'] < recent_candles['open']).sum()
        if (h1_trend_bias == "BUY" and bullish_count > 2) or (h1_trend_bias == "SELL" and bearish_count > 2): continue
        lookback_window = df.iloc[-12:-2]; swing_high, swing_low = lookback_window['high'].max(), lookback_window['low'].min()
        impulse_start, impulse_end, price_for_pb = (swing_low, swing_high, last_closed_candle['low']) if h1_trend_bias == "BUY" else (swing_high, swing_low, last_closed_candle['high'])
        if calculate_pullback_depth(impulse_start, impulse_end, price_for_pb, h1_trend_bias) < 0.30: continue
        fib_levels = calculate_fib_levels(swing_high, swing_low); tolerance = 0.5 * last_closed_candle['ATR']
        if not any(abs(last_closed_candle['M5_EMA8'] - fib_price) <= tolerance or abs(last_closed_candle['M5_EMA13'] - fib_price) <= tolerance for fib_price in fib_levels.values()): continue
        
        # --- Final Calculation Block (already synchronized) ---
               # --- PASTE THIS NEW CODE BLOCK IN ITS PLACE ---

        atr_val = last_closed_candle.get('ATR')
        if pd.isna(atr_val) or atr_val <= 0: 
            logger.debug(f"[{symbol}] Skipping due to invalid ATR value.")
            continue

        # --- SL Calculation (remains consistent across all entry strategies) ---
        sl_distance_atr = 4.0 * atr_val

        # --- Entry Price Calculation (based on the chosen strategy) ---
        entry_px, sl_px = (0, 0)

        # --- BUY LOGIC ---
        if h1_trend_bias == "BUY":
            if ENTRY_PRICE_STRATEGY == 'BREAKOUT':
                # Perfectly synchronized logic: breakout of the high of the 3 candles PRECEDING the signal candle.
                # Signal candle is at df.iloc[-2]. The 3 before it are at indices -5, -4, -3.
                entry_lookback = df.iloc[-4:-1] 
                pip_adj = 3 * props['trade_tick_size']
                entry_px = entry_lookback['high'].max() + pip_adj
            
            elif ENTRY_PRICE_STRATEGY == 'ATR_BUFFER':
                # Tighter entry: signal candle's high + a fraction of ATR
                entry_px = last_closed_candle['high'] + (0.5 * atr_val)

            elif ENTRY_PRICE_STRATEGY == 'FIXED_BUFFER':
                # Tighter entry: signal candle's high + a small fixed pip buffer
                fixed_buffer_pips = 3
                fixed_buffer_price = fixed_buffer_pips * props['pip_value_calc']
                entry_px = last_closed_candle['high'] + fixed_buffer_price

            elif ENTRY_PRICE_STRATEGY == 'NO_BUFFER':
                # Most aggressive: entry right at the signal candle's high
                entry_px = last_closed_candle['high']

            # Set the SL based on the calculated entry price
            sl_px = entry_px - sl_distance_atr

        # --- SELL LOGIC ---
        else: # h1_trend_bias == "SELL"
            if ENTRY_PRICE_STRATEGY == 'BREAKOUT':
                # Perfectly synchronized logic: breakout of the low of the 3 candles PRECEDING the signal candle.
                entry_lookback = df.iloc[-4:-1]
                pip_adj = 3 * props['trade_tick_size']
                entry_px = entry_lookback['low'].min() - pip_adj

            elif ENTRY_PRICE_STRATEGY == 'ATR_BUFFER':
                # Tighter entry: signal candle's low - a fraction of ATR
                entry_px = last_closed_candle['low'] - (0.5 * atr_val)

            elif ENTRY_PRICE_STRATEGY == 'FIXED_BUFFER':
                # Tighter entry: signal candle's low - a small fixed pip buffer
                fixed_buffer_pips = 3
                fixed_buffer_price = fixed_buffer_pips * props['pip_value_calc']
                entry_px = last_closed_candle['low'] - fixed_buffer_price

            elif ENTRY_PRICE_STRATEGY == 'NO_BUFFER':
                # Most aggressive: entry right at the signal candle's low
                entry_px = last_closed_candle['low']

            # Set the SL based on the calculated entry price
            sl_px = entry_px + sl_distance_atr

        # Round the final values to the correct number of digits for the symbol
        entry_px, sl_px = round(entry_px, props['digits']), round(sl_px, props['digits'])
        
        # Final check to ensure there's a valid distance between entry and SL
        if abs(entry_px - sl_px) <= 0: continue
        # --- END OF REPLACEMENT BLOCK ---
        swing_highs, swing_lows = get_swing_points(df_h1, order=5)
        targets = swing_highs if h1_trend_bias == 'BUY' else swing_lows
        tp_price, r_value = get_dynamic_tp(entry_px, sl_px, h1_trend_bias, targets)
        if tp_price is None:
            logger.debug(f"[{symbol}] Skipped: No valid market structure TP found with at least 2R potential.")
            continue
        logger.info(f"[{symbol}] Valid TP found at {tp_price:.{props['digits']}f} ({r_value:.2f}R potential).")
        tp_price = round(tp_price, props['digits'])
        lot_size = props['volume_min']
        est_risk = lot_size * (abs(entry_px - sl_px) / props['trade_tick_size']) * props['trade_tick_value'] if props['trade_tick_size'] > 0 else 0
        if est_risk > mt5.account_info().balance * RISK_PER_TRADE_PERCENT:
            logger.info(f"[{symbol}] Setup found but min lot risk ({est_risk:.2f}) exceeds max allowed. Skipping.")
            continue
        delayed_setups_queue.append({
            "symbol": symbol, "bias": h1_trend_bias, "entry_price": entry_px, 
            "sl_price": sl_px, "tp_price": tp_price, "lot_size": lot_size, 
            "risk_amt": est_risk, "confirm_count": 0
        })
        logger.info(f"[{symbol}] SETUP QUEUED with Dynamic TP. Bias: {h1_trend_bias}, Entry: {entry_px}, SL: {sl_px}, TP: {tp_price}")
        break 

    return daily_risk_allocated

# --- Main Execution (unchanged) ---

if __name__ == "__main__":
    if not initialize_mt5_interface(SYMBOLS_TO_TRADE):
        logger.error("Failed to initialize. Exiting.")
        exit()

    initialize_trade_history_file()
    load_state_from_csv()

    current_day = datetime.now(timezone.utc).date()
    daily_risk_allocated_today = 0.0
    max_daily_risk_budget = mt5.account_info().balance * DAILY_RISK_LIMIT_PERCENT
    consecutive_losses_count = 0

    last_processed_candle_time = None

    logger.info("--- Live Trading Bot Started ---")
    logger.info(f"Initial daily risk budget: {max_daily_risk_budget:.2f} USD")
    logger.info("Bot will now use a dynamic market clock (FX for weekdays, Crypto for weekends).")

    try:
        while True:
            clock_symbol = get_reference_clock_symbol()

            if not clock_symbol:
                logger.warning("Could not determine a valid market clock. Bot is paused. Checking again in 60s.")
                time.sleep(60)
                continue

            current_candle_time = get_latest_m5_candle_time(clock_symbol)

            if current_candle_time is not None and current_candle_time != last_processed_candle_time:
                logger.info(f"New M5 candle detected using clock '{clock_symbol}'. Time: {datetime.fromtimestamp(current_candle_time, tz=timezone.utc)}. Processing...")
                last_processed_candle_time = current_candle_time

                if datetime.now(timezone.utc).date() != current_day:
                    current_day = datetime.now(timezone.utc).date()
                    daily_risk_allocated_today, consecutive_losses_count = 0.0, 0
                    max_daily_risk_budget = mt5.account_info().balance * DAILY_RISK_LIMIT_PERCENT
                    logger.info(f"NEW DAY: {current_day}. Daily risk budget reset to {max_daily_risk_budget:.2f} USD.")

                manage_closed_positions()
                manage_open_positions()
                manage_pending_orders()

                open_positions = mt5.positions_get(magic=202405)
                pending_orders = mt5.orders_get(magic=202405)

                if not open_positions and not pending_orders:
                    if consecutive_losses_count < 5:
                        if daily_risk_allocated_today < max_daily_risk_budget:
                            logger.debug("No active/pending trades. Scanning for new setups...")
                            daily_risk_allocated_today = check_for_new_signals(daily_risk_allocated_today, max_daily_risk_budget)
                        else:
                            logger.info("Daily risk limit reached. No new trades will be sought today.")
                    else:
                        logger.warning("Consecutive loss limit hit. No new trades will be sought today.")
                else:
                    logger.info(f"Trade management cycle complete. Open: {len(open_positions)}, Pending: {len(pending_orders)}")

                logger.info("Processing complete. Waiting for next candle...")

            time.sleep(15)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting gracefully...")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("Performing final shutdown tasks.")
        calculate_and_append_performance_summary(TRADE_HISTORY_FILE, session_start_balance)
        shutdown_mt5_interface()