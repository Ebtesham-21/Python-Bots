import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta  # For EMAs and ATR
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import logging
import math
import os    # For file operations
import csv   # For CSV writing

# --- Logger Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables & State ---

SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}

# --- Bot State (populated at runtime) ---

logged_open_position_ids = set()
trade_details_for_closure = {}  # Holds details like original SL for management
delayed_setups_queue = []  # List of setups waiting for confirmation
session_start_balance = 0.0 # Will be set on initialization

# --- Strategy & Risk Parameters ---

SYMBOLS_TO_TRADE =  ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                    "AUDJPY", "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF",
                    "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                    "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD",
                    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500", "USTEC", "INTC", "MO", 
                    "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C",                                            
        
                             ]

TRADING_SESSIONS_UTC = { # (start_hour_inclusive, end_hour_exclusive)
    "EURUSD": [(7, 15)], "GBPUSD": [(7, 15)], "AUDUSD": [ (7, 15)],
    "USDCHF": [(7, 15)], "USDCAD": [(12, 15)], "USDJPY": [(0,4), (12, 15)],
    "EURJPY": [ (0,4) , (7, 12)], "GBPJPY": [ (7, 15)], "NZDUSD": [ (7, 15)],
    "EURCHF": [(7, 15)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 15)],
    "EURNZD": [ (7, 15)], "GBPNZD": [(7, 15)], "XAUUSD": [(7, 15)],
    "XAGUSD": [(7, 15)], "XPTUSD": [(7, 15)], "XAGGBP":[(7, 15)], "XAGEUR":[(7,15)], "XAGAUD": [(0,4), (7,10)], "BTCXAG":[(7,15)],
    "AAPL": [(14, 17)] , "MSFT": [(14, 17)], "GOOGL": [(14, 17)], "AMZN": [(14, 17)], "NVDA": [(14, 17)], "META": [(14, 17)], "TSLA": [(14, 17)], "AMD": [(14, 17)], "NFLX": [(14, 17)], "US500": [(14, 17)], 
                       "USTEC": [(14, 17)],"INTC":[(14, 17)], "MO":[(14, 17)], "BABA":[(14, 17)], "ABT":[(14, 17)], "LI":[(14, 17)], "TME":[(14, 17)], "ADBE":[(14, 17)], "MMM":[(14, 17)], "WMT":[(14, 17)], "PFE":[(14, 17)], "EQIX":[(14, 17)], "F":[(14, 17)], "ORCL":[(14, 17)], "BA":[(14, 17)], "NKE":[(14, 17)], "C":[(14, 17)],


}
TRADING_SESSIONS_UTC["USOIL"] = [(12, 15)]
TRADING_SESSIONS_UTC["UKOIL"] = [(7, 15)]
CRYPTO_SESSIONS_USER = {"BTCUSD":[(7, 15)], "BTCJPY":[(0, 15)], "BTCXAU":[(7, 15)], "ETHUSD":[(7, 15)]}
for crypto_sym, sess_val in CRYPTO_SESSIONS_USER.items():
    TRADING_SESSIONS_UTC[crypto_sym] = sess_val

# --- NEW: Define a list of crypto symbols for the weekend check ---
CRYPTO_SYMBOLS = list(CRYPTO_SESSIONS_USER.keys())

RISK_PER_TRADE_PERCENT = 0.01  # Risk 1% of current balance per trade
DAILY_RISK_LIMIT_PERCENT = 0.05 # Daily risk limit of 5% of balance at start of day

# --- Commission Structure ---

COMMISSIONS = {
    "EURUSD": 0.07, "AUDUSD": 0.10, "USDCHF": 0.10, "USDCAD": 0.10, "USDJPY":0.07, "EURJPY":0.11, "EURCHF":0.17,
    "NZDUSD": 0.13, "AUDJPY": 0.09, "EURNZD": 0.18, "USOIL": 0.16,
    "UKOIL": 0.65, "XAUUSD":0.11,"XAGUSD":1.05, "BTCUSD": 0.16, "BTCJPY": 0.21, "BTCXAU": 0.20,
    "ETHUSD": 0.30, "GBPUSD": 0.09, "USDJPY": 0.07, "GBPJPY": 0.15,
    "AAPL": 0.05, "MSFT":0.17, "GOOGL": 0.11, "AMZN": 0.05, "NVDA": 0.08, "META": 0.33, "TSLA": 0.07, 
    "AMD":0.14, "NFLX":0.98 , "US500":0.03 , 
                       "USTEC":0.03,
}

# --- News Filter Times (User Input) ---

NEWS_TIMES_UTC = {
"EURUSD":["12:30"], "USDCHF":["12:30"],   "GBPJPY":[], "GBPUSD":["12:30"], 
"AUDJPY":[],"EURNZD":[], "NZDUSD":["12:30"], "AUDUSD":["12:30"], "USDCAD":["12:30"],"USDJPY":["12:30"], "EURJPY":[],"EURCHF":[],
"USOIL":["12:30"], "UKOIL":[],"XAUUSD":["12:30"], "XAGUSD":["12:30"],
"BTCUSD":[], "BTCJPY":[], "BTCXAU":[], "ETHUSD":[],

}

# --- CSV File Recording Configuration ---

TRADE_HISTORY_FILE = "bookStrategy_trade_history.csv"
CSV_HEADERS = ["TicketID", "PositionID", "Symbol", "Type", "OpenTimeUTC", "EntryPrice",
"LotSize", "SL_Price", "TP_Price", "CloseTimeUTC", "ExitPrice",
"PNL_AccountCCY", "OpenComment", "CloseReason", "RiskedAmount"]

# --- CSV Helper Functions ---

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
    if not os.path.exists(TRADE_HISTORY_FILE):
        logger.info(f"{TRADE_HISTORY_FILE} not found. Starting with empty state.")
        return

    try:
        df = pd.read_csv(TRADE_HISTORY_FILE, dtype={'PositionID': str})
        df = df[df['PositionID'].str.isdigit().fillna(False)]
        open_trades_df = df[df['CloseTimeUTC'].isnull() | (df['CloseTimeUTC'] == '')]
        for _, row in open_trades_df.iterrows():
            pos_id = str(row['PositionID'])
            logged_open_position_ids.add(pos_id)
            trade_details_for_closure[pos_id] = {
                'symbol': row['Symbol'],
                'original_sl': float(row['SL_Price']),
                'current_sl': float(row['SL_Price']),
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
    logger.info(f"Calculating performance summary for trades in {csv_filepath} using session initial balance: {session_initial_balance:.2f}")
    if not os.path.exists(csv_filepath):
        logger.warning(f"Trade history file {csv_filepath} not found. Cannot calculate performance summary.")
        return
    try:
        df_all = pd.read_csv(csv_filepath, dtype={'PositionID': str, 'PNL_AccountCCY': str})
        df_trades_only = df_all[df_all['PositionID'].str.isdigit().fillna(False)].copy()
        if df_trades_only.empty:
            logger.info("No trades found in history file to summarize.")
            return
        df_trades_only['PNL_AccountCCY'] = pd.to_numeric(df_trades_only['PNL_AccountCCY'], errors='coerce')
        df_closed = df_trades_only[df_trades_only['PNL_AccountCCY'].notna()].copy()
        if df_closed.empty:
            logger.info("No closed trades found to summarize.")
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
        with open(csv_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([]); writer.writerow(["--- Performance Summary ---", f"Generated: {datetime.now(timezone.utc).isoformat()} ---"])
            writer.writerow(["Metric", "Value"]); writer.writerows(summary_data)
        logger.info(f"Performance summary appended to {csv_filepath}")
    except Exception as e:
        logger.error(f"Error calculating or appending performance summary: {e}", exc_info=True)

# --- MT5 Initialization and Shutdown ---

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

# --- Live Bot Helper Functions ---

def is_weekend_utc():
    now_utc = datetime.now(timezone.utc)
    return now_utc.weekday() >= 5

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

def get_live_data(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        logger.warning(f"No live data for {symbol} on {timeframe}. Err: {mt5.last_error()}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

# --- NEW: Helper function for candle-aware loop ---
def get_latest_m5_candle_time(reference_symbol="EURUSD"):
    """
    Fetches the timestamp of the most recent M5 candle from the broker.
    Uses a liquid symbol as a market clock.
    """
    try:
        rates = mt5.copy_rates_from_pos(reference_symbol, mt5.TIMEFRAME_M5, 0, 1)
        if rates is None or len(rates) == 0:
            logger.warning(f"Could not fetch latest candle time for {reference_symbol}. Last error: {mt5.last_error()}")
            return None
        return rates[0]['time'] # The 'time' field is a Unix timestamp (seconds)
    except Exception as e:
        logger.error(f"Error in get_latest_m5_candle_time: {e}")
        return None

# --- LOGIC CHANGE: This function now returns the MOST RECENTLY CLOSED candle ---
def fetch_latest_data(symbol):
    df_h4 = get_live_data(symbol, mt5.TIMEFRAME_H4, 100)
    df_h1 = get_live_data(symbol, mt5.TIMEFRAME_H1, 100)
    df_m5 = get_live_data(symbol, mt5.TIMEFRAME_M5, 100)

    # Ensure we have at least 2 M5 candles to get the last closed one
    if df_h4.empty or df_h1.empty or df_m5.empty or len(df_m5) < 2:
        logger.warning(f"Could not fetch complete data for {symbol} or not enough M5 candles. Skipping analysis.")
        return None, None # Return a tuple to avoid unpacking errors

    # H4 Indicators
    df_h4['H4_EMA8'] = ta.ema(df_h4['close'], length=8)
    df_h4['H4_EMA21'] = ta.ema(df_h4['close'], length=21)
    df_h4['RSI_H4'] = ta.rsi(df_h4['close'], length=14)

    # H1 Indicators
    df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
    df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
    df_h1['RSI_H1'] = ta.rsi(df_h1['close'], length=14)
    df_h1.rename(columns={'close': 'H1_Close_For_Bias'}, inplace=True)

    # M5 Indicators
    df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
    df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
    df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
    df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
    df_m5['RSI_M5'] = ta.rsi(df_m5['close'], length=14)

    # Combine dataframes
    combined_df = pd.merge_asof(df_m5.sort_index(), df_h1[['H1_Close_For_Bias', 'H1_EMA8', 'H1_EMA21', 'RSI_H1']].sort_index(),
                                left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta(hours=1))
    combined_df = pd.merge_asof(combined_df.sort_index(), df_h4[['H4_EMA8', 'H4_EMA21', 'RSI_H4']].sort_index(),
                                left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta(hours=4))

    combined_df.dropna(inplace=True)
    # Ensure we still have enough data after dropping NA
    if combined_df.empty or len(combined_df) < 2:
        logger.warning(f"Not enough data for {symbol} after combining and cleaning. Skipping analysis.")
        return None, None

    # Return the full DataFrame and the *most recently completed* candle (iloc[-2])
    # This ensures all analysis is done on closed data, mimicking the backtester.
    return combined_df, combined_df.iloc[-2]

# --- Strategy Logic (Unchanged) ---

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

# --- Live Trading Actions ---

def place_pending_order(symbol, props, order_type, entry_price, sl_price, lot_size, comment):
    trade_type = mt5.ORDER_TYPE_BUY_STOP if order_type == "BUY_STOP" else mt5.ORDER_TYPE_SELL_STOP
    request = {
        "action": mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": lot_size,
        "type": trade_type, "price": entry_price, "sl": sl_price, "tp": 0.0,
        "magic": 202405, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
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

# --- Main Management Routines ---

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
        entry_deals_df = deals_df[deals_df['entry'] == mt5.DEAL_ENTRY_IN]
        exit_deals_df = deals_df[deals_df['entry'] == mt5.DEAL_ENTRY_OUT]
        if exit_deals_df.empty:
            logger.warning(f"Position {pos_id} is closed, but no 'OUT' deal found in history yet. Retrying next cycle.")
            continue
        if entry_deals_df.empty:
            logger.error(f"CRITICAL: Position {pos_id} has an exit deal but no entry deal. Skipping update.")
            if pos_id in logged_open_position_ids: logged_open_position_ids.remove(pos_id)
            if pos_id in trade_details_for_closure: del trade_details_for_closure[pos_id]
            continue
        entry_deal = entry_deals_df.iloc[0]
        exit_deal = exit_deals_df.iloc[-1]
        commission_cost = COMMISSIONS.get(exit_deal.symbol, 0.0)
        net_pnl = exit_deal.profit + exit_deal.commission + exit_deal.swap - commission_cost
        update_values = {
            'CloseTimeUTC': pd.to_datetime(exit_deal.time, unit='s', utc=True).isoformat(),
            'ExitPrice': exit_deal.price, 'PNL_AccountCCY': f"{net_pnl:.2f}",
            'CloseReason': f"Closed by broker: {exit_deal.comment}"}
        update_closed_trade_in_csv(pos_id, update_values)
        logged_open_position_ids.remove(pos_id)
        if pos_id in trade_details_for_closure:
            del trade_details_for_closure[pos_id]

def manage_open_positions():
    open_positions = mt5.positions_get(magic=202405)
    if not open_positions: return

    for position in open_positions:
        pos_id_str = str(position.ticket)
        if pos_id_str not in logged_open_position_ids:
            risk_val_diff = abs(position.price_open - position.sl)
            tp_price = position.price_open + (4 * risk_val_diff) if position.type == mt5.ORDER_TYPE_BUY else position.price_open - (4 * risk_val_diff)
            props = ALL_SYMBOL_PROPERTIES[position.symbol]
            tp_price = round(tp_price, props['digits'])
            modify_position_sltp(position, position.sl, tp_price)
            risk_amount = (risk_val_diff / props['trade_tick_size']) * props['trade_tick_value'] * position.volume if props['trade_tick_size'] > 0 else 0
            trade_data = {
                "TicketID": position.ticket, "PositionID": position.ticket, "Symbol": position.symbol,
                "Type": "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL",
                "OpenTimeUTC": pd.to_datetime(position.time, unit='s', utc=True).isoformat(),
                "EntryPrice": position.price_open, "LotSize": position.volume, "SL_Price": position.sl,
                "TP_Price": tp_price, "CloseTimeUTC": "", "ExitPrice": "", "PNL_AccountCCY": "",
                "OpenComment": position.comment, "CloseReason": "", "RiskedAmount": f"{risk_amount:.2f}"}
            append_trade_to_csv(trade_data)
            logged_open_position_ids.add(pos_id_str)
            trade_details_for_closure[pos_id_str] = {'symbol': position.symbol, 'original_sl': position.sl, 'current_sl': position.sl, 'trailing_active': False, 'ts_next_atr_level': 1.5}
            continue

        details = trade_details_for_closure.get(pos_id_str)
        if not details: continue

        # --- LOGIC CHANGE: Use closed candle data for management ---
        df, last_closed_candle = fetch_latest_data(position.symbol)
        if df is None or last_closed_candle is None:
            logger.warning(f"Could not get valid closed candle data for managing position {position.ticket}. Skipping this cycle.")
            continue

        atr_val = last_closed_candle.get('ATR', np.nan)
        if pd.isna(atr_val) or atr_val <= 0: continue

        # Trailing is now based on the high/low of the *last closed candle*
        move_from_entry = (last_closed_candle['high'] - position.price_open) if position.type == mt5.ORDER_TYPE_BUY else (position.price_open - last_closed_candle['low'])
        atr_movement = move_from_entry / atr_val

        if atr_movement >= details['ts_next_atr_level']:
            props = ALL_SYMBOL_PROPERTIES[position.symbol]
            last_3_closed = df.iloc[-4:-1] # 3 closed candles before the last one
            new_sl = 0

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = last_3_closed['low'].min() - 2 * props['pip_value_calc']
                if new_sl > details['current_sl']:
                    logger.info(f"[{position.symbol}] Trailing SL for BUY {position.ticket}. New SL: {new_sl:.{props['digits']}f}")
                    if modify_position_sltp(position, round(new_sl, props['digits']), position.tp):
                        details['current_sl'] = new_sl
            else: # SELL
                new_sl = last_3_closed['high'].max() + 2 * props['pip_value_calc']
                if new_sl < details['current_sl']:
                    logger.info(f"[{position.symbol}] Trailing SL for SELL {position.ticket}. New SL: {new_sl:.{props['digits']}f}")
                    if modify_position_sltp(position, round(new_sl, props['digits']), position.tp):
                        details['current_sl'] = new_sl
            details['ts_next_atr_level'] += 0.5

def manage_pending_orders():
    pending_orders = mt5.orders_get(magic=202405)
    if not pending_orders: return

    for order in pending_orders:
        # --- LOGIC CHANGE: Use closed candle data for management ---
        _, last_closed_candle = fetch_latest_data(order.symbol)
        if last_closed_candle is None:
            logger.warning(f"Could not get valid closed candle data for managing pending order {order.ticket}. Skipping this cycle.")
            continue

        setup_bias = "BUY" if order.type == mt5.ORDER_TYPE_BUY_STOP else "SELL"
        m5_ema21 = last_closed_candle['M5_EMA21']

        if (setup_bias == "BUY" and last_closed_candle['close'] < m5_ema21) or \
           (setup_bias == "SELL" and last_closed_candle['close'] > m5_ema21):
            logger.info(f"[{order.symbol}] PENDING order {order.ticket} invalidated (Close vs M5_EMA21). Cancelling...")
            cancel_pending_order(order.ticket)

def check_for_new_signals(daily_risk_allocated, max_daily_risk):
    global delayed_setups_queue

    new_queue, order_placed_this_cycle = [], False
    for setup in delayed_setups_queue:
        setup['confirm_count'] += 1
        if setup['confirm_count'] < 2:
            new_queue.append(setup); continue
        
        _, last_closed_candle = fetch_latest_data(setup['symbol'])
        if last_closed_candle is None:
            new_queue.append(setup); continue

        # if (setup['bias'] == "BUY" and last_closed_candle['close'] < last_closed_candle['M5_EMA21']) or \
        #    (setup['bias'] == "SELL" and last_closed_candle['close'] > last_closed_candle['M5_EMA21']):
        #     logger.info(f"[{setup['symbol']}] Delayed {setup['bias']} setup invalidated on confirmation. Discarding.")
        #     continue

        if daily_risk_allocated + setup["risk_amt"] > max_daily_risk:
            logger.warning(f"[{setup['symbol']}] Delayed setup confirmed, but would exceed daily risk limit. Discarding.")
            continue

        props = ALL_SYMBOL_PROPERTIES[setup['symbol']]
        order_ticket = place_pending_order(setup['symbol'], props, f"{setup['bias']}_STOP", setup['entry_price'], setup['sl_price'], setup['lot_size'], "LiveBot_v1_Delayed")
        if order_ticket:
            daily_risk_allocated += setup["risk_amt"]
            order_placed_this_cycle = True
            break
        else:
            new_queue.append(setup)
    delayed_setups_queue = new_queue
    if order_placed_this_cycle: return daily_risk_allocated

    for symbol in SYMBOLS_AVAILABLE_FOR_TRADE:
        if is_weekend_utc() and symbol not in CRYPTO_SYMBOLS: continue
        if not is_within_session(TRADING_SESSIONS_UTC.get(symbol, [])): continue
        if not is_outside_news_blackout(symbol, NEWS_TIMES_UTC): continue

        df, last_closed_candle = fetch_latest_data(symbol)
        if df is None or last_closed_candle is None: continue

        props = ALL_SYMBOL_PROPERTIES[symbol]; pip_adj = 3 * props['trade_tick_size']

        h1_trend_bias = "BUY" if last_closed_candle['H1_EMA8'] > last_closed_candle['H1_EMA21'] and last_closed_candle['H1_Close_For_Bias'] > last_closed_candle['H1_EMA8'] else "SELL" if last_closed_candle['H1_EMA8'] < last_closed_candle['H1_EMA21'] and last_closed_candle['H1_Close_For_Bias'] < last_closed_candle['H1_EMA8'] else None
        if not h1_trend_bias: continue

        m5_fanned_buy = last_closed_candle['M5_EMA8'] > last_closed_candle['M5_EMA13'] or last_closed_candle['M5_EMA8'] > last_closed_candle['M5_EMA21']
        m5_fanned_sell = last_closed_candle['M5_EMA8'] < last_closed_candle['M5_EMA13'] or last_closed_candle['M5_EMA8'] < last_closed_candle['M5_EMA21']
        if not ((h1_trend_bias == "BUY" and m5_fanned_buy) or (h1_trend_bias == "SELL" and m5_fanned_sell)): continue
        if (h1_trend_bias == "BUY" and last_closed_candle['H4_EMA8'] < last_closed_candle['H4_EMA21']) or (h1_trend_bias == "SELL" and last_closed_candle['H4_EMA8'] > last_closed_candle['H4_EMA21']): continue
        if (h1_trend_bias == "BUY" and not (last_closed_candle['RSI_M5'] > 50 and last_closed_candle['RSI_H1'] > 50)) or (h1_trend_bias == "SELL" and not (last_closed_candle['RSI_M5'] < 50 and last_closed_candle['RSI_H1'] < 50)): continue
        if (h1_trend_bias == "BUY" and last_closed_candle['close'] < last_closed_candle['M5_EMA21']) or (h1_trend_bias == "SELL" and last_closed_candle['close'] > last_closed_candle['M5_EMA21']): continue
        if not ((h1_trend_bias == "BUY" and last_closed_candle['low'] <= last_closed_candle['M5_EMA8']) or (h1_trend_bias == "SELL" and last_closed_candle['high'] >= last_closed_candle['M5_EMA8'])): continue

        recent_candles = df.iloc[-6:-2]; bullish_count = (recent_candles['close'] > recent_candles['open']).sum(); bearish_count = (recent_candles['close'] < recent_candles['open']).sum()
        if (h1_trend_bias == "BUY" and bullish_count > 2) or (h1_trend_bias == "SELL" and bearish_count > 2): continue

        lookback_window = df.iloc[-12:-2]; swing_high, swing_low = lookback_window['high'].max(), lookback_window['low'].min()
        impulse_start, impulse_end, price_for_pb = (swing_low, swing_high, last_closed_candle['low']) if h1_trend_bias == "BUY" else (swing_high, swing_low, last_closed_candle['high'])
        if calculate_pullback_depth(impulse_start, impulse_end, price_for_pb, h1_trend_bias) < 0.30: continue

        fib_levels = calculate_fib_levels(swing_high, swing_low); tolerance = 0.5 * last_closed_candle['ATR']
        if not any(abs(last_closed_candle['M5_EMA8'] - fib_price) <= tolerance or abs(last_closed_candle['M5_EMA13'] - fib_price) <= tolerance for fib_price in fib_levels.values()): continue

        entry_lookback = df.iloc[-4:-1]; entry_px, sl_px = (0,0)
        if h1_trend_bias == "BUY":
            entry_px = entry_lookback['high'].max() + pip_adj
            sl_px = entry_px - (2.5 * last_closed_candle['ATR'])
        else:
            entry_px = entry_lookback['low'].min() - pip_adj
            sl_px = entry_px + (2.5 * last_closed_candle['ATR'])
        entry_px, sl_px = round(entry_px, props['digits']), round(sl_px, props['digits'])
        if abs(entry_px - sl_px) <= 0: continue

        lot_size = props['volume_min']; est_risk = lot_size * (abs(entry_px - sl_px) / props['trade_tick_size']) * props['trade_tick_value']
        if est_risk > mt5.account_info().balance * RISK_PER_TRADE_PERCENT:
            logger.info(f"[{symbol}] Setup found but min lot risk ({est_risk:.2f}) exceeds max allowed. Skipping.")
            continue

        delayed_setups_queue.append({"symbol": symbol, "bias": h1_trend_bias, "entry_price": entry_px, "sl_price": sl_px, "lot_size": lot_size, "risk_amt": est_risk, "confirm_count": 0})
        logger.info(f"[{symbol}] SETUP QUEUED for delayed confirmation. Bias: {h1_trend_bias}, Entry: {entry_px}")
        break
    return daily_risk_allocated

# --- Main Execution ---

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

    # --- NEW: State variable for candle-based execution ---
    last_processed_candle_time = None
    REFERENCE_CLOCK_SYMBOL = "EURUSD"
    if REFERENCE_CLOCK_SYMBOL not in SYMBOLS_AVAILABLE_FOR_TRADE:
        REFERENCE_CLOCK_SYMBOL = SYMBOLS_AVAILABLE_FOR_TRADE[0] # Fallback
    logger.info(f"Using {REFERENCE_CLOCK_SYMBOL} as the market clock for new M5 candles.")

    logger.info("--- Live Trading Bot Started ---")
    logger.info(f"Initial daily risk budget: {max_daily_risk_budget:.2f} USD")
    logger.info("Bot is now waiting for the close of the next M5 candle to begin...")

    try:
        while True:
            # --- NEW: Candle-aware loop logic ---
            current_candle_time = get_latest_m5_candle_time(REFERENCE_CLOCK_SYMBOL)

            # Check if a new M5 candle has formed and we haven't processed it yet
            if current_candle_time is not None and current_candle_time != last_processed_candle_time:
                logger.info(f"New M5 candle detected. Time: {datetime.fromtimestamp(current_candle_time, tz=timezone.utc)}. Processing...")

                # First, update the state to prevent reprocessing this same candle
                last_processed_candle_time = current_candle_time

                # --- ALL TRADING LOGIC IS NOW INSIDE THIS BLOCK ---
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

                logger.info("Processing complete for this candle. Waiting for next candle...")

            # Sleep for a short duration to poll for new candle without high CPU usage
            time.sleep(15)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting gracefully...")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("Performing final shutdown tasks.")
        calculate_and_append_performance_summary(TRADE_HISTORY_FILE, session_start_balance)
        shutdown_mt5_interface()