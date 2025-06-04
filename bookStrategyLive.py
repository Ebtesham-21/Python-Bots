# --- START OF FILE bookStrategyLive.py ---

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta # For EMAs and ATR
import numpy as np
import time
from datetime import datetime, timedelta, date, timezone
import logging
import math
import os # For file operations
import csv # For CSV writing

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}
RUN_BACKTEST = False
BOT_MAGIC_NUMBER = 12345

TRADE_HISTORY_FILE = "bookStrategy_trade_history.csv"
CSV_HEADERS = ["TicketID", "PositionID", "Symbol", "Type", "OpenTimeUTC", "EntryPrice",
               "LotSize", "SL_Price", "TP_Price", "CloseTimeUTC", "ExitPrice",
               "PNL_AccountCCY", "OpenComment", "CloseReason", "RiskedAmount"]

# In-memory state for current session
logged_open_position_ids = set() # Stores position.identifier of trades logged as open in CSV
trade_details_for_closure = {} # Stores details needed for when a trade closes {pos_id: {details}}


# --- Strategy & Trading Parameters ---
SYMBOLS_TO_TRADE = ["EURUSD", "AUDUSD", "USDCHF", "USDCAD",
                    "GBPJPY", "XAUUSD", "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                    "USOIL", "CADJPY", "XAGUSD", "XPTUSD", "UKOIL",
                    "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD" ,"XAGGBP", "XAGEUR", "XAGAUD", "BTCXAG"]

TRADING_SESSIONS_UTC = {
    "EURUSD": [(7, 16)], "GBPUSD": [(7, 16)], "AUDUSD": [(0, 4), (7, 16)],
    "USDCHF": [(7, 16)], "USDCAD": [(12, 17)], "USDJPY": [(0, 4), (12, 17)],
    "EURJPY": [(0, 4), (7, 12)], "GBPJPY": [(7, 16)], "NZDUSD": [(0, 4), (7, 16)],
    "EURCHF": [(7, 12)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 17)],
    "EURNZD": [(0, 4), (7, 12)], "GBPNZD": [(7, 12)], "XAUUSD": [(7, 16)],
    "XAGUSD": [(7, 16)], "XPTUSD": [(7, 16)], "USOIL": [(12, 17)], "UKOIL": [(7, 16)], "XAGGBP":[(7, 16)], "XAGEUR":[(7,16)], "XAGAUD": [(0,4), (7,10)], "BTCXAG":[(7,16)]
}

CRYPTO_SESSIONS_USER = {"BTCUSD":[(0, 16)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 16)], "ETHUSD":[(0, 16)]}
for crypto_sym, sess_val in CRYPTO_SESSIONS_USER.items():
    TRADING_SESSIONS_UTC[crypto_sym] = sess_val

RISK_PER_TRADE_PERCENT = 0.01
DAILY_RISK_LIMIT_PERCENT = 0.05
MAX_SPREAD_PIPS_DEFAULT = 7.0
MAX_SPREAD_PIPS_MAP = {
    "BTCUSD": 250.0, "BTCJPY": 35000.0, "ETHUSD": 25.0, "BTCXAU": 75.0,
    "XAUUSD": 20.0, "USOIL": 10.0, "UKOIL": 10.0,
}

H1_CANDLES_TO_FETCH = 50
M5_CANDLES_TO_FETCH = 50 # Ensure this is enough for ATR (e.g., 14 periods + buffer)
H4_CANDLES_TO_FETCH = 50
ATR_PERIOD_M5 = 14
LOOP_SLEEP_SECONDS = 30
LOGGING_INTERVAL_MINUTES = 5


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
        # Filter out summary lines if they exist from previous runs
        df = df[~df['TicketID'].astype(str).str.contains("--- Performance Summary ---|Metric", na=False)]
        
        open_trades_df = df[df['CloseTimeUTC'].isnull() | (df['CloseTimeUTC'] == '')]
        for _, row in open_trades_df.iterrows():
            pos_id = str(row['PositionID'])
            logged_open_position_ids.add(pos_id)
            trade_details_for_closure[pos_id] = {
                'original_sl': row['SL_Price'],
                'original_tp': row['TP_Price'],
                'symbol': row['Symbol'],
                'current_sl': row['SL_Price'], 
                'current_tp': row['TP_Price']
            }
        logger.info(f"Loaded {len(logged_open_position_ids)} open positions' IDs from {TRADE_HISTORY_FILE}")
    except pd.errors.EmptyDataError:
        logger.info(f"{TRADE_HISTORY_FILE} is empty. Starting with empty state.")
    except Exception as e:
        logger.error(f"Error loading state from CSV {TRADE_HISTORY_FILE}: {e}")

def append_trade_to_csv(trade_data_dict):
    try:
        trade_data_dict['PositionID'] = str(trade_data_dict['PositionID'])

        file_exists_and_has_content = os.path.exists(TRADE_HISTORY_FILE) and os.path.getsize(TRADE_HISTORY_FILE) > 0
        with open(TRADE_HISTORY_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            # Check if the file is empty or only contains summary lines before writing headers
            is_truly_empty_or_summary_only = True
            if file_exists_and_has_content:
                try:
                    # Quick check for actual trade data
                    df_check = pd.read_csv(TRADE_HISTORY_FILE, nrows=5) # Read a few rows
                    if CSV_HEADERS[0] in df_check.columns and len(df_check[df_check[CSV_HEADERS[0]] != "--- Performance Summary ---"]) > 0 :
                         is_truly_empty_or_summary_only = False
                except pd.errors.EmptyDataError:
                    pass # File is empty
                except Exception:
                    pass # Could be malformed, assume it needs headers if we write

            if not file_exists_and_has_content or is_truly_empty_or_summary_only :
                 if os.path.getsize(TRADE_HISTORY_FILE) == 0 : # Explicitly check size for new file
                    writer.writeheader()


            writer.writerow(trade_data_dict)
        logger.info(f"Appended open trade (PosID: {trade_data_dict['PositionID']}) to {TRADE_HISTORY_FILE}.")
    except IOError as e:
        logger.error(f"IOError appending to CSV {TRADE_HISTORY_FILE}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error appending to CSV {TRADE_HISTORY_FILE}: {e}")


def update_closed_trade_in_csv(position_id_to_update, update_values_dict):
    position_id_to_update_str = str(position_id_to_update)
    try:
        if not os.path.exists(TRADE_HISTORY_FILE):
            logger.error(f"CSV file {TRADE_HISTORY_FILE} not found. Cannot update.")
            initialize_trade_history_file() 
            return

        # Read lines, update in memory, write all back
        # This is safer than pandas for partial updates if summary exists
        lines = []
        updated = False
        with open(TRADE_HISTORY_FILE, 'r', newline='') as f_read:
            reader = csv.reader(f_read)
            try:
                header = next(reader)
                lines.append(header)
                for row in reader:
                    if row and len(row) == len(CSV_HEADERS): # Check for valid trade row
                        if row[CSV_HEADERS.index('PositionID')] == position_id_to_update_str and \
                           (not row[CSV_HEADERS.index('CloseTimeUTC')] or row[CSV_HEADERS.index('CloseTimeUTC')] == ''):
                            for key, value in update_values_dict.items():
                                if key in CSV_HEADERS:
                                    row[CSV_HEADERS.index(key)] = value
                            updated = True
                    lines.append(row) # Append original or updated row, or summary lines
            except StopIteration: # Empty file after header or completely empty
                 logger.error(f"CSV file {TRADE_HISTORY_FILE} seems to be empty or header only. Cannot update.")
                 return


        if updated:
            with open(TRADE_HISTORY_FILE, 'w', newline='') as f_write:
                writer = csv.writer(f_write)
                writer.writerows(lines)
            logger.info(f"Updated closed trade (PosID: {position_id_to_update_str}) in {TRADE_HISTORY_FILE}.")
        else:
            logger.warning(f"Could not find open trade with PositionID {position_id_to_update_str} in {TRADE_HISTORY_FILE} to update for closure. It might have been updated already or was never logged as open.")

    except FileNotFoundError:
        logger.error(f"CSV file {TRADE_HISTORY_FILE} not found (safeguard). Cannot update.")
    except Exception as e:
        logger.error(f"Error updating CSV for position {position_id_to_update_str}: {e}")

# --- Performance Summary Function ---
def calculate_and_append_performance_summary(csv_filepath, session_initial_balance):
    logger.info(f"Calculating performance summary for trades in {csv_filepath} using session initial balance: {session_initial_balance:.2f}")
    if not os.path.exists(csv_filepath):
        logger.warning(f"Trade history file {csv_filepath} not found. Cannot calculate performance summary.")
        return

    try:
        df_all = pd.read_csv(csv_filepath, dtype={'PositionID': str, 'PNL_AccountCCY': str})
        # Remove any pre-existing summary lines before processing
        df_trades_only = df_all[~df_all[CSV_HEADERS[0]].astype(str).str.contains("--- Performance Summary ---|Metric", na=False)].copy()
        
        if df_trades_only.empty:
            logger.info("No trade data in CSV to summarize.")
            # Optionally, write an empty summary or just return
            with open(csv_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([]) # Blank line
                writer.writerow(["--- Performance Summary ---", "No Trades Found ---"])
            return

        # Coerce PNL to numeric, errors='coerce' will turn non-numeric PNLs (e.g., empty strings) into NaN
        df_trades_only['PNL_AccountCCY'] = pd.to_numeric(df_trades_only['PNL_AccountCCY'], errors='coerce')
        
        # Filter for closed trades with valid PNL
        df_closed = df_trades_only[
            (df_trades_only['CloseTimeUTC'].notna()) & (df_trades_only['CloseTimeUTC'] != '') &
            (df_trades_only['PNL_AccountCCY'].notna())
        ].copy()

        if df_closed.empty:
            logger.info("No closed trades with PNL data found in CSV to summarize.")
            with open(csv_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(["--- Performance Summary ---", "No Closed Trades with PNL Found ---"])
            return

        df_closed['CloseTimeUTC_dt'] = pd.to_datetime(df_closed['CloseTimeUTC'], errors='coerce', utc=True)
        df_closed = df_closed.sort_values(by='CloseTimeUTC_dt').reset_index(drop=True)

        total_closed_trades = len(df_closed)
        winning_trades = len(df_closed[df_closed['PNL_AccountCCY'] > 0])
        losing_trades = len(df_closed[df_closed['PNL_AccountCCY'] < 0])
        total_pnl = df_closed['PNL_AccountCCY'].sum()

        gross_profit = df_closed[df_closed['PNL_AccountCCY'] > 0]['PNL_AccountCCY'].sum()
        gross_loss = abs(df_closed[df_closed['PNL_AccountCCY'] < 0]['PNL_AccountCCY'].sum())

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0: # Gross loss is 0, but profit is positive
            profit_factor = float('inf')
        else: # Both gross profit and gross loss are 0
            profit_factor = 0.0 

        # Max Drawdown Calculation
        equity_curve = [session_initial_balance]
        current_equity = session_initial_balance
        for pnl_val in df_closed['PNL_AccountCCY']:
            current_equity += pnl_val
            equity_curve.append(current_equity)

        max_drawdown_usd = 0.0
        max_drawdown_percent = 0.0
        
        if equity_curve: # Should always be true if df_closed is not empty
            peak_val_for_dd = equity_curve[0]
            for i in range(len(equity_curve)):
                peak_val_for_dd = max(peak_val_for_dd, equity_curve[i])
                drawdown_usd = peak_val_for_dd - equity_curve[i]
                if drawdown_usd > max_drawdown_usd:
                    max_drawdown_usd = drawdown_usd
                
                if peak_val_for_dd > 0: # Avoid division by zero if equity peak is zero or negative
                    drawdown_pct = (drawdown_usd / peak_val_for_dd) * 100
                    if drawdown_pct > max_drawdown_percent:
                        max_drawdown_percent = drawdown_pct
                # If peak_val_for_dd is 0 or negative, drawdown_pct is less meaningful or undefined.
                # max_drawdown_percent will remain based on prior valid calculations or 0.

        summary_data = [
            ["Total Closed Trades", total_closed_trades],
            ["Winning Trades", winning_trades],
            ["Losing Trades", losing_trades],
            ["Total PNL (Account CCY)", f"{total_pnl:.2f}"],
            ["Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinity"],
            ["Max Drawdown (USD)", f"{max_drawdown_usd:.2f}"],
            ["Max Drawdown (%)", f"{max_drawdown_percent:.2f}%"]
        ]

        with open(csv_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([]) # Blank line separator
            writer.writerow(["--- Performance Summary ---", f"Generated: {datetime.now(timezone.utc).isoformat()} ---"])
            writer.writerow(["Metric", "Value"])
            for row in summary_data:
                writer.writerow(row)
        logger.info(f"Performance summary appended to {csv_filepath}")

    except pd.errors.EmptyDataError:
        logger.info(f"{csv_filepath} is empty. No summary generated.")
    except Exception as e:
        logger.error(f"Error calculating or appending performance summary: {e}", exc_info=True)


# --- MT5 Initialization and Shutdown ---
def initialize_mt5_interface(symbols_to_check):
    global SYMBOLS_AVAILABLE_FOR_TRADE, ALL_SYMBOL_PROPERTIES
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    logger.info("MetaTrader 5 Initialized")
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to get account info, error code = {mt5.last_error()}")
        mt5.shutdown(); return False
    else:
        logger.info(f"Account Login: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")

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

        if symbol_info_obj.point == 0: logger.warning(f"Symbol {symbol_name} point value is 0. Skipping."); continue
        if symbol_info_obj.trade_tick_size == 0: logger.warning(f"Symbol {symbol_name} trade_tick_size is 0. Skipping."); continue

        point = symbol_info_obj.point
        digits = symbol_info_obj.digits
        sym_upper = symbol_name.upper()
        current_pip_value = 0.0

        if "BTCUSD" == sym_upper or "ETHUSD" == sym_upper: current_pip_value = 1.0
        elif "BTCJPY" == sym_upper: current_pip_value = 100.0
        elif "BTCXAU" == sym_upper:
            current_pip_value = 10 * point if digits >= 4 else (point if digits > 0 else point)
        elif 'JPY' in sym_upper: current_pip_value = 0.01
        elif sym_upper in ["XAUUSD", "XAGUSD", "XPTUSD"] or "OIL" in sym_upper or "USOIL" in sym_upper or "UKOIL" in sym_upper:
            current_pip_value = 0.01
        elif point > 0:
            if digits == 5 or digits == 3: current_pip_value = 10 * point
            elif digits == 2 or digits == 4: current_pip_value = point
            else:
                logger.warning(f"Uncommon digit count ({digits}) for {symbol_name}. Using point ({point}) as pip_value_calc. Please verify.")
                current_pip_value = point
        else: logger.error(f"Could not determine pip value for {symbol_name}, point is {point}, digits {digits}. Skipping."); continue

        if current_pip_value <= 1e-9:
            logger.error(f"Calculated pip_value_calc for {symbol_name} is zero or too small ({current_pip_value}). Point: {point}, Digits: {digits}. Skipping symbol.")
            continue

        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point, 'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
            'trade_tick_value': symbol_info_obj.trade_tick_value_profit if symbol_info_obj.currency_profit == account_info.currency else symbol_info_obj.trade_tick_value_loss,
            'volume_min': symbol_info_obj.volume_min,
            'volume_step': symbol_info_obj.volume_step if symbol_info_obj.volume_step > 0 else 0.01,
            'volume_max': symbol_info_obj.volume_max,
            'trade_contract_size': symbol_info_obj.trade_contract_size,
            'currency_profit': symbol_info_obj.currency_profit,
            'currency_margin': symbol_info_obj.currency_margin,
            'pip_value_calc': current_pip_value,
            'description': symbol_info_obj.description
        }
        successfully_initialized_symbols.append(symbol_name)

    if not successfully_initialized_symbols: logger.error("No symbols were successfully initialized."); return False
    SYMBOLS_AVAILABLE_FOR_TRADE = successfully_initialized_symbols
    ALL_SYMBOL_PROPERTIES = temp_symbol_properties
    logger.info(f"Properties recorded for: {SYMBOLS_AVAILABLE_FOR_TRADE}")
    return True

def shutdown_mt5_interface():
    mt5.shutdown()
    logger.info("MetaTrader 5 Shutdown")

# --- Helper Functions ---
def is_within_session(symbol_sessions):
    if not symbol_sessions: return True
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    for start_hour, end_hour in symbol_sessions:
        if start_hour <= current_hour < end_hour: return True
    return False

def calculate_lot_size(account_balance_for_risk_calc, risk_percent, sl_price_diff, symbol_props):
    if sl_price_diff <= 0:
        logger.warning(f"SL price difference is not positive ({sl_price_diff}) for {symbol_props.get('description', 'N/A')}. Cannot calculate lot size.")
        return 0
    risk_amount_currency = account_balance_for_risk_calc * risk_percent
    if 'trade_tick_size' not in symbol_props or symbol_props['trade_tick_size'] <= 1e-9 or \
       'trade_tick_value' not in symbol_props or symbol_props['trade_tick_value'] <= 1e-9:
        logger.error(f"Symbol {symbol_props.get('description', 'N/A')} has zero, negative, or missing tick_size/tick_value. Props: {symbol_props}")
        return 0
    sl_distance_ticks = sl_price_diff / symbol_props['trade_tick_size']
    sl_cost_per_lot = sl_distance_ticks * symbol_props['trade_tick_value']
    if sl_cost_per_lot <= 1e-9:
        logger.warning(f"Calculated SL cost per lot is too low or zero for {symbol_props.get('description', 'N/A')}: {sl_cost_per_lot}. SL diff: {sl_price_diff}, TickSize: {symbol_props['trade_tick_size']}, TickValue: {symbol_props['trade_tick_value']}")
        return 0
    lot_size = risk_amount_currency / sl_cost_per_lot
    volume_step = symbol_props.get('volume_step', 0.01)
    if volume_step <= 1e-9: volume_step = 0.01
    lot_size = math.floor(lot_size / volume_step) * volume_step
    volume_min = symbol_props.get('volume_min', 0.01)
    volume_max = symbol_props.get('volume_max', 1000.0)
    if lot_size < volume_min:
        cost_at_min_volume = volume_min * sl_cost_per_lot
        if cost_at_min_volume > risk_amount_currency * 1.5: # Allow slightly higher risk for min lot
            logger.warning(f"Min lot size ({volume_min}) for {symbol_props.get('description','N/A')} results in risk ({cost_at_min_volume:.2f}) > 1.5x target risk ({risk_amount_currency:.2f}). Calculated lot: {lot_size:.{int(-math.log10(volume_step)) if volume_step > 0 else 2}f}. Skipping.")
            return 0
        logger.info(f"Lot size {lot_size} too small for {symbol_props.get('description','N/A')}, using min_volume {volume_min}. Target risk: {risk_amount_currency:.2f}, Risk at min vol: {cost_at_min_volume:.2f}")
        lot_size = volume_min
    lot_size = min(lot_size, volume_max)
    if lot_size < volume_min and abs(lot_size - volume_min) > 1e-9: # Final check
         logger.warning(f"Final lot size {lot_size} is less than min volume {volume_min} for {symbol_props.get('description','N/A')}. SL diff:{sl_price_diff}")
         return 0
    if lot_size <= 1e-9:
        logger.warning(f"Final lot size is effectively zero ({lot_size}) for {symbol_props.get('description','N/A')}. Skipping.")
        return 0
    precision = int(-math.log10(volume_step)) if volume_step > 1e-9 else 2
    return round(lot_size, precision)

def get_current_spread_pips(symbol, symbol_props):
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info:
        spread_points = tick_info.ask - tick_info.bid
        pip_value_for_calc = symbol_props.get('pip_value_calc', 0.0)
        if pip_value_for_calc <= 1e-9 :
            logger.warning(f"pip_value_calc for {symbol} is zero or too small ({pip_value_for_calc}). Cannot calculate spread in pips.")
            return float('inf')
        spread_pips = spread_points / pip_value_for_calc
        return spread_pips
    return float('inf')

def get_live_data_with_emas(symbol, h1_candles_req, m5_candles_req, h4_candles_req):
    latest_h1_data_series, latest_m5_data_series, m5_lookback_df_out, latest_h4_data_series = None, None, None, None

    if h1_candles_req > 0:
        rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, h1_candles_req)
        if rates_h1 is not None and len(rates_h1) > 0:
            df_h1 = pd.DataFrame(rates_h1); df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s', utc=True)
            if len(df_h1) >= 8:  df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
            if len(df_h1) >= 21: df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
            latest_h1_data_series = df_h1.iloc[-1].copy()

    if m5_candles_req > 0:
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, m5_candles_req)
        if rates_m5 is not None and len(rates_m5) > 0: 
            df_m5 = pd.DataFrame(rates_m5); df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s', utc=True)
            if len(df_m5) >= 8: df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
            if len(df_m5) >= 13: df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
            if len(df_m5) >= 21: df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
            
            if len(df_m5) >= ATR_PERIOD_M5:
                df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=ATR_PERIOD_M5)
            else:
                df_m5['ATR'] = np.nan 

            latest_m5_data_series = df_m5.iloc[-1].copy()
            m5_lookback_df_out = df_m5.iloc[-5:].copy() 

    if h4_candles_req > 0:
        rates_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, h4_candles_req)
        if rates_h4 is not None and len(rates_h4) > 0:
            df_h4 = pd.DataFrame(rates_h4)
            df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s', utc=True)
            if len(df_h4) >= 8: df_h4['H4_EMA8'] = ta.ema(df_h4['close'], length=8)
            if len(df_h4) >= 21: df_h4['H4_EMA21'] = ta.ema(df_h4['close'], length=21)
            latest_h4_data_series = df_h4.iloc[-1].copy()
    
    if latest_h1_data_series is None or latest_m5_data_series is None or latest_h4_data_series is None:
        logger.debug(f"Incomplete candle data fetched for {symbol}. H1: {'OK' if latest_h1_data_series is not None else 'Fail'}, M5: {'OK' if latest_m5_data_series is not None else 'Fail'}, H4: {'OK' if latest_h4_data_series is not None else 'Fail'}")

    return latest_h1_data_series, latest_m5_data_series, m5_lookback_df_out, latest_h4_data_series

# --- Signal Checking Function ---
def get_trade_signal_status(symbol, props, latest_h1, latest_m5, latest_h4):
    if props.get('pip_value_calc', 0.0) <= 1e-9:
        return "NONE", f"Invalid pip_value_calc for {symbol}"

    if latest_h1 is None or latest_m5 is None or latest_h4 is None:
        return "NONE", f"Incomplete candle data for {symbol} (H1:{latest_h1 is not None}, M5:{latest_m5 is not None}, H4:{latest_h4 is not None})"

    if not all(k in latest_h1 and not pd.isna(latest_h1[k]) for k in ['H1_EMA8', 'H1_EMA21', 'close']):
        return "NONE", f"H1 EMA data missing/NaN for {symbol}"
    h1_ema8, h1_ema21, h1_close = latest_h1['H1_EMA8'], latest_h1['H1_EMA21'], latest_h1['close']
    h1_trend_bias = None
    if h1_ema8 > h1_ema21 and h1_close > h1_ema8: h1_trend_bias = "BUY"
    elif h1_ema8 < h1_ema21 and h1_close < h1_ema8: h1_trend_bias = "SELL"
    if not h1_trend_bias:
        return "NONE", f"No H1 trend ({props.get('digits',2)}f) (EMA8:{h1_ema8:.{props['digits']}f}, EMA21:{h1_ema21:.{props['digits']}f}, Close:{h1_close:.{props['digits']}f})"

    if not all(k in latest_m5 and not pd.isna(latest_m5[k]) for k in ['M5_EMA8', 'M5_EMA13', 'M5_EMA21']):
        return "NONE", f"M5 EMA data missing/NaN for {symbol}"
    m5_ema8, m5_ema13, m5_ema21_val = latest_m5['M5_EMA8'], latest_m5['M5_EMA13'], latest_m5['M5_EMA21']
    m5_fanned = (h1_trend_bias == "BUY" and m5_ema8 > m5_ema13 > m5_ema21_val) or \
                (h1_trend_bias == "SELL" and m5_ema8 < m5_ema13 < m5_ema21_val)
    if not m5_fanned:
        return "NONE", f"M5 EMAs not fanned for {h1_trend_bias} (M5_8:{m5_ema8:.{props['digits']}f}, M5_13:{m5_ema13:.{props['digits']}f}, M5_21:{m5_ema21_val:.{props['digits']}f})"

    if pd.isna(latest_m5['close']):
        return "NONE", f"M5 close data missing/NaN for {symbol}"
    if (h1_trend_bias == "BUY" and latest_m5['close'] < m5_ema21_val) or \
       (h1_trend_bias == "SELL" and latest_m5['close'] > m5_ema21_val):
        return "NONE", f"M5 close ({latest_m5['close']:.{props['digits']}f}) vs M5_EMA21 ({m5_ema21_val:.{props['digits']}f}) invalidates {h1_trend_bias} setup"

    if pd.isna(latest_m5['low']) or pd.isna(latest_m5['high']):
        return "NONE", f"M5 low/high data missing/NaN for {symbol}"
    pullback = (h1_trend_bias == "BUY" and latest_m5['low'] <= m5_ema8) or \
               (h1_trend_bias == "SELL" and latest_m5['high'] >= m5_ema8)
    if not pullback:
        return "NONE", f"No M5 pullback to M5_EMA8 ({m5_ema8:.{props['digits']}f}). Low: {latest_m5['low']:.{props['digits']}f}, High: {latest_m5['high']:.{props['digits']}f}"

    if 'H4_EMA8' not in latest_h4 or pd.isna(latest_h4['H4_EMA8']) \
       or 'H4_EMA21' not in latest_h4 or pd.isna(latest_h4['H4_EMA21']):
        return "NONE", f"H4 EMA data not available or incomplete/NaN for {symbol}"
    h4_ema8_val = latest_h4['H4_EMA8']
    h4_ema21_val = latest_h4['H4_EMA21']

    if h1_trend_bias == "BUY" and h4_ema8_val < h4_ema21_val:
        return "NONE", f"H4 bias bearish (H4_8:{h4_ema8_val:.{props['digits']}f} < H4_21:{h4_ema21_val:.{props['digits']}f}), conflicts with BUY"
    if h1_trend_bias == "SELL" and h4_ema8_val > h4_ema21_val:
        return "NONE", f"H4 bias bullish (H4_8:{h4_ema8_val:.{props['digits']}f} > H4_21:{h4_ema21_val:.{props['digits']}f}), conflicts with SELL"

    return h1_trend_bias, f"{h1_trend_bias} signal: H1 Trend aligned, M5 EMAs fanned, M5 Pullback to EMA8, H4 EMA confirmed."


# --- Trade Execution and Management ---
def place_pending_order(symbol, order_type, price, sl, tp, lot_size, comment=""):
    props = ALL_SYMBOL_PROPERTIES[symbol]
    request = {"action": mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": lot_size, "type": order_type,
               "price": round(price, props['digits']), 
               "sl": round(sl, props['digits']), 
               "tp": round(tp, props['digits']),
               "magic": BOT_MAGIC_NUMBER,
               "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    
    logger.info(f"Attempting PENDING order: {request}")
    result = mt5.order_send(request)
    if result is None or (result.retcode != mt5.TRADE_RETCODE_DONE and result.retcode != mt5.TRADE_RETCODE_PLACED):
        logger.error(f"order_send failed, retcode={result.retcode if result else 'None'}, comment: {result.comment if result else mt5.last_error()}, request: {result.request if result else 'N/A'}")
        return None
    logger.info(f"Pending order PLACED: Ticket {result.order}, Symbol {symbol}, Type {order_type}, Price {price}, SL {sl}, TP {tp}, Lot {lot_size}")
    return result.order

def cancel_pending_order(ticket_id):
    request = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket_id, "magic": BOT_MAGIC_NUMBER}
    logger.info(f"Attempting CANCEL pending order ticket: {ticket_id}")
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"cancel_pending_order (ticket {ticket_id}) failed, retcode={result.retcode if result else 'None'}, comment: {result.comment if result else mt5.last_error()}")
        return False
    logger.info(f"Pending order ticket {ticket_id} CANCELED successfully.")
    return True

def manage_trailing_sl(position, symbol_props, m5_lookback_df_for_tsl):
    if not position or position.magic != BOT_MAGIC_NUMBER: return

    pos_id_str = str(position.identifier)
    details = trade_details_for_closure.get(pos_id_str)

    if not details or 'r_diff' not in details or 'next_trailing_r_level' not in details:
        logger.debug(f"R-level TSL details missing for pos {position.ticket} (ID: {pos_id_str}). Skipping progressive TSL. Details: {details}")
        return 

    r_diff = details['r_diff']
    current_next_r = details['next_trailing_r_level'] 
    max_r = details.get('max_trailing_r_level', 3.5) 
    r_step = details.get('r_step', 0.5)         

    pip_val_calc = symbol_props.get('pip_value_calc', 0.0)
    if pip_val_calc <= 1e-9: 
        logger.warning(f"Cannot TSL for {position.symbol}, pip_value_calc invalid: {pip_val_calc}. Skipping TSL."); return
    
    current_tick = mt5.symbol_info_tick(position.symbol)
    if not current_tick: 
        logger.warning(f"No tick for {position.symbol} for TSL. Skipping TSL."); return
    current_market_price_for_tsl = current_tick.bid if position.type == mt5.ORDER_TYPE_BUY else current_tick.ask

    attempt_sl_modification_due_to_r_hit = False

    if current_next_r <= max_r: 
        trigger_price = position.price_open + current_next_r * r_diff if position.type == mt5.ORDER_TYPE_BUY else position.price_open - current_next_r * r_diff
        
        if (position.type == mt5.ORDER_TYPE_BUY and current_market_price_for_tsl >= trigger_price) or \
           (position.type == mt5.ORDER_TYPE_SELL and current_market_price_for_tsl <= trigger_price):
            logger.info(f"R-target ({current_next_r:.1f}R) MET for pos {position.ticket} on {position.symbol}. Market: {current_market_price_for_tsl:.{symbol_props['digits']}f}, Trigger: {trigger_price:.{symbol_props['digits']}f}")
            attempt_sl_modification_due_to_r_hit = True

    is_sl_already_profitable_or_be = (position.type == mt5.ORDER_TYPE_BUY and position.sl >= position.price_open) or \
                                     (position.type == mt5.ORDER_TYPE_SELL and position.sl <= position.price_open)
    
    if attempt_sl_modification_due_to_r_hit or is_sl_already_profitable_or_be:
        if not is_sl_already_profitable_or_be and attempt_sl_modification_due_to_r_hit:
             logger.info(f"TSL activated for pos {position.ticket} due to R-level ({current_next_r:.1f}R) hit. Market price: {current_market_price_for_tsl:.{symbol_props['digits']}f}")
        elif is_sl_already_profitable_or_be and not attempt_sl_modification_due_to_r_hit: 
             logger.debug(f"TSL continues for pos {position.ticket} as SL is already at/beyond BE. Current SL: {position.sl}, Open: {position.price_open}")

        pip_adj_for_candle_sl = 3 * pip_val_calc 

        new_sl_from_candles = 0 
        if len(m5_lookback_df_for_tsl) >= 4: 
            last_3_closed_candles = m5_lookback_df_for_tsl.iloc[-4:-1] 
            if len(last_3_closed_candles) < 3: 
                logger.warning(f"Not enough M5 lookback candles for TSL ({len(last_3_closed_candles)} of 3 closed needed), skipping TSL adjustment for {position.ticket}")
                return

            sl_candidate_price = 0.0
            if position.type == mt5.ORDER_TYPE_BUY:
                sl_candidate_price = last_3_closed_candles['low'].min() - pip_adj_for_candle_sl
                if sl_candidate_price > position.price_open and sl_candidate_price > position.sl and sl_candidate_price < current_tick.bid : 
                     new_sl_from_candles = round(sl_candidate_price, symbol_props['digits'])
            elif position.type == mt5.ORDER_TYPE_SELL:
                sl_candidate_price = last_3_closed_candles['high'].max() + pip_adj_for_candle_sl
                if sl_candidate_price < position.price_open and sl_candidate_price < position.sl and sl_candidate_price > current_tick.ask:
                    new_sl_from_candles = round(sl_candidate_price, symbol_props['digits'])

            if new_sl_from_candles != 0 and abs(new_sl_from_candles - position.sl) > (symbol_props['point'] / 2): 
                request = {"action": mt5.TRADE_ACTION_SLTP, "position": position.ticket, "sl": new_sl_from_candles, "tp": position.tp, "magic": BOT_MAGIC_NUMBER}
                log_reason = f"R-level {current_next_r:.1f}R hit" if attempt_sl_modification_due_to_r_hit else "continuous trailing"
                logger.info(f"Attempting TRAIL SL for pos {position.ticket} from {position.sl} to {new_sl_from_candles} (Reason: {log_reason})")
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE: 
                    logger.info(f"SL for pos {position.ticket} TRAILED to {new_sl_from_candles}.")
                    details['current_sl'] = new_sl_from_candles 
                    
                    if attempt_sl_modification_due_to_r_hit and current_next_r <= max_r:
                        details['next_trailing_r_level'] += r_step
                        if details['next_trailing_r_level'] > max_r:
                            logger.info(f"Pos {position.ticket} has now effectively PASSED max_trailing_r_level ({max_r:.1f}R). Next R-level target is {details['next_trailing_r_level']:.1f} (R-progression stops).")
                        else:
                            logger.info(f"Pos {position.ticket} advanced to next R-level target: {details['next_trailing_r_level']:.1f}R")
                else: 
                    logger.error(f"Failed to trail SL for {position.ticket}. Ret: {result.retcode if result else 'N/A'}, MT5 Err: {mt5.last_error()}, Comm: {result.comment if result else 'N/A'}")
        else: 
            logger.warning(f"m5_lookback for TSL on {position.symbol} has {len(m5_lookback_df_for_tsl)} rows, need >=4 for TSL candle logic.")


def get_todays_realized_pnl_from_csv():
    pnl = 0.0
    if not os.path.exists(TRADE_HISTORY_FILE):
        return pnl
    try:
        df_all = pd.read_csv(TRADE_HISTORY_FILE, dtype={'PositionID': str, 'PNL_AccountCCY': str})
        # Filter out summary lines if they exist
        df = df_all[~df_all[CSV_HEADERS[0]].astype(str).str.contains("--- Performance Summary ---|Metric", na=False)].copy()

        if df.empty:
            return pnl
            
        df['PNL_AccountCCY'] = pd.to_numeric(df['PNL_AccountCCY'], errors='coerce')
        df['CloseTimeUTC_dt'] = pd.to_datetime(df['CloseTimeUTC'], errors='coerce', utc=True)
        today_utc_date = datetime.now(timezone.utc).date()
        
        today_closed_trades = df[
            (df['CloseTimeUTC_dt'].notna()) & \
            (df['CloseTimeUTC_dt'].dt.date == today_utc_date) & \
            (df['PNL_AccountCCY'].notna())
        ]
        pnl = today_closed_trades['PNL_AccountCCY'].sum()
        logger.info(f"Calculated today's realized PNL from CSV: {pnl:.2f}")
    except Exception as e:
        logger.error(f"Error calculating PNL from CSV {TRADE_HISTORY_FILE}: {e}")
    return pnl

# --- Helper function for periodic logging timestamp ---
def get_next_log_interval_timestamp(dt_object, interval_minutes=LOGGING_INTERVAL_MINUTES):
    floored_minute = (dt_object.minute // interval_minutes) * interval_minutes
    current_interval_start = dt_object.replace(minute=floored_minute, second=0, microsecond=0)
    return current_interval_start + timedelta(minutes=interval_minutes)

# --- Main Execution ---
if __name__ == "__main__":
    initialize_trade_history_file()
    load_state_from_csv() # Loads open positions, not summary

    if not initialize_mt5_interface(SYMBOLS_TO_TRADE):
        logger.error("Failed to initialize MT5 or critical symbols. Exiting.")
        exit()

    initial_balance_for_session_stats = mt5.account_info().balance # For overall summary
    balance_at_start_of_day_for_daily_risk = initial_balance_for_session_stats # For daily risk limit

    todays_realized_pnl = get_todays_realized_pnl_from_csv()
    current_day_for_risk_reset = datetime.now(timezone.utc).date()
    next_periodic_signal_log_time = get_next_log_interval_timestamp(datetime.now(timezone.utc))


    logger.info(f"Bot starting. Magic: {BOT_MAGIC_NUMBER}, Risk/Trade: {RISK_PER_TRADE_PERCENT*100:.2f}%, Daily Risk: {DAILY_RISK_LIMIT_PERCENT*100:.2f}%")
    logger.info(f"Initial Session Balance (for stats): {initial_balance_for_session_stats:.2f}, Current SOD Balance (for daily risk): {balance_at_start_of_day_for_daily_risk:.2f}, Initial Today's Realized PNL (from CSV): {todays_realized_pnl:.2f}")
    logger.info(f"Next periodic signal log scheduled around: {next_periodic_signal_log_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")


    try:
        while True:
            current_utc_time = datetime.now(timezone.utc)

            if current_utc_time >= next_periodic_signal_log_time:
                logger.info(f"--- Performing {LOGGING_INTERVAL_MINUTES}-Minute Periodic Signal Check at {current_utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
                for sym_to_log in SYMBOLS_AVAILABLE_FOR_TRADE:
                    props_log = ALL_SYMBOL_PROPERTIES.get(sym_to_log)
                    if not props_log:
                        logger.debug(f"Signal Log: Skipping {sym_to_log}, properties not found.")
                        continue
                    if not is_within_session(TRADING_SESSIONS_UTC.get(sym_to_log, [])):
                        continue
                    log_h1, log_m5, _, log_h4 = get_live_data_with_emas(
                        sym_to_log, H1_CANDLES_TO_FETCH, M5_CANDLES_TO_FETCH, H4_CANDLES_TO_FETCH
                    )
                    signal_type, signal_message = get_trade_signal_status(
                        sym_to_log, props_log, log_h1, log_m5, log_h4
                    )
                    if signal_type != "NONE": logger.info(f"Signal Log for {sym_to_log}: {signal_type} - {signal_message}")
                    else: logger.info(f"Signal Log for {sym_to_log}: NO SIGNAL - {signal_message}")
                next_periodic_signal_log_time = get_next_log_interval_timestamp(current_utc_time)
                logger.info(f"--- Periodic Signal Check Complete. Next log scheduled around: {next_periodic_signal_log_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

            if current_utc_time.date() != current_day_for_risk_reset:
                balance_at_start_of_day_for_daily_risk = mt5.account_info().balance # Update for daily risk
                todays_realized_pnl = get_todays_realized_pnl_from_csv() 
                current_day_for_risk_reset = current_utc_time.date()
                logger.info(f"New Day: {current_day_for_risk_reset}. SOD Balance (for daily risk): {balance_at_start_of_day_for_daily_risk:.2f}. Realized PNL (from CSV): {todays_realized_pnl:.2f}")

            current_account_balance = mt5.account_info().balance
            max_daily_loss_allowed = balance_at_start_of_day_for_daily_risk * DAILY_RISK_LIMIT_PERCENT

            open_positions = mt5.positions_get(magic=BOT_MAGIC_NUMBER)
            pending_orders = mt5.orders_get(magic=BOT_MAGIC_NUMBER)
            current_mt5_open_position_ids = set()

            if open_positions:
                for pos in open_positions:
                    pos_id_str = str(pos.identifier)
                    current_mt5_open_position_ids.add(pos_id_str)
                    if pos_id_str not in logged_open_position_ids:
                        symbol_props = ALL_SYMBOL_PROPERTIES.get(pos.symbol)
                        if not symbol_props: 
                            logger.error(f"Cannot log open pos {pos_id_str}, sym props not found for {pos.symbol}"); continue
                        
                        risked_amount = 0
                        if pos.sl > 0 and symbol_props.get('trade_tick_size', 0) > 0 and symbol_props.get('trade_tick_value', 0) > 0:
                            sl_diff_opened = abs(pos.price_open - pos.sl)
                            risked_amount = pos.volume * (sl_diff_opened / symbol_props['trade_tick_size']) * symbol_props['trade_tick_value']
                        
                        open_time_dt = datetime.fromtimestamp(pos.time, tz=timezone.utc)
                        trade_data = {
                            "TicketID": pos.ticket, "PositionID": pos_id_str, "Symbol": pos.symbol,
                            "Type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                            "OpenTimeUTC": open_time_dt.isoformat(), "EntryPrice": pos.price_open,
                            "LotSize": pos.volume, "SL_Price": pos.sl, "TP_Price": pos.tp,
                            "CloseTimeUTC": "", "ExitPrice": "", "PNL_AccountCCY": "",
                            "OpenComment": pos.comment if pos.comment else f"BotOpen:{BOT_MAGIC_NUMBER}",
                            "CloseReason": "", "RiskedAmount": round(risked_amount, 2) if risked_amount else ""
                        }
                        append_trade_to_csv(trade_data)
                        logged_open_position_ids.add(pos_id_str)

                        if pos.sl == 0: 
                            logger.error(f"Position {pos.ticket} (ID: {pos_id_str}) has SL=0. Cannot calculate r_diff for TSL. R-tracking NOT initialized.")
                            trade_details_for_closure[pos_id_str] = {
                                'original_sl': pos.sl, 'original_tp': pos.tp, 'symbol': pos.symbol,
                                'current_sl': pos.sl, 'current_tp': pos.tp
                            }
                        else:
                            risk_price_diff = abs(pos.price_open - pos.sl)
                            if risk_price_diff < symbol_props.get('point', 1e-5): 
                                logger.warning(f"Position {pos.ticket} (ID: {pos_id_str}) has very small r_diff ({risk_price_diff:.{symbol_props['digits']}f}). R-tracking initialized, but behavior might be rapid.")
                            
                            trade_details_for_closure[pos_id_str] = {
                                'original_sl': pos.sl,
                                'original_tp': pos.tp,
                                'symbol': pos.symbol,
                                'current_sl': pos.sl, 
                                'current_tp': pos.tp, 
                                'r_diff': risk_price_diff,
                                'next_trailing_r_level': 1.5,
                                'max_trailing_r_level': 3.5, 
                                'r_step': 0.5
                            }
                            logger.info(f"Initialized R-tracking for pos {pos.ticket}: r_diff={risk_price_diff:.{symbol_props['digits']}f}, next_R=1.5")

                    elif pos_id_str in trade_details_for_closure: 
                         trade_details_for_closure[pos_id_str]['current_sl'] = pos.sl
                         trade_details_for_closure[pos_id_str]['current_tp'] = pos.tp

                    props_active_pos = ALL_SYMBOL_PROPERTIES.get(pos.symbol)
                    if props_active_pos:
                        _, _, m5_lookback_active, _ = get_live_data_with_emas(pos.symbol, 0, M5_CANDLES_TO_FETCH, 0)
                        if m5_lookback_active is not None and not m5_lookback_active.empty:
                            manage_trailing_sl(pos, props_active_pos, m5_lookback_active)

            positions_closed_since_last_check = logged_open_position_ids - current_mt5_open_position_ids
            if positions_closed_since_last_check:
                logger.info(f"Positions closed or no longer in MT5: {positions_closed_since_last_check}")
                deals_from_time = datetime.now(timezone.utc) - timedelta(days=7)
                deals_to_time = datetime.now(timezone.utc)
                all_recent_deals = mt5.history_deals_get(int(deals_from_time.timestamp()), int(deals_to_time.timestamp()))
                temp_closed_ids_to_remove = set()
                if all_recent_deals:
                    sorted_deals = sorted(all_recent_deals, key=lambda d: d.time, reverse=True)
                    for pos_id_closed_str in positions_closed_since_last_check:
                        closing_deal = None
                        for deal in sorted_deals:
                            if str(deal.position_id) == pos_id_closed_str and deal.entry == mt5.DEAL_ENTRY_OUT and deal.magic == BOT_MAGIC_NUMBER:
                                closing_deal = deal; break
                        if closing_deal:
                            logger.info(f"Found closing deal for position {pos_id_closed_str}: Deal Ticket {closing_deal.ticket}, Magic: {closing_deal.magic}, Comment: '{closing_deal.comment}'")
                            close_reason = closing_deal.comment if closing_deal.comment else "Closed"
                            deal_comment_lower = (closing_deal.comment or "").lower()
                            if "sl" in deal_comment_lower : close_reason = "Stop Loss Hit"
                            elif "tp" in deal_comment_lower: close_reason = "Take Profit Hit"
                            
                            close_time_dt = datetime.fromtimestamp(closing_deal.time, tz=timezone.utc)
                            update_data = {
                                "CloseTimeUTC": close_time_dt.isoformat(), "ExitPrice": closing_deal.price,
                                "PNL_AccountCCY": closing_deal.profit, "CloseReason": close_reason
                            }
                            update_closed_trade_in_csv(pos_id_closed_str, update_data)
                            temp_closed_ids_to_remove.add(pos_id_closed_str)
                            if pos_id_closed_str in trade_details_for_closure: 
                                del trade_details_for_closure[pos_id_closed_str] 
                            if close_time_dt.date() == current_day_for_risk_reset:
                                todays_realized_pnl += closing_deal.profit
                                logger.info(f"Updated in-session PNL by {closing_deal.profit:.2f}. New total: {todays_realized_pnl:.2f}")
                        else:
                            logger.warning(f"Could not find BOT closing deal for PosID {pos_id_closed_str}. It might be manual or error in logging. Will mark as closed in memory.")
                            temp_closed_ids_to_remove.add(pos_id_closed_str) 
                            if pos_id_closed_str in trade_details_for_closure:
                                del trade_details_for_closure[pos_id_closed_str]
                logged_open_position_ids -= temp_closed_ids_to_remove


            num_bot_positions = len(current_mt5_open_position_ids)
            num_bot_pending_orders = len(pending_orders) if pending_orders else 0

            if num_bot_positions > 0:
                time.sleep(LOOP_SLEEP_SECONDS)
                continue

            if num_bot_pending_orders > 0:
                for order in pending_orders:
                    props_pending = ALL_SYMBOL_PROPERTIES.get(order.symbol)
                    if not props_pending: continue
                    _, latest_m5_data_order, _, _ = get_live_data_with_emas(order.symbol, 0, M5_CANDLES_TO_FETCH, 0)
                    if latest_m5_data_order is None: logger.warning(f"No M5 data for pending order invalidation on {order.symbol}."); continue
                    if 'M5_EMA21' not in latest_m5_data_order or pd.isna(latest_m5_data_order['M5_EMA21']) or \
                       'close' not in latest_m5_data_order or pd.isna(latest_m5_data_order['close']):
                        logger.warning(f"M5_EMA21 or close missing in M5 data for {order.symbol}. Skipping invalidation."); continue
                    m5_ema21_inv = latest_m5_data_order['M5_EMA21']; current_m5_close_inv = latest_m5_data_order['close']
                    setup_bias_inv = "BUY" if order.type == mt5.ORDER_TYPE_BUY_STOP else "SELL"
                    invalidated = (setup_bias_inv == "BUY" and current_m5_close_inv < m5_ema21_inv) or \
                                  (setup_bias_inv == "SELL" and current_m5_close_inv > m5_ema21_inv)
                    if invalidated:
                        logger.info(f"Pending {setup_bias_inv} order {order.ticket} on {order.symbol} invalidated by M5 close ({current_m5_close_inv:.{props_pending['digits']}f}) vs EMA21 ({m5_ema21_inv:.{props_pending['digits']}f}).")
                        cancel_pending_order(order.ticket)
                time.sleep(LOOP_SLEEP_SECONDS) 
                continue

            if num_bot_positions == 0 and num_bot_pending_orders == 0:
                logger.debug(f"Scanning for new setups. Balance: {current_account_balance:.2f}, Today's PNL: {todays_realized_pnl:.2f}")
                current_day_loss = abs(todays_realized_pnl) if todays_realized_pnl < 0 else 0
                if current_day_loss >= max_daily_loss_allowed:
                    logger.info(f"Daily loss limit hit ({todays_realized_pnl:.2f} vs -{max_daily_loss_allowed:.2f}). No new trades today.")
                else:
                    for sym_to_check_setup in SYMBOLS_AVAILABLE_FOR_TRADE:
                        props_setup = ALL_SYMBOL_PROPERTIES.get(sym_to_check_setup)
                        if not props_setup: continue
                        if not is_within_session(TRADING_SESSIONS_UTC.get(sym_to_check_setup, [])): continue
                        
                        current_spread_pips_val = get_current_spread_pips(sym_to_check_setup, props_setup)
                        max_allowed_spread_for_symbol = MAX_SPREAD_PIPS_MAP.get(sym_to_check_setup, MAX_SPREAD_PIPS_DEFAULT)
                        if current_spread_pips_val > max_allowed_spread_for_symbol:
                            logger.info(f"Spread {sym_to_check_setup} ({current_spread_pips_val:.2f}) > max ({max_allowed_spread_for_symbol:.2f}). Skip.")
                            continue

                        latest_h1, latest_m5, m5_lookback_df_setup, latest_h4 = get_live_data_with_emas(
                            sym_to_check_setup, H1_CANDLES_TO_FETCH, M5_CANDLES_TO_FETCH, H4_CANDLES_TO_FETCH
                        )
                        
                        if latest_h1 is None or latest_m5 is None or m5_lookback_df_setup is None or m5_lookback_df_setup.empty or latest_h4 is None:
                            logger.debug(f"Trade Scan: Incomplete candle data for {sym_to_check_setup}. Skip.")
                            continue

                        signal_type_trade, signal_message_trade = get_trade_signal_status(
                            sym_to_check_setup, props_setup, latest_h1, latest_m5, latest_h4
                        )

                        if signal_type_trade != "NONE":
                            logger.info(f"Potential Trade Setup: {sym_to_check_setup} - {signal_type_trade} based on: {signal_message_trade}")
                            
                            if 'ATR' not in latest_m5 or pd.isna(latest_m5['ATR']) or latest_m5['ATR'] <= 0:
                                logger.warning(f"ATR not available or invalid for {sym_to_check_setup} on M5. ATR: {latest_m5.get('ATR', 'N/A')}. Skipping trade setup.")
                                continue
                            atr_val = latest_m5['ATR']
                            
                            sl_atr_multiplier = 1.5
                            tp_sl_ratio = 4.0 

                            sl_distance = sl_atr_multiplier * atr_val
                            tp_distance = tp_sl_ratio * sl_distance

                            pip_val_calc_setup = props_setup.get('pip_value_calc', 0.0)
                            if pip_val_calc_setup <= 1e-9: 
                                logger.warning(f"Invalid pip_value_calc for entry setup {sym_to_check_setup}"); continue
                            pip_adj_setup = 1 * pip_val_calc_setup 

                            entry_px, sl_px, tp_px = 0, 0, 0
                            order_type_mt5 = None
                            
                            if len(m5_lookback_df_setup) < 1: 
                                logger.warning(f"Not enough M5 lookback candles ({len(m5_lookback_df_setup)}) for entry price calc on {sym_to_check_setup}. Skipping.")
                                continue


                            if signal_type_trade == "BUY":
                                entry_px_candidate = m5_lookback_df_setup['high'].max() 
                                entry_px = entry_px_candidate + pip_adj_setup
                                sl_px = entry_px - sl_distance
                                tp_px = entry_px + tp_distance
                                order_type_mt5 = mt5.ORDER_TYPE_BUY_STOP
                            elif signal_type_trade == "SELL":
                                entry_px_candidate = m5_lookback_df_setup['low'].min() 
                                entry_px = entry_px_candidate - pip_adj_setup
                                sl_px = entry_px + sl_distance
                                tp_px = entry_px - tp_distance
                                order_type_mt5 = mt5.ORDER_TYPE_SELL_STOP
                            
                            entry_px = round(entry_px, props_setup['digits'])
                            sl_px = round(sl_px, props_setup['digits'])
                            tp_px = round(tp_px, props_setup['digits'])
                            
                            sl_diff_for_lot_calc = abs(entry_px - sl_px)
                            min_sl_dist_points = props_setup.get('trade_tick_size', props_setup.get('point', 0.00001)) * 2 

                            if (signal_type_trade == "BUY" and sl_px >= entry_px) or \
                               (signal_type_trade == "SELL" and sl_px <= entry_px) or \
                               sl_diff_for_lot_calc < min_sl_dist_points:
                                logger.info(f"Invalid SL/Entry for {sym_to_check_setup}. Entry: {entry_px}, SL: {sl_px}, TP: {tp_px}, SL_Diff: {sl_diff_for_lot_calc:.{props_setup['digits']}f}, MinSLDist: {min_sl_dist_points:.{props_setup['digits']}f}, ATR: {atr_val:.{props_setup['digits']}f}. Skip.")
                                continue
                            
                            current_tick_info = mt5.symbol_info_tick(sym_to_check_setup)
                            if not current_tick_info:
                                logger.warning(f"Could not get current tick for {sym_to_check_setup} to validate entry price. Skipping.")
                                continue

                            if signal_type_trade == "BUY" and entry_px <= current_tick_info.ask:
                                logger.info(f"BUY STOP for {sym_to_check_setup}: Entry {entry_px} is at or below current Ask {current_tick_info.ask}. Skip.")
                                continue 
                            if signal_type_trade == "SELL" and entry_px >= current_tick_info.bid:
                                logger.info(f"SELL STOP for {sym_to_check_setup}: Entry {entry_px} is at or above current Bid {current_tick_info.bid}. Skip.")
                                continue

                            potential_loss_this_trade = current_account_balance * RISK_PER_TRADE_PERCENT
                            if (current_day_loss + potential_loss_this_trade) > max_daily_loss_allowed:
                                logger.info(f"Daily risk limit would be breached by new trade for {sym_to_check_setup}. Current Loss: {current_day_loss:.2f}, Pot.Loss: {potential_loss_this_trade:.2f}, Daily Limit: {max_daily_loss_allowed:.2f}. Skip.")
                                continue

                            calc_lot = calculate_lot_size(current_account_balance, RISK_PER_TRADE_PERCENT, sl_diff_for_lot_calc, props_setup)
                            if calc_lot <= 1e-9:
                                logger.info(f"Lot size zero for {sym_to_check_setup}. SL diff: {sl_diff_for_lot_calc}. ATR: {atr_val}. Skip.")
                                continue

                            order_comment = f"VBotATR;{signal_type_trade};{sym_to_check_setup};H4C"
                            order_ticket = place_pending_order(sym_to_check_setup, order_type_mt5, entry_px, sl_px, tp_px, calc_lot, comment=order_comment)
                            if order_ticket:
                                logger.info(f"Placed PENDING for {sym_to_check_setup}. Ticket: {order_ticket}. Entry: {entry_px}, SL: {sl_px}, TP: {tp_px}, Lot: {calc_lot}, ATR: {atr_val:.{props_setup['digits']}f}")
                                break 
            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("Bot stopping via KeyboardInterrupt.")
    except Exception as e:
        logger.exception(f"Unhandled exception in main loop: {e}")
    finally:
        logger.info("Shutting down bot...")
        open_positions_final = mt5.positions_get(magic=BOT_MAGIC_NUMBER)
        if open_positions_final is not None:
            current_mt5_open_ids_final = {str(p.identifier) for p in open_positions_final}
            final_closed_ids = logged_open_position_ids - current_mt5_open_ids_final
            if final_closed_ids:
                logger.info(f"Processing {len(final_closed_ids)} positions found closed during shutdown sequence.")
                deals_from_time_final = datetime.now(timezone.utc) - timedelta(days=7)
                deals_to_time_final = datetime.now(timezone.utc)
                all_recent_deals_final = mt5.history_deals_get(int(deals_from_time_final.timestamp()), int(deals_to_time_final.timestamp()))
                if all_recent_deals_final:
                    sorted_deals_final = sorted(all_recent_deals_final, key=lambda d: d.time, reverse=True)
                    for pos_id_final_str in final_closed_ids:
                        try: 
                            # Check if already updated in CSV before trying again
                            df_check_shutdown = pd.read_csv(TRADE_HISTORY_FILE, dtype={'PositionID': str})
                             # Filter out summary lines if they exist
                            df_check_shutdown = df_check_shutdown[~df_check_shutdown[CSV_HEADERS[0]].astype(str).str.contains("--- Performance Summary ---|Metric", na=False)]
                            
                            trade_row_shutdown = df_check_shutdown[df_check_shutdown['PositionID'] == pos_id_final_str]
                            if not trade_row_shutdown.empty and \
                               (pd.notna(trade_row_shutdown['CloseTimeUTC'].iloc[0]) and trade_row_shutdown['CloseTimeUTC'].iloc[0] != ''):
                                logger.info(f"Position {pos_id_final_str} already logged as closed in CSV. Skipping update at shutdown.")
                                continue 
                        except Exception as csv_read_err:
                            logger.error(f"Error reading CSV during shutdown check for {pos_id_final_str}: {csv_read_err}")

                        closing_deal_final = None
                        for deal_final in sorted_deals_final:
                            if str(deal_final.position_id) == pos_id_final_str and deal_final.entry == mt5.DEAL_ENTRY_OUT and deal_final.magic == BOT_MAGIC_NUMBER:
                                closing_deal_final = deal_final; break
                        if closing_deal_final:
                            close_reason_final = closing_deal_final.comment if closing_deal_final.comment else "Closed at shutdown"
                            close_time_dt_final = datetime.fromtimestamp(closing_deal_final.time, tz=timezone.utc)
                            update_data_final = {"CloseTimeUTC": close_time_dt_final.isoformat(), "ExitPrice": closing_deal_final.price,
                                                 "PNL_AccountCCY": closing_deal_final.profit, "CloseReason": close_reason_final}
                            update_closed_trade_in_csv(pos_id_final_str, update_data_final)
                        else:
                             logger.warning(f"Could not find closing deal for PosID {pos_id_final_str} during shutdown. CSV may remain incomplete for this trade.")


        pending_orders_final = mt5.orders_get(magic=BOT_MAGIC_NUMBER)
        if pending_orders_final:
            logger.info(f"Found {len(pending_orders_final)} pending orders. Cancelling...")
            for order in pending_orders_final: cancel_pending_order(order.ticket)
        
        # Calculate and append performance summary before shutting down MT5
        calculate_and_append_performance_summary(TRADE_HISTORY_FILE, initial_balance_for_session_stats)
        
        shutdown_mt5_interface()
        logger.info("Bot has shut down.")

# --- END OF FILE bookStrategyLive.py ---