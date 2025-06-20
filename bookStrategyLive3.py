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

SYMBOLS_TO_TRADE = ["EURUSD", "USDCHF", "GBPJPY", "GBPUSD",
"AUDJPY",  "XAUUSD", "USOIL",
"BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD"]

TRADING_SESSIONS_UTC = { # (start_hour_inclusive, end_hour_exclusive)
"EURUSD": [(7, 14)], "GBPUSD": [(7, 14)], "AUDUSD": [ (7, 14)],
"USDCHF": [(7, 14)], "USDCAD": [(12, 14)], "USDJPY": [ (12, 14)],
"EURJPY": [ (7, 12)], "GBPJPY": [(7, 14)], "NZDUSD": [ (7, 14)],
"EURCHF": [(7, 14)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 14)],
"EURNZD": [ (7, 14)], "GBPNZD": [(7, 14)], "XAUUSD": [(7, 14)],
"XAGUSD": [(7, 14)], "XPTUSD": [(7, 14)], "XAGGBP":[(7, 14)], "XAGEUR":[(7,14)], "XAGAUD": [(0,4), (7,10)], "BTCXAG":[(7,14)]
}
TRADING_SESSIONS_UTC["USOIL"] = [(12, 14)]
TRADING_SESSIONS_UTC["UKOIL"] = [(7, 14)]
CRYPTO_SESSIONS_USER = {"BTCUSD":[(7, 14)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 14)], "ETHUSD":[(7, 14)]}
for crypto_sym, sess_val in CRYPTO_SESSIONS_USER.items():
    TRADING_SESSIONS_UTC[crypto_sym] = sess_val

RISK_PER_TRADE_PERCENT = 0.01  # Risk 1% of current balance per trade
DAILY_RISK_LIMIT_PERCENT = 0.05 # Daily risk limit of 5% of balance at start of day

# --- Commission Structure ---

COMMISSIONS = {
"EURUSD": 0.07, "AUDUSD": 0.10, "USDCHF": 0.10, "USDCAD": 0.10,
"NZDUSD": 0.13, "AUDJPY": 0.09, "EURNZD": 0.18, "USOIL": 0.16,
"UKOIL": 0.65, "BTCUSD": 0.16, "BTCJPY": 0.21, "BTCXAU": 0.20,
"ETHUSD": 0.30, "GBPUSD": 0.09, "USDJPY": 0.07, "GBPJPY": 0.15,
}

# --- News Filter Times (User Input) ---

NEWS_TIMES_UTC = {
"USDCHF": [],  "USDCAD": [], "NZDUSD": [],
"ETHUSD": [], "BTCUSD": [], "EURUSD": [],
"AUDJPY": ["6:40"], "GBPUSD": ["6:00"], "USDJPY": [],
"USOIL": [], "XAUUSD":[], "GBPJPY":["6:00","6:40"], "BTCJPY":["6:40"]
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
        
        # --- FIX: Filter for rows where PositionID is a valid number string ---
        # This robustly removes summary lines and other non-trade data which caused the error.
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
                    # Find the correct row: matching PositionID and not yet closed
                    if row[CSV_HEADERS.index('PositionID')] == position_id_to_update_str and \
                       (not row[CSV_HEADERS.index('CloseTimeUTC')] or row[CSV_HEADERS.index('CloseTimeUTC')] == ''):
                        # Update the values in the row
                        for key, value in update_values_dict.items():
                            if key in CSV_HEADERS:
                                row[CSV_HEADERS.index(key)] = value
                        updated = True
                lines.append(row) # Append every row, modified or not

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
        
        # Filter out any non-trade rows (e.g., previous summaries)
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
            time.sleep(0.5); symbol_info_obj = mt5.symbol_info(symbol_name) # Re-fetch
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

def fetch_latest_data(symbol):
    # Fetch enough data to ensure indicators are stable
    df_h4 = get_live_data(symbol, mt5.TIMEFRAME_H4, 100)
    df_h1 = get_live_data(symbol, mt5.TIMEFRAME_H1, 100)
    df_m5 = get_live_data(symbol, mt5.TIMEFRAME_M5, 100)

    if df_h4.empty or df_h1.empty or df_m5.empty:
        logger.warning(f"Could not fetch complete data for {symbol}. Skipping analysis.")
        return None

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
    if combined_df.empty: return None

    return combined_df, combined_df.iloc[-1] # Return full df for lookbacks and last row for signals

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
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot_size,
        "type": trade_type,
        "price": entry_price,
        "sl": sl_price,
        "tp": 0.0, # TP is calculated and set after entry
        "magic": 202405,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        logger.error(f"[{symbol}] Order Send FAILED. No result object. Error: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"[{symbol}] Order Send FAILED. Retcode: {result.retcode}, Comment: {result.comment}")
        return None
        
    logger.info(f"[{symbol}] PENDING ORDER PLACED. Ticket: {result.order}, Type: {order_type}, Price: {entry_price}, Lot: {lot_size}")
    return result.order

def modify_position_sltp(position, new_sl, new_tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol": position.symbol,
        "sl": new_sl,
        "tp": new_tp,
        "magic": 202405,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[{position.symbol}] Position {position.ticket} MODIFIED. New SL: {new_sl}, New TP: {new_tp}")
        return True
    else:
        logger.error(f"[{position.symbol}] Position {position.ticket} MODIFY FAILED. Retcode: {result.retcode if result else 'N/A'}, Error: {mt5.last_error()}")
        return False

def cancel_pending_order(order_ticket):
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": order_ticket,
        "magic": 202405,
    }
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
        
        # Give MT5 a moment to log the deal
        time.sleep(2)
        deals = mt5.history_deals_get(position=int(pos_id))
        if not deals:
            logger.warning(f"Could not find deals for closed position {pos_id}. Will retry.")
            continue
            
        deals_df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        entry_deal = deals_df[deals_df['entry'] == mt5.DEAL_ENTRY_IN].iloc[0]
        exit_deal = deals_df[deals_df['entry'] == mt5.DEAL_ENTRY_OUT].iloc[-1]
        
        commission_cost = COMMISSIONS.get(exit_deal.symbol, 0.0)
        net_pnl = exit_deal.profit + exit_deal.commission + exit_deal.swap - commission_cost
        
        update_values = {
            'CloseTimeUTC': pd.to_datetime(exit_deal.time, unit='s', utc=True).isoformat(),
            'ExitPrice': exit_deal.price,
            'PNL_AccountCCY': f"{net_pnl:.2f}",
            'CloseReason': f"Closed by broker: {exit_deal.comment}"
        }
        update_closed_trade_in_csv(pos_id, update_values)

        # Clean up state
        logged_open_position_ids.remove(pos_id)
        if pos_id in trade_details_for_closure:
            del trade_details_for_closure[pos_id]

def manage_open_positions():
    open_positions = mt5.positions_get()
    if not open_positions: return

    for position in open_positions:
        if position.magic != 202405: continue
        
        pos_id_str = str(position.ticket)
        # First-time management for a newly opened position
        if pos_id_str not in logged_open_position_ids:
            risk_val_diff = abs(position.price_open - position.sl)
            tp_price = position.price_open + (4 * risk_val_diff) if position.type == mt5.ORDER_TYPE_BUY else position.price_open - (4 * risk_val_diff)
            
            props = ALL_SYMBOL_PROPERTIES[position.symbol]
            tp_price = round(tp_price, props['digits'])

            # Set the 4R TP
            modify_position_sltp(position, position.sl, tp_price)

            # Log to CSV
            risk_amount = 0
            if props['trade_tick_size'] > 0:
                risk_amount = (risk_val_diff / props['trade_tick_size']) * props['trade_tick_value'] * position.volume

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
            trade_details_for_closure[pos_id_str] = {'symbol': position.symbol, 'original_sl': position.sl, 'current_sl': position.sl, 'trailing_active': False, 'ts_next_atr_level': 1.5}
            continue

        # Ongoing management (Trailing Stop)
        details = trade_details_for_closure.get(pos_id_str)
        if not details: continue

        data = fetch_latest_data(position.symbol)
        if data is None: continue
        df, current_candle = data
        
        atr_val = current_candle.get('ATR', np.nan)
        if pd.isna(atr_val) or atr_val <= 0: continue
            
        move_from_entry = (current_candle['high'] - position.price_open) if position.type == mt5.ORDER_TYPE_BUY else (position.price_open - current_candle['low'])
        atr_movement = move_from_entry / atr_val

        if atr_movement >= details['ts_next_atr_level']:
            props = ALL_SYMBOL_PROPERTIES[position.symbol]
            last_3 = df.iloc[-3:]
            new_sl = 0

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = last_3['low'].min() - 2 * props['pip_value_calc']
                if new_sl > details['current_sl']:
                    logger.info(f"[{position.symbol}] Trailing SL for BUY {position.ticket}. New SL: {new_sl:.{props['digits']}f}")
                    if modify_position_sltp(position, round(new_sl, props['digits']), position.tp):
                        details['current_sl'] = new_sl
            else: # SELL
                new_sl = last_3['high'].max() + 2 * props['pip_value_calc']
                if new_sl < details['current_sl']:
                    logger.info(f"[{position.symbol}] Trailing SL for SELL {position.ticket}. New SL: {new_sl:.{props['digits']}f}")
                    if modify_position_sltp(position, round(new_sl, props['digits']), position.tp):
                        details['current_sl'] = new_sl

            details['ts_next_atr_level'] += 0.5

def manage_pending_orders():
    pending_orders = mt5.orders_get()
    if not pending_orders: return

    for order in pending_orders:
        if order.magic != 202405: continue
        
        data = fetch_latest_data(order.symbol)
        if data is None: continue
        _, current_candle = data

        setup_bias = "BUY" if order.type == mt5.ORDER_TYPE_BUY_STOP else "SELL"
        m5_ema21 = current_candle['M5_EMA21']

        if (setup_bias == "BUY" and current_candle['close'] < m5_ema21) or \
           (setup_bias == "SELL" and current_candle['close'] > m5_ema21):
            logger.info(f"[{order.symbol}] PENDING order {order.ticket} invalidated (Close vs M5_EMA21). Cancelling...")
            cancel_pending_order(order.ticket)

def check_for_new_signals(daily_risk_allocated, max_daily_risk):
    global delayed_setups_queue

    # Process the queue first
    new_queue = []
    order_placed_this_cycle = False
    for setup in delayed_setups_queue:
        setup['confirm_count'] += 1
        if setup['confirm_count'] < 2:
            new_queue.append(setup)
            continue
        
        # Re-check conditions for confirmation
        data = fetch_latest_data(setup['symbol'])
        if data is None: 
            new_queue.append(setup) # Keep in queue if data fails
            continue
        _, current_candle = data

        if (setup['bias'] == "BUY" and current_candle['close'] < current_candle['M5_EMA21']) or \
           (setup['bias'] == "SELL" and current_candle['close'] > current_candle['M5_EMA21']):
            logger.info(f"[{setup['symbol']}] Delayed {setup['bias']} setup invalidated on confirmation. Discarding.")
            continue # Discard invalid setup
            
        if daily_risk_allocated + setup["risk_amt"] > max_daily_risk:
            logger.warning(f"[{setup['symbol']}] Delayed setup confirmed, but would exceed daily risk limit. Discarding.")
            continue

        # Place the order
        props = ALL_SYMBOL_PROPERTIES[setup['symbol']]
        order_ticket = place_pending_order(setup['symbol'], props, f"{setup['bias']}_STOP", setup['entry_price'], setup['sl_price'], setup['lot_size'], "LiveBot_v1_Delayed")
        
        if order_ticket:
            daily_risk_allocated += setup["risk_amt"]
            order_placed_this_cycle = True
            # Since an order is now pending, stop processing more setups from the queue
            break 
        else:
            # If placing fails, keep it in the queue for another try
            new_queue.append(setup)
            
    delayed_setups_queue = new_queue

    if order_placed_this_cycle: return daily_risk_allocated # Return updated risk

    # Look for new setups only if no order was placed this cycle
    for symbol in SYMBOLS_AVAILABLE_FOR_TRADE:
        # Check if we can trade this symbol now
        if not is_within_session(TRADING_SESSIONS_UTC.get(symbol, [])): continue
        if not is_outside_news_blackout(symbol, NEWS_TIMES_UTC): continue
        
        data = fetch_latest_data(symbol)
        if data is None: continue
        df, current_candle = data

        props = ALL_SYMBOL_PROPERTIES[symbol]
        pip_adj = 3 * props['trade_tick_size']

        # H1 Bias
        if not all(k in current_candle for k in ['H1_EMA8', 'H1_EMA21', 'H1_Close_For_Bias']) or pd.isna(current_candle[['H1_EMA8', 'H1_EMA21', 'H1_Close_For_Bias']]).any(): continue
        h1_trend_bias = "BUY" if current_candle['H1_EMA8'] > current_candle['H1_EMA21'] and current_candle['H1_Close_For_Bias'] > current_candle['H1_EMA8'] else "SELL" if current_candle['H1_EMA8'] < current_candle['H1_EMA21'] and current_candle['H1_Close_For_Bias'] < current_candle['H1_EMA8'] else None
        if not h1_trend_bias: continue

        # M5 Fanning
        if not all(k in current_candle for k in ['M5_EMA8', 'M5_EMA13', 'M5_EMA21']) or pd.isna(current_candle[['M5_EMA8', 'M5_EMA13', 'M5_EMA21']]).any(): continue
        m5_fanned_buy = current_candle['M5_EMA8'] > current_candle['M5_EMA13'] or current_candle['M5_EMA8'] > current_candle['M5_EMA21']
        m5_fanned_sell = current_candle['M5_EMA8'] < current_candle['M5_EMA13'] or current_candle['M5_EMA8'] < current_candle['M5_EMA21']
        if not ((h1_trend_bias == "BUY" and m5_fanned_buy) or (h1_trend_bias == "SELL" and m5_fanned_sell)): continue
        
        # H4 Confirmation
        if not all(k in current_candle for k in ['H4_EMA8', 'H4_EMA21']) or pd.isna(current_candle[['H4_EMA8', 'H4_EMA21']]).any(): continue
        if (h1_trend_bias == "BUY" and current_candle['H4_EMA8'] < current_candle['H4_EMA21']) or \
           (h1_trend_bias == "SELL" and current_candle['H4_EMA8'] > current_candle['H4_EMA21']): continue

        # RSI Filter
        if not all(k in current_candle for k in ['RSI_M5', 'RSI_H1']) or pd.isna(current_candle[['RSI_M5', 'RSI_H1']]).any(): continue
        if (h1_trend_bias == "BUY" and not (current_candle['RSI_M5'] > 50 and current_candle['RSI_H1'] > 50)) or \
           (h1_trend_bias == "SELL" and not (current_candle['RSI_M5'] < 50 and current_candle['RSI_H1'] < 50)): continue

        # Pullback and Price vs EMA21
        if (h1_trend_bias == "BUY" and current_candle['close'] < current_candle['M5_EMA21']) or \
           (h1_trend_bias == "SELL" and current_candle['close'] > current_candle['M5_EMA21']): continue
        if not ((h1_trend_bias == "BUY" and current_candle['low'] <= current_candle['M5_EMA8']) or \
                (h1_trend_bias == "SELL" and current_candle['high'] >= current_candle['M5_EMA8'])): continue

        # Weakness Filter
        recent_candles = df.iloc[-5:-1] # 4 candles before the current one
        if len(recent_candles) < 4: continue
        bullish_count = (recent_candles['close'] > recent_candles['open']).sum()
        bearish_count = (recent_candles['close'] < recent_candles['open']).sum()
        if (h1_trend_bias == "BUY" and bullish_count > 2) or (h1_trend_bias == "SELL" and bearish_count > 2): continue

        # Pullback Depth Filter
        lookback_window = df.iloc[-11:-1]
        swing_high, swing_low = lookback_window['high'].max(), lookback_window['low'].min()
        impulse_start, impulse_end, price_for_pb = (swing_low, swing_high, current_candle['low']) if h1_trend_bias == "BUY" else (swing_high, swing_low, current_candle['high'])
        if calculate_pullback_depth(impulse_start, impulse_end, price_for_pb, h1_trend_bias) < 0.30: continue
        
        # EMA-Fib Confluence
        fib_levels = calculate_fib_levels(swing_high, swing_low)
        tolerance = 0.5 * current_candle['ATR']
        if not any(abs(current_candle['M5_EMA8'] - fib_price) <= tolerance or abs(current_candle['M5_EMA13'] - fib_price) <= tolerance for fib_price in fib_levels.values()): continue

        # Entry & SL
        entry_lookback = df.iloc[-3:]
        entry_px, sl_px = (0,0)
        if h1_trend_bias == "BUY":
            entry_px = entry_lookback['high'].max() + pip_adj
            sl_px = entry_px - (1.5 * current_candle['ATR'])
        else:
            entry_px = entry_lookback['low'].min() - pip_adj
            sl_px = entry_px + (1.5 * current_candle['ATR'])
        
        entry_px, sl_px = round(entry_px, props['digits']), round(sl_px, props['digits'])
        if abs(entry_px - sl_px) <= 0: continue
            
        # Risk Check
        lot_size = props['volume_min']
        est_risk = lot_size * (abs(entry_px - sl_px) / props['trade_tick_size']) * props['trade_tick_value']
        if est_risk > mt5.account_info().balance * RISK_PER_TRADE_PERCENT:
            logger.info(f"[{symbol}] Setup found but min lot risk ({est_risk:.2f}) exceeds max allowed. Skipping.")
            continue
            
        # Add to Delayed Queue
        delayed_setups_queue.append({
            "symbol": symbol, "bias": h1_trend_bias, "entry_price": entry_px, "sl_price": sl_px,
            "lot_size": lot_size, "risk_amt": est_risk, "confirm_count": 0
        })
        logger.info(f"[{symbol}] SETUP QUEUED for delayed confirmation. Bias: {h1_trend_bias}, Entry: {entry_px}")
        break # One setup queued per cycle

    return daily_risk_allocated

# --- Main Execution ---

if __name__ == "__main__":
    if not initialize_mt5_interface(SYMBOLS_TO_TRADE):
        logger.error("Failed to initialize. Exiting.")
        exit()

    initialize_trade_history_file()
    load_state_from_csv()

    # Daily state variables
    current_day = datetime.now(timezone.utc).date()
    daily_risk_allocated_today = 0.0
    max_daily_risk_budget = mt5.account_info().balance * DAILY_RISK_LIMIT_PERCENT
    consecutive_losses_count = 0

    logger.info("--- Live Trading Bot Started ---")
    logger.info(f"Initial daily risk budget: {max_daily_risk_budget:.2f} USD")

    try:
        while True:
            # Daily Reset Logic
            if datetime.now(timezone.utc).date() != current_day:
                current_day = datetime.now(timezone.utc).date()
                daily_risk_allocated_today = 0.0
                max_daily_risk_budget = mt5.account_info().balance * DAILY_RISK_LIMIT_PERCENT
                consecutive_losses_count = 0
                logger.info(f"NEW DAY: {current_day}. Daily risk budget reset to {max_daily_risk_budget:.2f} USD.")

            # 1. Check for closed trades and update CSV
            manage_closed_positions()

            # 2. Manage currently open positions (newly opened and trailing stops)
            manage_open_positions()

            # 3. Manage pending orders (check for invalidation)
            manage_pending_orders()

            # 4. Check for new signals if no positions are open/pending and daily limits allow
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
                logger.debug(f"Trade management cycle. Open: {len(open_positions) if open_positions else 0}, Pending: {len(pending_orders) if pending_orders else 0}")

            logger.info("Cycle complete. Waiting for 60 seconds...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting gracefully...")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("Performing final shutdown tasks.")
        calculate_and_append_performance_summary(TRADE_HISTORY_FILE, session_start_balance)
        shutdown_mt5_interface()