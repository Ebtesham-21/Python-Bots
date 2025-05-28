import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time # Still used for occasional pauses if needed, but not for main loop timing
import datetime
import numpy as np
import logging
import os
import math

# --- Backtesting Configuration ---
BACKTEST_MODE = True # <<<< SET TO True FOR BACKTESTING, False FOR LIVE >>>>

BACKTEST_START_DATE = datetime.datetime(2024, 7, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
BACKTEST_END_DATE = datetime.datetime(2025, 5, 27, 23, 59, 59, tzinfo=datetime.timezone.utc) # Inclusive
INITIAL_BACKTEST_BALANCE = 250.0
# Optional: For more realistic backtest, you can define these
BACKTEST_SPREAD_PIPS = {"DEFAULT": 1.5} # Default spread in pips, can be symbol-specific e.g. "EURUSD": 0.5
BACKTEST_COMMISSION_PER_LOT_ROUND_TRIP = 0.0 # e.g., 7.0 for $7 per lot round trip

# --- General Configuration ---
SYMBOLS_TO_TRADE_INITIAL = ["EURUSD", "AUDUSD", "USDCHF", "USDCAD",
                             "GBPJPY", "XAUUSD", "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                            "USOIL", "CADJPY",   "XAGUSD", "XPTUSD", "UKOIL",
                            "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD"]
# SYMBOLS_TO_TRADE_INITIAL = ["EURUSD", "AUDUSD"] # For very quick test

ENTRY_TIMEFRAME_MT5 = mt5.TIMEFRAME_M5
FILTER_TIMEFRAME_MT5 = mt5.TIMEFRAME_H1

# --- Live Trading Specific Configuration (Used for symbol properties in backtest) ---
MAGIC_NUMBER = 123457
MAX_SLIPPAGE = 5 # Ignored in this backtest version

# --- Trading Hours Configuration (UTC) ---
TRADING_HOUR_START_UTC = 0
TRADING_HOUR_END_UTC = 20 # Trades won't be opened outside these hours

# --- Symbol-Specific Session Filters (UTC) ---
SYMBOL_SESSIONS = {
    "EURUSD": [(7, 16)], "GBPUSD": [(7, 16)], "AUDUSD": [(0, 4), (7, 16)],
    "USDCHF": [(7, 16)], "USDCAD": [(12, 16)], "USDJPY": [(0, 4), (12, 16)],
    "EURJPY": [(0, 4), (7, 12)], "GBPJPY": [(7, 16)], "NZDUSD": [(0, 4), (7, 16)],
    "EURCHF": [(7, 12)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 16)],
    "EURNZD": [(0, 4), (7, 12)], "GBPNZD": [(7, 12)], "XAUUSD": [(7, 16)],
    "XAGUSD": [(7, 16)], "XPTUSD": [(7, 16)], "USOIL":  [(12, 17)],
    "UKOIL":  [(7, 16)], "BTCUSD":[(7, 16)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 16)], "ETHUSD":[(7, 16)]
}

# --- Strategy Configuration ---
RISK_REWARD_RATIO = 2.0
M5_EMA_SHORT_PERIOD = 20
M5_EMA_MID_PERIOD = 50
M5_EMA_LONG_PERIOD = 100
H1_EMA_SHORT_PERIOD = 20
H1_EMA_MID_PERIOD = 50
H1_EMA_LONG_PERIOD = 100
H1_RSI_PERIOD = 14
H1_RSI_BULL_THRESHOLD = 55
H1_RSI_BEAR_THRESHOLD = 45
H1_MACD_FAST = 12
H1_MACD_SLOW = 26
H1_MACD_SIGNAL = 9

FRACTAL_LOOKBACK = 2
FRACTAL_LOOKFORWARD = 2
N_BARS_FOR_INDICATORS = 250 # Minimum bars needed before indicators are reliable
ADX_PERIOD = 14
ADX_THRESHOLD = 25

ATR_PERIOD = 14
ATR_ROLLING_MEAN_PERIOD = 50
ATR_MULTIPLIER_LOW = 0.8
ATR_MULTIPLIER_HIGH = 2.5

# --- Risk Management (Applied to simulated balance in backtest) ---
MAX_TRADES_PER_SYMBOL_PER_DAY = 3
DAILY_MAX_ACCOUNT_RISK_PERCENT = 5.0
RISK_PER_TRADE_ACCOUNT_PERCENT = 1.0

# --- Trailing Stop Loss Configuration ---
TSL_ACTIVATION_RR_RATIO = 1.5

# --- Global State Variables ---
SYMBOLS_TO_TRADE = []
SYMBOL_PROPERTIES = {} # Populated once at init
# --- Backtest Specific State ---
simulated_account_balance = INITIAL_BACKTEST_BALANCE
simulated_open_positions = [] # List of dicts representing open trades
simulated_current_time_utc = None # Timestamp of the current bar in backtest
historical_data_m5 = {} # {symbol: dataframe}
historical_data_h1 = {} # {symbol: dataframe}
backtest_trade_id_counter = 0

# --- Shared State (used by both live & backtest logic, reset daily) ---
daily_trade_counts = {}
daily_start_balance_utc = 0.0 # For backtest, this is the balance at UTC 00:00
last_checked_day_utc_for_reset = None
daily_risk_budget_currency_global = 0.0
current_daily_risked_amount_global = 0.0 # Sum of RiskedAmount for trades opened today
daily_risk_budget_exceeded_today_global = False


# --- Logging Setup ---
log_file_name = "backtesting_bot.log" if BACKTEST_MODE else "trading_bot.log"
# Clear log file at start of each run
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()])

# --- Trade History Logging ---
TRADE_HISTORY_FILE_BASE = "trading_history.csv"
TRADE_HISTORY_FILE = "backtest_" + TRADE_HISTORY_FILE_BASE if BACKTEST_MODE else TRADE_HISTORY_FILE_BASE
TRADE_HISTORY_COLUMNS = [
    "TicketID", "PositionID", "Symbol", "Type", "OpenTimeUTC", "EntryPrice",
    "LotSize", "SL_Price", "TP_Price", "CloseTimeUTC", "ExitPrice",
    "PNL_AccountCCY", "OpenComment", "CloseReason", "RiskedAmount"
]
trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)


def load_or_initialize_trade_history(): # Used by live, for backtest, we always start fresh
    global trade_history_df
    if BACKTEST_MODE:
        trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
        logging.info("Initialized fresh trade history for backtest.")
        return

    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            temp_df_list = []
            with open(TRADE_HISTORY_FILE, 'r') as f:
                header = f.readline().strip()
                if not header:
                     trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
                     logging.info(f"Trade history file {TRADE_HISTORY_FILE} is empty. Initialized new history.")
                     return

                num_columns_expected = len(TRADE_HISTORY_COLUMNS)
                num_columns_file = len(header.split(','))

                if num_columns_file not in [num_columns_expected, num_columns_expected -1]:
                    logging.warning(f"Trade history CSV header mismatch. Expected {num_columns_expected} or {num_columns_expected-1} columns, got {num_columns_file}. Re-initializing.")
                    trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
                    return

                temp_df_list.append(header)
                for line in f:
                    if line.strip().startswith("--- Performance Summary ---"): # Stop before any summary
                        break
                    if len(line.strip().split(',')) == num_columns_file:
                         temp_df_list.append(line.strip())

            if len(temp_df_list) > 1:
                from io import StringIO
                csv_data_str = "\n".join(temp_df_list)
                trade_history_df = pd.read_csv(StringIO(csv_data_str))
                if 'RiskedAmount' not in trade_history_df.columns:
                    trade_history_df['RiskedAmount'] = np.nan
            else:
                trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)

            for col in ["OpenTimeUTC", "CloseTimeUTC"]:
                if col in trade_history_df.columns:
                    trade_history_df[col] = pd.to_datetime(trade_history_df[col], errors='coerce', utc=True)
            numeric_cols = ['EntryPrice', 'LotSize', 'SL_Price', 'TP_Price', 'ExitPrice', 'PNL_AccountCCY', 'TicketID', 'PositionID', 'RiskedAmount']
            for col in numeric_cols:
                if col in trade_history_df.columns:
                    trade_history_df[col] = pd.to_numeric(trade_history_df[col], errors='coerce')
            logging.info(f"Loaded {len(trade_history_df)} trade records from {TRADE_HISTORY_FILE}")
        except Exception as e:
            logging.error(f"Error loading trade history: {e}. Initializing new history.")
            trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
    else:
        trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
        logging.info(f"No existing trade history file. Initialized new history.")

def save_trade_history():
    global trade_history_df
    try:
        df_to_save = trade_history_df.copy()
        for col in ['PNL_AccountCCY', 'RiskedAmount']:
             if col in df_to_save.columns:
                df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce').round(2)

        df_to_save.to_csv(TRADE_HISTORY_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S.%f') # Write data first

        all_summary_lines = [] # Collect all summary lines to write them at once

        # --- Overall Performance Summary ---
        overall_summary_lines = ["\n--- Overall Performance Summary ---"]
        closed_trades_overall = df_to_save[pd.notna(df_to_save['CloseTimeUTC']) & pd.notna(df_to_save['PNL_AccountCCY'])].copy()
        closed_trades_overall.sort_values(by='CloseTimeUTC', inplace=True)
        
        overall_currency = "USD" 
        if SYMBOL_PROPERTIES and SYMBOLS_TO_TRADE:
            first_symbol_overall = SYMBOLS_TO_TRADE[0]
            if first_symbol_overall in SYMBOL_PROPERTIES and SYMBOL_PROPERTIES[first_symbol_overall].get('currency_profit'):
                overall_currency = SYMBOL_PROPERTIES[first_symbol_overall]['currency_profit']
        if not BACKTEST_MODE:
            acc_info_live = mt5.account_info()
            if acc_info_live and acc_info_live.currency: overall_currency = acc_info_live.currency
            else:
                ti = mt5.terminal_info()
                if ti and hasattr(ti, 'currency') and ti.currency: overall_currency = ti.currency

        if not closed_trades_overall.empty:
            total_trades = len(closed_trades_overall)
            winning_trades_df = closed_trades_overall[closed_trades_overall['PNL_AccountCCY'] > 0]
            losing_trades_df = closed_trades_overall[closed_trades_overall['PNL_AccountCCY'] < 0]
            num_wins = len(winning_trades_df)
            num_losses = len(losing_trades_df)
            win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
            total_pnl = closed_trades_overall['PNL_AccountCCY'].sum()
            total_risked = closed_trades_overall['RiskedAmount'].sum(skipna=True) if 'RiskedAmount' in closed_trades_overall.columns else np.nan
            sum_profits = winning_trades_df['PNL_AccountCCY'].sum()
            sum_losses = abs(losing_trades_df['PNL_AccountCCY'].sum())
            profit_factor = (sum_profits / sum_losses) if sum_losses > 0 else (float('inf') if sum_profits > 0 else 0.0)

            overall_summary_lines.append(f"Total Closed Trades: {total_trades}")
            overall_summary_lines.append(f"Winning Trades: {num_wins}")
            overall_summary_lines.append(f"Losing Trades: {num_losses}")
            overall_summary_lines.append(f"Win Rate: {win_rate:.2f}%")
            overall_summary_lines.append(f"Total PNL ({overall_currency}): {total_pnl:.2f}")
            if not pd.isna(total_risked):
                 overall_summary_lines.append(f"Total Amount Risked ({overall_currency}): {total_risked:.2f}")
            overall_summary_lines.append(f"Profit Factor: {profit_factor:.2f}")

            max_dd_val, max_dd_pct, final_bal = calculate_max_drawdown(closed_trades_overall['PNL_AccountCCY'].values, INITIAL_BACKTEST_BALANCE if BACKTEST_MODE else None)
            
            if BACKTEST_MODE:
                overall_summary_lines.append(f"Initial Balance ({overall_currency}): {INITIAL_BACKTEST_BALANCE:.2f}")
                overall_summary_lines.append(f"Final Balance ({overall_currency}): {final_bal:.2f}")
            overall_summary_lines.append(f"Max Drawdown ({overall_currency}): {max_dd_val:.2f}")
            if pd.notna(max_dd_pct):
                overall_summary_lines.append(f"Max Drawdown (%): {max_dd_pct:.2f}%")
            else:
                overall_summary_lines.append(f"Max Drawdown (%): N/A")
        else:
            overall_summary_lines.append("No closed trades to analyze.")
            if BACKTEST_MODE:
                 overall_summary_lines.append(f"Initial Balance ({overall_currency}): {INITIAL_BACKTEST_BALANCE:.2f}")
                 overall_summary_lines.append(f"Final Balance ({overall_currency}): {INITIAL_BACKTEST_BALANCE:.2f}")
            overall_summary_lines.append(f"Max Drawdown ({overall_currency}): 0.00")
            overall_summary_lines.append(f"Max Drawdown (%): 0.00%")
        
        all_summary_lines.extend(overall_summary_lines)

        # --- Symbol-Specific Performance Summary ---
        traded_symbols = closed_trades_overall['Symbol'].unique()
        for symbol in traded_symbols:
            symbol_summary_lines = [f"\n--- Performance Summary for {symbol} ---"]
            closed_trades_symbol = closed_trades_overall[closed_trades_overall['Symbol'] == symbol].copy()
            # No need to re-sort, already sorted overall
            
            symbol_currency = SYMBOL_PROPERTIES.get(symbol, {}).get('currency_profit', overall_currency)

            if not closed_trades_symbol.empty:
                s_total_trades = len(closed_trades_symbol)
                s_winning_df = closed_trades_symbol[closed_trades_symbol['PNL_AccountCCY'] > 0]
                s_losing_df = closed_trades_symbol[closed_trades_symbol['PNL_AccountCCY'] < 0]
                s_num_wins = len(s_winning_df)
                s_num_losses = len(s_losing_df)
                s_win_rate = (s_num_wins / s_total_trades * 100) if s_total_trades > 0 else 0
                s_total_pnl = closed_trades_symbol['PNL_AccountCCY'].sum()
                s_total_risked = closed_trades_symbol['RiskedAmount'].sum(skipna=True) if 'RiskedAmount' in closed_trades_symbol.columns else np.nan
                s_sum_profits = s_winning_df['PNL_AccountCCY'].sum()
                s_sum_losses = abs(s_losing_df['PNL_AccountCCY'].sum())
                s_profit_factor = (s_sum_profits / s_sum_losses) if s_sum_losses > 0 else (float('inf') if s_sum_profits > 0 else 0.0)

                symbol_summary_lines.append(f"Total Closed Trades: {s_total_trades}")
                symbol_summary_lines.append(f"Winning Trades: {s_num_wins}")
                symbol_summary_lines.append(f"Losing Trades: {s_num_losses}")
                symbol_summary_lines.append(f"Win Rate: {s_win_rate:.2f}%")
                symbol_summary_lines.append(f"Total PNL ({symbol_currency}): {s_total_pnl:.2f}")
                if not pd.isna(s_total_risked):
                    symbol_summary_lines.append(f"Total Amount Risked ({symbol_currency}): {s_total_risked:.2f}")
                symbol_summary_lines.append(f"Profit Factor: {s_profit_factor:.2f}")

                # For symbol-specific drawdown, calculate from a starting base of 0 for that symbol's PNL stream
                s_max_dd_val, s_max_dd_pct, _ = calculate_max_drawdown(closed_trades_symbol['PNL_AccountCCY'].values, 0.0)
                symbol_summary_lines.append(f"Max Drawdown ({symbol_currency}, from symbol PNL stream): {s_max_dd_val:.2f}")
                if pd.notna(s_max_dd_pct):
                    symbol_summary_lines.append(f"Max Drawdown (%, from symbol PNL stream): {s_max_dd_pct:.2f}%")
                else:
                    symbol_summary_lines.append(f"Max Drawdown (%, from symbol PNL stream): N/A")
            else:
                symbol_summary_lines.append("No closed trades to analyze for this symbol.")
            
            all_summary_lines.extend(symbol_summary_lines)

        with open(TRADE_HISTORY_FILE, 'a') as f:
            for line in all_summary_lines:
                f.write(line + "\n")
        logging.info(f"Trade history and all summaries saved to {TRADE_HISTORY_FILE}")
    except Exception as e:
        logging.error(f"Error saving trade history or calculating summary: {e}", exc_info=True)

def calculate_max_drawdown(pnl_series_values, initial_balance_for_calc):
    """Calculates max drawdown value and percentage from a series of PNLs."""
    if initial_balance_for_calc is None: # Live mode, cannot reliably get initial balance for historical trades
        # Calculate drawdown purely based on PNL stream assuming 0 start (relative drawdown)
        current_pnl_sum = 0
        peak_pnl_sum = 0
        max_dd_val = 0
        equity_curve_pnl_only = [0.0]
        for pnl in pnl_series_values:
            current_pnl_sum += pnl
            equity_curve_pnl_only.append(current_pnl_sum)
            peak_pnl_sum = max(peak_pnl_sum, current_pnl_sum)
            drawdown = peak_pnl_sum - current_pnl_sum
            max_dd_val = max(max_dd_val, drawdown)
        
        # Percentage for PNL-only stream is tricky and can be misleading without a proper capital base
        # For simplicity, we'll return NaN for percentage if initial_balance is None
        return max_dd_val, np.nan, (INITIAL_BACKTEST_BALANCE + current_pnl_sum if BACKTEST_MODE else np.nan)


    current_equity = initial_balance_for_calc
    peak_equity = initial_balance_for_calc
    max_dd_val = 0.0
    equity_curve = [initial_balance_for_calc]

    for pnl in pnl_series_values:
        current_equity += pnl
        equity_curve.append(current_equity)
        peak_equity = max(peak_equity, current_equity)
        drawdown = peak_equity - current_equity
        max_dd_val = max(max_dd_val, drawdown)
    
    final_balance = current_equity

    max_dd_pct = np.nan
    if len(equity_curve) > 1:
        equity_series = pd.Series(equity_curve)
        running_max_equity = equity_series.expanding(min_periods=1).max()
        drawdown_percentages = []
        for i in range(len(equity_series)):
            # Only calculate % if the peak was positive, to avoid division by zero or misleading % from negative peaks
            if running_max_equity[i] > 1e-6: 
                dd_val_at_i = running_max_equity[i] - equity_series[i]
                drawdown_percentages.append((dd_val_at_i / running_max_equity[i]) * 100.0)
        
        if drawdown_percentages:
            max_dd_pct = max(drawdown_percentages)
        elif max_dd_val > 0 : # If there was a drawdown value but no valid percentage (e.g. all peaks were <=0)
            max_dd_pct = np.nan 
        else: # No drawdown value, so 0%
            max_dd_pct = 0.0
            
    return max_dd_val, max_dd_pct, final_balance


# --- LIVE TRADING FUNCTIONS (some adapted for backtest use) ---
def log_opened_trade(symbol, trade_type, lot_size, entry_price, sl_price, tp_price, open_time, comment, risked_amount, trade_id):
    global trade_history_df
    new_trade_record = {
        "TicketID": trade_id, "PositionID": trade_id,
        "Symbol": symbol, "Type": "BUY" if trade_type == mt5.ORDER_TYPE_BUY else "SELL",
        "OpenTimeUTC": open_time, "EntryPrice": entry_price, "LotSize": lot_size,
        "SL_Price": sl_price, "TP_Price": tp_price,
        "CloseTimeUTC": pd.NaT, "ExitPrice": np.nan, "PNL_AccountCCY": np.nan,
        "OpenComment": comment, "CloseReason": "", "RiskedAmount": risked_amount
    }
    new_row_df = pd.DataFrame([new_trade_record])
    trade_history_df = pd.concat([trade_history_df, new_row_df], ignore_index=True)
    logging.info(f"BACKTEST: Logged opened trade. ID: {trade_id}, Symbol: {symbol}, Type: {new_trade_record['Type']}, Risked: {risked_amount:.2f}")

def log_closed_trade_backtest(trade_id, close_time, exit_price, pnl, close_reason):
    global trade_history_df, simulated_account_balance
    idx = trade_history_df[trade_history_df['TicketID'] == trade_id].index
    if not idx.empty:
        trade_history_df.loc[idx[0], 'CloseTimeUTC'] = close_time
        trade_history_df.loc[idx[0], 'ExitPrice'] = exit_price
        trade_history_df.loc[idx[0], 'PNL_AccountCCY'] = pnl
        trade_history_df.loc[idx[0], 'CloseReason'] = close_reason
        simulated_account_balance += pnl 
        logging.info(f"BACKTEST: Logged closed trade. ID: {trade_id}, Exit: {exit_price}, PNL: {pnl:.2f}, Reason: {close_reason}, New Bal: {simulated_account_balance:.2f}")
    else:
        logging.error(f"BACKTEST: Could not find trade ID {trade_id} to log closure.")


def timeframe_to_string(tf_int): 
    timeframes = {
        mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M2: "M2", mt5.TIMEFRAME_M3: "M3",
        mt5.TIMEFRAME_M4: "M4", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M6: "M6",
        mt5.TIMEFRAME_M10: "M10", mt5.TIMEFRAME_M12: "M12", mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M20: "M20", mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H2: "H2", mt5.TIMEFRAME_H3: "H3",
        mt5.TIMEFRAME_H4: "H4", mt5.TIMEFRAME_H6: "H6", mt5.TIMEFRAME_H8: "H8",
        mt5.TIMEFRAME_H12: "H12",
        mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1"
    }
    return timeframes.get(tf_int, f"UnknownTF({tf_int})")

def initialize_mt5_for_properties(): 
    global SYMBOLS_TO_TRADE, SYMBOL_PROPERTIES
    if not mt5.initialize():
        logging.error(f"MT5 initialize() failed for properties, error code = {mt5.last_error()}")
        return False
    logging.info("MetaTrader 5 Initialized for Symbol Properties")
    
    term_info = mt5.terminal_info()
    if term_info:
        term_currency = term_info.currency if hasattr(term_info, 'currency') else "N/A"
        logging.info(f"Terminal Info (for context): Name: {term_info.name}, Company: {term_info.company}, Path: {term_info.path}, Currency: {term_currency}")
        
        acc_info = mt5.account_info() 
        if acc_info:
            logging.info(f"Account Info (for context): Login: {acc_info.login}, Server: {acc_info.server}, Balance: {acc_info.balance} {acc_info.currency}")
        else:
            logging.warning(f"Could not get account_info for detailed context, error: {mt5.last_error()}")
    else:
        logging.warning(f"Failed to get terminal_info during property initialization, error code = {mt5.last_error()}")


    symbols_to_check_locally = list(SYMBOLS_TO_TRADE_INITIAL)
    successfully_initialized_symbols = []
    temp_symbol_properties = {}

    for symbol_name in symbols_to_check_locally:
        symbol_info_obj = mt5.symbol_info(symbol_name)
        if symbol_info_obj is None:
            logging.warning(f"Symbol {symbol_name} not found by broker (for properties). Skipping.")
            continue
        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point, 'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
            'trade_tick_value': symbol_info_obj.trade_tick_value,
            'volume_min': symbol_info_obj.volume_min,
            'volume_step': symbol_info_obj.volume_step,
            'volume_max': symbol_info_obj.volume_max,
            'trade_contract_size': symbol_info_obj.trade_contract_size,
            'spread': symbol_info_obj.spread, 
            'currency_profit': symbol_info_obj.currency_profit,
            'currency_margin': symbol_info_obj.currency_margin,
        }
        successfully_initialized_symbols.append(symbol_name)
        logging.info(f"Properties for {symbol_name} fetched. Min Lot: {temp_symbol_properties[symbol_name]['volume_min']}, Point: {temp_symbol_properties[symbol_name]['point']}")

    if not successfully_initialized_symbols:
        logging.error("No symbols were successfully initialized for properties.")
        ti = mt5.terminal_info()
        if ti and ti.connected:
            mt5.shutdown()
        return False

    SYMBOLS_TO_TRADE = successfully_initialized_symbols
    SYMBOL_PROPERTIES = temp_symbol_properties
    logging.info(f"Successfully fetched properties for symbols: {SYMBOLS_TO_TRADE}")
    return True


def williams_fractals(df_high, df_low, n_left=FRACTAL_LOOKBACK, n_right=FRACTAL_LOOKFORWARD): 
    up_fractals = pd.Series(index=df_high.index, dtype='bool').fillna(False)
    down_fractals = pd.Series(index=df_low.index, dtype='bool').fillna(False)
    if len(df_high) < n_left + n_right + 1:
        return pd.DataFrame({'fractal_up': up_fractals, 'fractal_down': down_fractals})

    for i in range(n_left, len(df_high) - n_right):
        is_up = True
        for j in range(1, n_left + 1):
            if df_high.iloc[i] < df_high.iloc[i-j]: 
                is_up = False; break
        if not is_up: continue
        for j in range(1, n_right + 1):
            if df_high.iloc[i] <= df_high.iloc[i+j]: 
                is_up = False; break
        if is_up: up_fractals.iloc[i] = True

        is_down = True
        for j in range(1, n_left + 1):
            if df_low.iloc[i] > df_low.iloc[i-j]: 
                is_down = False; break
        if not is_down: continue
        for j in range(1, n_right + 1):
            if df_low.iloc[i] >= df_low.iloc[i+j]: 
                is_down = False; break
        if is_down: down_fractals.iloc[i] = True
            
    return pd.DataFrame({'fractal_up': up_fractals, 'fractal_down': down_fractals}, index=df_high.index)


def calculate_indicators_for_df(df_input, ema_short_len, ema_mid_len, ema_long_len, adx_len=None, rsi_len=None, prefix="", **kwargs):
    df = df_input.copy()
    if df.empty: return df

    df[f'{prefix}EMA_{ema_short_len}'] = ta.ema(df['close'], length=ema_short_len)
    df[f'{prefix}EMA_{ema_mid_len}'] = ta.ema(df['close'], length=ema_mid_len)
    df[f'{prefix}EMA_{ema_long_len}'] = ta.ema(df['close'], length=ema_long_len)

    if adx_len:
        adx_indicator = ta.adx(df['high'], df['low'], df['close'], length=adx_len)
        if adx_indicator is not None and not adx_indicator.empty and f'ADX_{adx_len}' in adx_indicator.columns:
            df[f'{prefix}ADX_{adx_len}'] = adx_indicator[f'ADX_{adx_len}']
        else: df[f'{prefix}ADX_{adx_len}'] = np.nan
    if rsi_len:
        df[f'{prefix}RSI_{rsi_len}'] = ta.rsi(df['close'], length=rsi_len)

    if not prefix and 'atr_len' in kwargs: 
        atr_len_val = kwargs['atr_len']
        atr_col_name = f'ATR_{atr_len_val}'
        atr_sma_col_name = f'ATR_{atr_len_val}_SMA{ATR_ROLLING_MEAN_PERIOD}'
        df[atr_col_name] = ta.atr(df['high'], df['low'], df['close'], length=atr_len_val)
        df[atr_sma_col_name] = df[atr_col_name].rolling(window=ATR_ROLLING_MEAN_PERIOD, min_periods=max(1, ATR_ROLLING_MEAN_PERIOD // 2)).mean()

    if prefix == "H1_": 
        if kwargs.get('add_macd', False):
            macd_df = ta.macd(df['close'], fast=H1_MACD_FAST, slow=H1_MACD_SLOW, signal=H1_MACD_SIGNAL)
            if macd_df is not None and not macd_df.empty and len(macd_df.columns) >=3:
                df[f'{prefix}MACD_LINE'] = macd_df.iloc[:,0] 
                df[f'{prefix}MACD_HIST'] = macd_df.iloc[:,1] 
                df[f'{prefix}MACD_SIGNAL'] = macd_df.iloc[:,2]
            else:
                df[f'{prefix}MACD_LINE'], df[f'{prefix}MACD_HIST'], df[f'{prefix}MACD_SIGNAL'] = np.nan, np.nan, np.nan
    
    if not prefix: 
        if len(df) >= FRACTAL_LOOKBACK + FRACTAL_LOOKFORWARD + 1:
            fractal_df = williams_fractals(df['high'], df['low']) 
            df = pd.concat([df, fractal_df], axis=1)
        else:
            df['fractal_up'] = False
            df['fractal_down'] = False
    return df


def get_value_of_one_point_generic(symbol, reference_price_for_calc): 
    props = SYMBOL_PROPERTIES.get(symbol)
    if not props:
        logging.error(f"Symbol {symbol} not found in SYMBOL_PROPERTIES for point value calculation.")
        return None
    point = props['point']

    if reference_price_for_calc == 0:
        logging.error(f"Reference price for {symbol} is 0. Cannot calculate point value.")
        return None

    profit = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, reference_price_for_calc, reference_price_for_calc + point)
    if profit is None:
        logging.error(f"mt5.order_calc_profit returned None for {symbol} at price {reference_price_for_calc}, point {point}. Error: {mt5.last_error()}")
        profit_10_points = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, reference_price_for_calc, reference_price_for_calc + (10 * point))
        if profit_10_points is not None:
            profit = profit_10_points / 10.0
            logging.warning(f"Used 10-point deviation for {symbol} pip value calculation. Result per point: {profit}")
        else:
            logging.error(f"mt5.order_calc_profit also failed for 10-point deviation for {symbol}. Error: {mt5.last_error()}")
            return None
    if profit < 0: profit = abs(profit) 
    if profit == 0:
        logging.warning(f"Calculated point value is 0 for {symbol} at price {reference_price_for_calc} with point size {point} using order_calc_profit.")
    return profit


def calculate_dynamic_lot_size(symbol, sl_pips, account_balance, risk_per_trade_percent, reference_price_for_pip_value_calc):
    props = SYMBOL_PROPERTIES.get(symbol)
    if not props:
        logging.error(f"Symbol {symbol} not found in SYMBOL_PROPERTIES for lot calculation.")
        return props.get('volume_min', 0.01) if props else 0.01

    if account_balance is None or account_balance <= 0:
        logging.error(f"Invalid account balance ({account_balance}) for lot calculation. Using min lot for {symbol}.")
        return props['volume_min']
    if sl_pips <= 0:
        logging.error(f"Stop loss in pips ({sl_pips}) is zero or negative for {symbol}. Cannot calculate dynamic lot. Using min lot.")
        return props['volume_min']

    amount_to_risk = account_balance * (risk_per_trade_percent / 100.0)
    
    value_of_one_point = get_value_of_one_point_generic(symbol, reference_price_for_pip_value_calc)
    if value_of_one_point is None or value_of_one_point <= 1e-9:
        logging.error(f"Invalid or zero point value ({value_of_one_point}) for {symbol}. Cannot calculate dynamic lot size. Using min lot.")
        return props['volume_min']

    value_of_sl_pips_one_lot = sl_pips * value_of_one_point
    if value_of_sl_pips_one_lot <= 1e-9:
        logging.error(f"SL pips monetary value ({value_of_sl_pips_one_lot}) is zero or negative for {symbol} (SL pips: {sl_pips}, Point value: {value_of_one_point}). Using min lot.")
        return props['volume_min']

    lot_size = amount_to_risk / value_of_sl_pips_one_lot
    
    lot_size = max(lot_size, props['volume_min'])
    if props['volume_max'] > 0: lot_size = min(lot_size, props['volume_max'])

    if props['volume_step'] > 0:
        lot_size = math.floor(lot_size / props['volume_step']) * props['volume_step']
        lot_precision = 0
        if props['volume_step'] < 1:
            step_str = format(props['volume_step'], '.8f').rstrip('0')
            if '.' in step_str: lot_precision = len(step_str.split('.')[1])
        lot_size = round(lot_size, lot_precision)

    lot_size = max(lot_size, props['volume_min']) 
    
    if lot_size < props['volume_min']: return props['volume_min']
    if props['volume_max'] > 0 and lot_size > props['volume_max']: return props['volume_max']
        
    return lot_size


# --- BACKTESTING SPECIFIC FUNCTIONS ---
def get_historical_data_for_symbol(symbol, timeframe_mt5, start_dt, end_dt):
    logging.info(f"Fetching historical data for {symbol} on {timeframe_to_string(timeframe_mt5)} from {start_dt} to {end_dt}")
    rates = mt5.copy_rates_range(symbol, timeframe_mt5, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        logging.warning(f"Could not fetch {timeframe_to_string(timeframe_mt5)} data for {symbol} in range.")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time_dt', inplace=True)
    df.rename(columns={'tick_volume': 'volume', 'real_volume': 'volume'} , errors='ignore', inplace=True)
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']

    return df

def get_historical_data_for_all_symbols(symbols_list, start_dt, end_dt):
    global historical_data_m5, historical_data_h1, SYMBOLS_TO_TRADE # Allow modification of SYMBOLS_TO_TRADE
    
    # Create a copy of the initial list to iterate over, as SYMBOLS_TO_TRADE might be modified
    symbols_to_process = list(symbols_list)
    valid_symbols_for_backtest = []


    min_bars_needed = N_BARS_FOR_INDICATORS + FRACTAL_LOOKFORWARD + ATR_ROLLING_MEAN_PERIOD + 20 

    if ENTRY_TIMEFRAME_MT5 == mt5.TIMEFRAME_M5:
        buffer_days_m5 = math.ceil(min_bars_needed / (24 * 12 * 0.8)) + 5 
        buffer_days_h1 = math.ceil(min_bars_needed / (24 * 0.8)) + 2
    else: 
        buffer_days_m5 = 10
        buffer_days_h1 = 10

    fetch_start_dt_m5 = start_dt - datetime.timedelta(days=buffer_days_m5)
    fetch_start_dt_h1 = start_dt - datetime.timedelta(days=buffer_days_h1)

    for symbol in symbols_to_process:
        df_m5 = get_historical_data_for_symbol(symbol, ENTRY_TIMEFRAME_MT5, fetch_start_dt_m5, end_dt)
        df_h1 = get_historical_data_for_symbol(symbol, FILTER_TIMEFRAME_MT5, fetch_start_dt_h1, end_dt)

        if df_m5.empty or df_h1.empty:
            logging.error(f"Could not load sufficient M5 or H1 data for {symbol}. It will be excluded from backtest.")
            continue # Skip this symbol

        # Pre-calculate all indicators on the full dataset once
        df_m5_with_indicators = calculate_indicators_for_df(df_m5, M5_EMA_SHORT_PERIOD, M5_EMA_MID_PERIOD, M5_EMA_LONG_PERIOD,
                                                           adx_len=ADX_PERIOD, prefix="", atr_len=ATR_PERIOD)
        historical_data_m5[symbol] = df_m5_with_indicators
        logging.info(f"Loaded and pre-calculated M5 indicators for {symbol}, {len(df_m5_with_indicators)} bars.")
        
        df_h1_with_indicators = calculate_indicators_for_df(df_h1, H1_EMA_SHORT_PERIOD, H1_EMA_MID_PERIOD, H1_EMA_LONG_PERIOD,
                                                           rsi_len=H1_RSI_PERIOD, prefix="H1_", add_macd=True)
        historical_data_h1[symbol] = df_h1_with_indicators
        logging.info(f"Loaded and pre-calculated H1 indicators for {symbol}, {len(df_h1_with_indicators)} bars.")
        
        valid_symbols_for_backtest.append(symbol)

    SYMBOLS_TO_TRADE = valid_symbols_for_backtest # Update global list to only contain valid symbols


def get_latest_data_and_indicators_backtest(symbol, current_bar_time_m5):
    df_m5_full = historical_data_m5.get(symbol)
    df_h1_full = historical_data_h1.get(symbol)

    if df_m5_full is None or df_h1_full is None:
        return None, None, None, None 

    try:
        current_m5_bar_index_loc = df_m5_full.index.get_loc(current_bar_time_m5)
    except KeyError:
        return None, None, None, None
        
    m5_signal_candle_idx = current_m5_bar_index_loc - 1
    m5_prev_signal_candle_idx = m5_signal_candle_idx -1
    m5_fractal_ref_candle_idx = m5_signal_candle_idx - FRACTAL_LOOKFORWARD 

    min_idx_needed_for_fractal_calc = FRACTAL_LOOKBACK 
    
    if m5_signal_candle_idx < 0 or \
       m5_prev_signal_candle_idx < 0 or \
       m5_fractal_ref_candle_idx < 0 or \
       m5_fractal_ref_candle_idx < FRACTAL_LOOKBACK: 
        return None, None, None, None

    m5_signal_candle_series = df_m5_full.iloc[m5_signal_candle_idx]
    m5_prev_signal_candle_series = df_m5_full.iloc[m5_prev_signal_candle_idx]
    m5_fractal_candle_series = df_m5_full.iloc[m5_fractal_ref_candle_idx] 

    h1_aligned_candles = df_h1_full[df_h1_full.index <= m5_signal_candle_series.name]
    if h1_aligned_candles.empty:
        return None, None, None, None
    h1_signal_candle_series = h1_aligned_candles.iloc[-1]

    return m5_signal_candle_series, m5_prev_signal_candle_series, m5_fractal_candle_series, h1_signal_candle_series


def simulate_get_current_tick_backtest(symbol, current_m5_bar_data):
    return {
        'ask': current_m5_bar_data['open'], 
        'bid': current_m5_bar_data['open'],
        'last': current_m5_bar_data['close'], 
        'time': int(current_m5_bar_data.name.timestamp()) 
    }

def simulate_place_trade_order(symbol, trade_type, lot_size, entry_price_sim, sl_price, tp_price, open_time_sim, comment_sim, risked_amount_sim):
    global backtest_trade_id_counter, simulated_open_positions
    backtest_trade_id_counter += 1
    trade_id = backtest_trade_id_counter

    position = {
        'ticket': trade_id, 
        'symbol': symbol,
        'type': trade_type, 
        'volume': lot_size,
        'price_open': entry_price_sim,
        'sl': sl_price,
        'tp': tp_price,
        'time_open': open_time_sim,
        'comment': comment_sim,
        'initial_sl_pips_risked': (abs(entry_price_sim - sl_price) / SYMBOL_PROPERTIES[symbol]['point']) if SYMBOL_PROPERTIES[symbol]['point'] > 0 else 0,
        'risked_amount': risked_amount_sim 
    }
    simulated_open_positions.append(position)
    log_opened_trade(symbol, trade_type, lot_size, entry_price_sim, sl_price, tp_price, open_time_sim, comment_sim, risked_amount_sim, trade_id)
    
    return {'retcode': mt5.TRADE_RETCODE_DONE, 'order': trade_id, 'position': trade_id, 'price': entry_price_sim, 'deal': trade_id*10}, trade_id

def manage_trailing_stop_loss_backtest(position, current_m5_bar_data):
    symbol = position['symbol']
    trade_type = position['type']
    entry_price = position['price_open']
    current_sl = position['sl']

    initial_sl_pips_risked = position.get('initial_sl_pips_risked', 0)
    if initial_sl_pips_risked <= 0:
        return 

    props = SYMBOL_PROPERTIES[symbol]
    point = props['point']; digits = props['digits']

    tsl_considered_active_or_at_be = False
    new_sl_proposed = current_sl

    if trade_type == mt5.ORDER_TYPE_BUY:
        current_market_price_for_profit_check = current_m5_bar_data['high'] 
        if current_sl >= entry_price: tsl_considered_active_or_at_be = True
        
        if not tsl_considered_active_or_at_be:
            activation_profit_pips = initial_sl_pips_risked * TSL_ACTIVATION_RR_RATIO
            activation_target_price = entry_price + (activation_profit_pips * point)
            if current_market_price_for_profit_check >= activation_target_price:
                new_sl_at_be = round(entry_price, digits)
                if new_sl_at_be > current_sl:
                    new_sl_proposed = new_sl_at_be
                    tsl_considered_active_or_at_be = True
                    logging.info(f"BACKTEST TSL: Activating for BUY {symbol} (Pos:{position['ticket']}). SL to BE: {new_sl_proposed:.{digits}f}")
        
        if tsl_considered_active_or_at_be: 
            potential_trailed_sl = round(current_market_price_for_profit_check - (initial_sl_pips_risked * point), digits)
            if potential_trailed_sl > new_sl_proposed: 
                new_sl_proposed = potential_trailed_sl
                logging.info(f"BACKTEST TSL: Trailing SL for BUY {symbol} (Pos:{position['ticket']}). New SL: {new_sl_proposed:.{digits}f}")

    elif trade_type == mt5.ORDER_TYPE_SELL:
        current_market_price_for_profit_check = current_m5_bar_data['low'] 
        if current_sl <= entry_price and current_sl != 0.0: tsl_considered_active_or_at_be = True

        if not tsl_considered_active_or_at_be:
            activation_profit_pips = initial_sl_pips_risked * TSL_ACTIVATION_RR_RATIO
            activation_target_price = entry_price - (activation_profit_pips * point)
            if current_market_price_for_profit_check <= activation_target_price:
                new_sl_at_be = round(entry_price, digits)
                if new_sl_at_be < current_sl or current_sl == 0.0:
                    new_sl_proposed = new_sl_at_be
                    tsl_considered_active_or_at_be = True
                    logging.info(f"BACKTEST TSL: Activating for SELL {symbol} (Pos:{position['ticket']}). SL to BE: {new_sl_proposed:.{digits}f}")

        if tsl_considered_active_or_at_be:
            potential_trailed_sl = round(current_market_price_for_profit_check + (initial_sl_pips_risked * point), digits)
            if potential_trailed_sl < new_sl_proposed: 
                new_sl_proposed = potential_trailed_sl
                logging.info(f"BACKTEST TSL: Trailing SL for SELL {symbol} (Pos:{position['ticket']}). New SL: {new_sl_proposed:.{digits}f}")
    
    if new_sl_proposed != current_sl:
        position['sl'] = new_sl_proposed 


def simulate_manage_open_positions(current_m5_bar_data_dict):
    global simulated_open_positions, simulated_account_balance
    positions_to_remove = []

    for position in simulated_open_positions:
        symbol = position['symbol']
        current_m5_bar_for_symbol = current_m5_bar_data_dict.get(symbol)
        if current_m5_bar_for_symbol is None:
            continue 

        manage_trailing_stop_loss_backtest(position, current_m5_bar_for_symbol)
        
        bar_high = current_m5_bar_for_symbol['high']
        bar_low = current_m5_bar_for_symbol['low']
        bar_time = current_m5_bar_for_symbol.name 

        exit_price = None
        close_reason = None
        pnl = 0.0

        if position['type'] == mt5.ORDER_TYPE_BUY:
            if position['sl'] != 0 and bar_low <= position['sl']:
                exit_price = position['sl']
                close_reason = "StopLoss"
            elif position['tp'] != 0 and bar_high >= position['tp']:
                exit_price = position['tp']
                close_reason = "TakeProfit"
        
        elif position['type'] == mt5.ORDER_TYPE_SELL:
            if position['sl'] != 0 and bar_high >= position['sl']:
                exit_price = position['sl']
                close_reason = "StopLoss"
            elif position['tp'] != 0 and bar_low <= position['tp']:
                exit_price = position['tp']
                close_reason = "TakeProfit"

        if exit_price is not None:
            pnl = mt5.order_calc_profit(position['type'], symbol, position['volume'], 
                                        position['price_open'], exit_price)
            if pnl is None:
                logging.error(f"mt5.order_calc_profit failed for closing trade {position['ticket']}. PNL set to 0.")
                pnl = 0.0
            
            log_closed_trade_backtest(position['ticket'], bar_time, exit_price, pnl, close_reason)
            positions_to_remove.append(position)

    simulated_open_positions = [p for p in simulated_open_positions if p not in positions_to_remove]


def check_for_new_trade_signals_backtest():
    global daily_trade_counts, daily_start_balance_utc, last_checked_day_utc_for_reset, simulated_current_time_utc
    global daily_risk_budget_currency_global, current_daily_risked_amount_global, daily_risk_budget_exceeded_today_global
    global backtest_trade_id_counter, simulated_open_positions, simulated_account_balance

    current_day_str = simulated_current_time_utc.strftime('%Y-%m-%d')
    if last_checked_day_utc_for_reset != current_day_str:
        logging.info(f"BACKTEST: New UTC Day: {current_day_str}. Resetting daily counters and risk budget.")
        daily_trade_counts = {day_sym: 0 for day_sym in SYMBOLS_TO_TRADE}
        
        daily_start_balance_utc = simulated_account_balance 
        daily_risk_budget_currency_global = daily_start_balance_utc * (DAILY_MAX_ACCOUNT_RISK_PERCENT / 100.0)
        current_daily_risked_amount_global = 0.0 
        daily_risk_budget_exceeded_today_global = False
        account_currency = "USD"
        if SYMBOLS_TO_TRADE and SYMBOL_PROPERTIES and SYMBOLS_TO_TRADE[0] in SYMBOL_PROPERTIES: # Check if list/dict are not empty and key exists
            account_currency = SYMBOL_PROPERTIES[SYMBOLS_TO_TRADE[0]]['currency_profit']
        
        logging.info(f"BACKTEST: Daily Start Balance: {daily_start_balance_utc:.2f} {account_currency}. Daily Risk Budget: {daily_risk_budget_currency_global:.2f} {account_currency}")
        last_checked_day_utc_for_reset = current_day_str

    if daily_risk_budget_exceeded_today_global:
        return
    if daily_risk_budget_currency_global > 0 and current_daily_risked_amount_global >= daily_risk_budget_currency_global:
        if not daily_risk_budget_exceeded_today_global:
            logging.warning(
                f"BACKTEST: Daily max risk budget of {DAILY_MAX_ACCOUNT_RISK_PERCENT:.2f}% reached. "
                f"Budget: {daily_risk_budget_currency_global:.2f}, Risked Today: {current_daily_risked_amount_global:.2f}. "
                f"No new trades for UTC day: {current_day_str}."
            )
            daily_risk_budget_exceeded_today_global = True
        return

    if len(simulated_open_positions) > 0:
        return

    current_hour_utc = simulated_current_time_utc.hour
    if not (TRADING_HOUR_START_UTC <= current_hour_utc < TRADING_HOUR_END_UTC):
        return

    for symbol in SYMBOLS_TO_TRADE:
        if symbol not in SYMBOL_PROPERTIES or symbol not in historical_data_m5 or symbol not in historical_data_h1:
            continue
        
        props = SYMBOL_PROPERTIES[symbol]
        in_session = False
        symbol_specific_sessions = SYMBOL_SESSIONS.get(symbol)
        if symbol_specific_sessions:
            for start_h, end_h in symbol_specific_sessions:
                if start_h <= current_hour_utc < end_h:
                    in_session = True; break
            if not in_session:
                continue
        
        if daily_trade_counts.get(symbol, 0) >= MAX_TRADES_PER_SYMBOL_PER_DAY:
            continue

        m5_signal, m5_prev_signal, m5_fractal_ref, h1_signal = get_latest_data_and_indicators_backtest(symbol, simulated_current_time_utc)

        if m5_signal is None or h1_signal is None or m5_prev_signal is None or m5_fractal_ref is None:
            continue
        
        adx_val = m5_signal.get(f'ADX_{ADX_PERIOD}')
        if pd.isna(adx_val) or adx_val < ADX_THRESHOLD: continue

        atr_val = m5_signal.get(f'ATR_{ATR_PERIOD}') 
        average_atr_val = m5_signal.get(f'ATR_{ATR_PERIOD}_SMA{ATR_ROLLING_MEAN_PERIOD}')
        if pd.isna(atr_val) or pd.isna(average_atr_val) or average_atr_val == 0: continue
        if not (average_atr_val * ATR_MULTIPLIER_LOW <= atr_val <= average_atr_val * ATR_MULTIPLIER_HIGH): continue

        h1_ema_short = h1_signal.get(f'H1_EMA_{H1_EMA_SHORT_PERIOD}')
        h1_ema_mid = h1_signal.get(f'H1_EMA_{H1_EMA_MID_PERIOD}')
        h1_ema_long = h1_signal.get(f'H1_EMA_{H1_EMA_LONG_PERIOD}')
        h1_rsi_val = h1_signal.get(f'H1_RSI_{H1_RSI_PERIOD}')
        h1_macd_hist = h1_signal.get("H1_MACD_HIST")
        if any(pd.isna(v) for v in [h1_ema_short, h1_ema_mid, h1_ema_long, h1_rsi_val, h1_macd_hist]): continue

        h1_is_uptrend = h1_ema_short > h1_ema_mid > h1_ema_long
        h1_is_downtrend = h1_ema_long > h1_ema_mid > h1_ema_short
        h1_filter_bullish = h1_is_uptrend and h1_rsi_val > H1_RSI_BULL_THRESHOLD and h1_macd_hist > 0
        h1_filter_bearish = h1_is_downtrend and h1_rsi_val < H1_RSI_BEAR_THRESHOLD and h1_macd_hist < 0

        m5_ema_short_val = m5_signal.get(f'EMA_{M5_EMA_SHORT_PERIOD}')
        m5_ema_mid_val = m5_signal.get(f'EMA_{M5_EMA_MID_PERIOD}')
        m5_ema_long_val = m5_signal.get(f'EMA_{M5_EMA_LONG_PERIOD}')
        m5_close_signal = m5_signal['close']
        m5_low_signal = m5_signal['low']; m5_high_signal = m5_signal['high']
        if any(pd.isna(v) for v in [m5_ema_short_val, m5_ema_mid_val, m5_ema_long_val]): continue

        m5_buy_frac = m5_fractal_ref.get('fractal_down', False)
        m5_sell_frac = m5_fractal_ref.get('fractal_up', False)

        order_type_to_place = None; sl_reference_price = 0.0

        m5_prev_close_val = m5_prev_signal['close']
        m5_prev_ema_short_val = m5_prev_signal.get(f'EMA_{M5_EMA_SHORT_PERIOD}')
        if pd.isna(m5_prev_ema_short_val): continue

        m5_is_uptrend_strong = m5_ema_short_val > m5_ema_mid_val > m5_ema_long_val
        m5_close_above_short_ema = m5_close_signal > m5_ema_short_val
        m5_prev_close_below_short_ema = m5_prev_close_val < m5_prev_ema_short_val
        m5_curr_low_below_short_ema = m5_low_signal < m5_ema_short_val
        m5_pullback_long_condition = m5_prev_close_below_short_ema or m5_curr_low_below_short_ema
        m5_close_above_long_ema_crit = m5_close_signal > m5_ema_long_val

        if m5_is_uptrend_strong and m5_pullback_long_condition and m5_close_above_short_ema and m5_buy_frac and m5_close_above_long_ema_crit and h1_filter_bullish:
            order_type_to_place = mt5.ORDER_TYPE_BUY
            sl_candidate1 = m5_ema_mid_val 
            sl_candidate2 = m5_ema_long_val
            sl_candidate3_fractal_low = m5_fractal_ref['low'] 
            if m5_low_signal >= m5_ema_mid_val : sl_reference_price = min(sl_candidate1, sl_candidate3_fractal_low)
            else: sl_reference_price = min(sl_candidate2, sl_candidate3_fractal_low)
            logging.info(f"BACKTEST: BUY Signal for {symbol} at {m5_signal.name}")

        if order_type_to_place is None:
            m5_is_downtrend_strong = m5_ema_long_val > m5_ema_mid_val > m5_ema_short_val
            m5_close_below_short_ema_sell = m5_close_signal < m5_ema_short_val
            m5_prev_close_above_short_ema_sell = m5_prev_close_val > m5_prev_ema_short_val
            m5_curr_high_above_short_ema_sell = m5_high_signal > m5_ema_short_val
            m5_pullback_short_condition = m5_prev_close_above_short_ema_sell or m5_curr_high_above_short_ema_sell
            m5_close_below_long_ema_crit = m5_close_signal < m5_ema_long_val

            if m5_is_downtrend_strong and m5_pullback_short_condition and m5_close_below_short_ema_sell and m5_sell_frac and m5_close_below_long_ema_crit and h1_filter_bearish:
                order_type_to_place = mt5.ORDER_TYPE_SELL
                sl_candidate1_sell = m5_ema_mid_val
                sl_candidate2_sell = m5_ema_long_val
                sl_candidate3_fractal_high = m5_fractal_ref['high'] 
                if m5_high_signal <= m5_ema_mid_val: sl_reference_price = max(sl_candidate1_sell, sl_candidate3_fractal_high)
                else: sl_reference_price = max(sl_candidate2_sell, sl_candidate3_fractal_high)
                logging.info(f"BACKTEST: SELL Signal for {symbol} at {m5_signal.name}")

        if order_type_to_place is not None:
            logging.info(f"BACKTEST: CONFIRMED TRADE SIGNAL: {'BUY' if order_type_to_place == mt5.ORDER_TYPE_BUY else 'SELL'} for {symbol} at {simulated_current_time_utc}")
            point = props['point']; digits = props['digits']
            
            entry_price_now = historical_data_m5[symbol].loc[simulated_current_time_utc]['open']
            if entry_price_now == 0:
                 logging.warning(f"BACKTEST: Entry price for {symbol} at {simulated_current_time_utc} is 0. Skipping.")
                 continue

            sl_pips_calculated = 0.0
            if order_type_to_place == mt5.ORDER_TYPE_BUY:
                if sl_reference_price < entry_price_now: sl_pips_calculated = (entry_price_now - sl_reference_price) / point if point > 0 else 0
            else: 
                if sl_reference_price > entry_price_now: sl_pips_calculated = (sl_reference_price - entry_price_now) / point if point > 0 else 0
            
            if sl_pips_calculated == 0 and atr_val > 0 and point > 0 :
                sl_pips_calculated = atr_val / point 

            min_sl_pips_atr_factor = 1.0; min_sl_pips_absolute = 10.0
            if "JPY" in symbol.upper(): min_sl_pips_absolute = 10.0
            elif any(metal in symbol.upper() for metal in ["XAU", "XAG", "XPT"]): min_sl_pips_absolute = 100.0
            elif "OIL" in symbol.upper() : min_sl_pips_absolute = 100.0
            min_sl_pips_from_atr = (atr_val / point * min_sl_pips_atr_factor) if point > 0 and atr_val > 0 else min_sl_pips_absolute
            sl_pips_final = max(sl_pips_calculated, min_sl_pips_from_atr, min_sl_pips_absolute)

            if sl_pips_final <= 0:
                logging.warning(f"BACKTEST: {symbol} final SL pips {sl_pips_final:.2f} is invalid. Skipping.")
                continue

            sl_price_final, tp_price_final = (0.0,0.0)
            if order_type_to_place == mt5.ORDER_TYPE_BUY:
                sl_price_final = entry_price_now - (sl_pips_final * point)
                tp_price_final = entry_price_now + (sl_pips_final * RISK_REWARD_RATIO * point)
            else: 
                sl_price_final = entry_price_now + (sl_pips_final * point)
                tp_price_final = entry_price_now - (sl_pips_final * RISK_REWARD_RATIO * point)

            if (order_type_to_place == mt5.ORDER_TYPE_BUY and (sl_price_final >= entry_price_now or tp_price_final <= entry_price_now)) or \
               (order_type_to_place == mt5.ORDER_TYPE_SELL and (sl_price_final <= entry_price_now or tp_price_final >= entry_price_now)):
                logging.warning(f"BACKTEST: Invalid SL/TP for {symbol}. E:{entry_price_now}, SL:{sl_price_final}, TP:{tp_price_final}. Skipping.")
                continue

            current_sim_balance = simulated_account_balance 
            lot_size_final = calculate_dynamic_lot_size(symbol, sl_pips_final, current_sim_balance, RISK_PER_TRADE_ACCOUNT_PERCENT, entry_price_now)
            
            if lot_size_final < props['volume_min'] or lot_size_final == 0:
                 lot_size_final = props['volume_min']
            if lot_size_final == 0:
                logging.error(f"BACKTEST: Final lot size for {symbol} is 0. Cannot place trade.")
                continue

            value_of_one_point_trade = get_value_of_one_point_generic(symbol, entry_price_now)
            if value_of_one_point_trade is None or value_of_one_point_trade <= 0:
                logging.error(f"BACKTEST: Could not get valid point value for {symbol} for risk calc. Skipping.")
                continue
            risked_amount_this_trade = lot_size_final * sl_pips_final * value_of_one_point_trade

            if daily_risk_budget_currency_global > 0 and \
               (current_daily_risked_amount_global + risked_amount_this_trade > daily_risk_budget_currency_global):
                logging.info(f"BACKTEST: Trade for {symbol} (risk: {risked_amount_this_trade:.2f}) would exceed daily budget. Skipping.")
                continue 
            
            bot_version_comment = "V1.5BT"
            sl_pips_comment_str = f"{sl_pips_final:.0f}"
            rr_comment_str = f"{RISK_REWARD_RATIO:.0f}" if RISK_REWARD_RATIO == int(RISK_REWARD_RATIO) else f"{RISK_REWARD_RATIO:.1f}"
            lot_precision_comment = 0
            if props['volume_step'] > 0 and props['volume_step'] < 1:
                step_str_comm = format(props['volume_step'], '.8f').rstrip('0')
                if '.' in step_str_comm: lot_precision_comment = len(step_str_comm.split('.')[1])
            lot_comment_str = f"{lot_size_final:.{lot_precision_comment}f}"
            trade_comment_str = f"{bot_version_comment};SL{sl_pips_comment_str};R{rr_comment_str};L{lot_comment_str}"
            MAX_COMMENT_LENGTH = 31 
            if len(trade_comment_str) > MAX_COMMENT_LENGTH:
                trade_comment_str = f"SL{sl_pips_comment_str};R{rr_comment_str};L{lot_comment_str}"
                if len(trade_comment_str) > MAX_COMMENT_LENGTH:
                     trade_comment_str = f"SL{sl_pips_comment_str};R{rr_comment_str}"
                     if len(trade_comment_str) > MAX_COMMENT_LENGTH: trade_comment_str = trade_comment_str[:MAX_COMMENT_LENGTH]

            logging.info(f"BACKTEST: Attempting to place {'BUY' if order_type_to_place == mt5.ORDER_TYPE_BUY else 'SELL'} for {symbol}: "
                         f"Lots: {lot_size_final}, Entry: {entry_price_now:.{digits}f}, SL: {sl_price_final:.{digits}f}, "
                         f"TP: {tp_price_final:.{digits}f}, Risk: {risked_amount_this_trade:.2f}")

            sim_order_result, sim_position_id = simulate_place_trade_order(
                symbol, order_type_to_place, lot_size_final, entry_price_now, sl_price_final, tp_price_final,
                simulated_current_time_utc, 
                trade_comment_str, risked_amount_this_trade
            )

            if sim_order_result['retcode'] == mt5.TRADE_RETCODE_DONE:
                daily_trade_counts[symbol] = daily_trade_counts.get(symbol, 0) + 1
                current_daily_risked_amount_global += risked_amount_this_trade
                logging.info(f"BACKTEST: Trade execution successful for {symbol}. Position ID: {sim_position_id}. "
                             f"Daily count for {symbol}: {daily_trade_counts[symbol]}. "
                             f"Total daily risked: {current_daily_risked_amount_global:.2f}/{daily_risk_budget_currency_global:.2f}.")
                return 
            else:
                logging.error(f"BACKTEST: Failed to simulate trade for {symbol}.")


def run_backtest():
    logging.info("Initializing Backtester...")
    global simulated_account_balance, simulated_current_time_utc, last_checked_day_utc_for_reset
    global daily_start_balance_utc, daily_trade_counts, SYMBOLS_TO_TRADE 
    global daily_risk_budget_currency_global, current_daily_risked_amount_global, daily_risk_budget_exceeded_today_global


    if not initialize_mt5_for_properties(): 
        logging.critical("Failed to initialize MT5 for properties. Backtester cannot start.")
        return

    load_or_initialize_trade_history() 

    get_historical_data_for_all_symbols(list(SYMBOLS_TO_TRADE_INITIAL), BACKTEST_START_DATE, BACKTEST_END_DATE)

    if not SYMBOLS_TO_TRADE or not historical_data_m5: # SYMBOLS_TO_TRADE is now updated by get_historical_data
        logging.critical("No symbols with sufficient data loaded for backtest. Exiting.")
        ti = mt5.terminal_info()
        if ti and ti.connected: 
            mt5.shutdown()
        return
    
    main_iteration_symbol = None
    for sym_candidate in SYMBOLS_TO_TRADE: # Iterate over the filtered list
        if sym_candidate in historical_data_m5 and not historical_data_m5[sym_candidate].empty:
            main_iteration_symbol = sym_candidate
            break
    
    if not main_iteration_symbol:
        logging.critical("No symbol found with M5 data to drive backtest iteration after filtering. Exiting.")
        ti = mt5.terminal_info()
        if ti and ti.connected: 
            mt5.shutdown()
        return
        
    logging.info(f"Using {main_iteration_symbol} M5 data as the primary time iterator for backtest.")
    
    main_m5_df = historical_data_m5[main_iteration_symbol]
    main_m5_df_filtered = main_m5_df[(main_m5_df.index >= BACKTEST_START_DATE) & (main_m5_df.index <= BACKTEST_END_DATE)]

    if main_m5_df_filtered.empty:
        logging.critical(f"No M5 data for primary symbol {main_iteration_symbol} within the specified backtest range "
                         f"{BACKTEST_START_DATE} to {BACKTEST_END_DATE}. Exiting.")
        ti = mt5.terminal_info()
        if ti and ti.connected: 
            mt5.shutdown()
        return

    simulated_current_time_utc = main_m5_df_filtered.index[0]
    last_checked_day_utc_for_reset = (simulated_current_time_utc - datetime.timedelta(days=1)).strftime('%Y-%m-%d') 
    daily_start_balance_utc = INITIAL_BACKTEST_BALANCE 
    daily_trade_counts = {symbol: 0 for symbol in SYMBOLS_TO_TRADE}
    simulated_account_balance = INITIAL_BACKTEST_BALANCE 
    
    logging.info(f"Starting backtest from {main_m5_df_filtered.index[0]} to {main_m5_df_filtered.index[-1]}...")
    total_bars = len(main_m5_df_filtered)

    for i, current_bar_timestamp in enumerate(main_m5_df_filtered.index):
        simulated_current_time_utc = current_bar_timestamp 
        
        if i % 1000 == 0: 
            logging.info(f"Backtesting progress: {i+1}/{total_bars} bars. Current time: {simulated_current_time_utc}. Balance: {simulated_account_balance:.2f}")

        current_m5_bar_data_for_all_symbols = {}
        for sym in SYMBOLS_TO_TRADE:
            if sym in historical_data_m5:
                sym_df_m5 = historical_data_m5[sym]
                if simulated_current_time_utc in sym_df_m5.index:
                     current_m5_bar_data_for_all_symbols[sym] = sym_df_m5.loc[simulated_current_time_utc]

        if simulated_open_positions: 
            simulate_manage_open_positions(current_m5_bar_data_for_all_symbols)

        if i > 0 : 
            check_for_new_trade_signals_backtest() 

    logging.info("Backtest loop finished.")
    logging.info(f"Final Simulated Balance: {simulated_account_balance:.2f}")
    save_trade_history() 
    ti = mt5.terminal_info()
    if ti and ti.connected:
        mt5.shutdown()
    logging.info("Backtest complete.")


# --- LIVE TRADING MAIN LOOP (Original, for reference or if BACKTEST_MODE is False) ---
def main_bot_loop_live():
    logging.info("Initializing Trading Bot for LIVE Trading...")
    global last_checked_day_utc_for_reset, daily_start_balance_utc, daily_trade_counts, SYMBOLS_TO_TRADE
    global daily_risk_budget_currency_global, current_daily_risked_amount_global, daily_risk_budget_exceeded_today_global

    if not initialize_mt5_for_properties(): 
        logging.critical("Failed to initialize MT5 for live. Bot cannot start.")
        return

    load_or_initialize_trade_history()

    now_utc_init = datetime.datetime.now(datetime.timezone.utc)
    last_checked_day_utc_for_reset = now_utc_init.strftime('%Y-%m-%d')
    daily_trade_counts = {symbol: 0 for symbol in SYMBOLS_TO_TRADE}
    
    acc_info_live = mt5.account_info() # For live, directly use account_info
    initial_bal = acc_info_live.balance if acc_info_live else None
    account_currency_main = acc_info_live.currency if acc_info_live else "N/A"


    if initial_bal is not None and initial_bal > 0:
        daily_start_balance_utc = initial_bal
        daily_risk_budget_currency_global = daily_start_balance_utc * (DAILY_MAX_ACCOUNT_RISK_PERCENT / 100.0)
        current_daily_risked_amount_global = 0.0
        daily_risk_budget_exceeded_today_global = False
        logging.info(f"Bot Started (Live). Initial Daily Start Balance: {daily_start_balance_utc:.2f} {account_currency_main}. "
                     f"Daily Risk Budget: {daily_risk_budget_currency_global:.2f} {account_currency_main}.")
    else:
        logging.critical("CRITICAL (Live): Could not get initial account balance or balance is zero! Daily risk budget is 0.")
        daily_start_balance_utc = 0; daily_risk_budget_currency_global = 0.0; current_daily_risked_amount_global = 0.0
        daily_risk_budget_exceeded_today_global = True

    last_signal_check_minute = -1
    last_closed_trade_check_time = time.time()

    logging.info("Bot Initialized (Live). Starting Trading Loop...")
    try:
        while True:
            term_info_live = mt5.terminal_info()
            if not (term_info_live and term_info_live.connected):
                logging.error("MT5 terminal disconnected. Attempting to reconnect...")
                if not initialize_mt5_for_properties(): 
                    logging.error("Failed to reconnect MT5. Sleeping for 60s.")
                    time.sleep(60)
                    continue
                else:
                    logging.info("MT5 reconnected.")

            current_time_utc = datetime.datetime.now(datetime.timezone.utc)
            # Placeholder for live versions of these functions
            # manage_trailing_stop_loss_live() 
            # if time.time() - last_closed_trade_check_time > 30:
            #     check_and_log_closed_trades() 
            #     last_closed_trade_check_time = time.time()

            if current_time_utc.minute % 5 == 0 and current_time_utc.minute != last_signal_check_minute:
                if current_time_utc.second < 15 :
                    logging.info(f"--- Scheduled Signal Check (Live) at {current_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
                    # check_for_new_trade_signals_live() # Needs to be defined for live
                    last_signal_check_minute = current_time_utc.minute
            elif current_time_utc.minute != last_signal_check_minute :
                 last_signal_check_minute = -1
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Bot shutdown requested by user (Ctrl+C).")
    except Exception as e:
        logging.exception("Critical error in main trading loop (Live):")
    finally:
        logging.info("Bot shutting down (Live). Performing final check for closed trades...")
        # check_and_log_closed_trades() 
        term_info_final = mt5.terminal_info()
        if term_info_final and term_info_final.connected: 
            mt5.shutdown()
        logging.info("Bot shutdown complete (Live).")


if __name__ == "__main__":
    if BACKTEST_MODE:
        run_backtest()
    else:
        logging.error("Live trading mode (main_bot_loop_live) is currently a placeholder. This script is set up for BACKTEST_MODE.") 
        logging.info("To run the original live bot, please use the previous version of the script and set BACKTEST_MODE=False.")