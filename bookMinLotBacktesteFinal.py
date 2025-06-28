import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta # For EMAs and ATR
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import math
import matplotlib.pyplot as plt # Added for plotting
import csv # ✅ Step 1: Import the Required Modules
import os # ✅ Step 1: Import the Required Modules

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}
RUN_BACKTEST = True # Important for the provided MT5 init function

# ✅ Step 2: 🧾 Define the CSV File & Headers
TRADE_HISTORY_FILE = "backtest_trade_history.csv"
CSV_HEADERS = [
    "Symbol", "Type", "EntryTimeUTC", "EntryPrice",
    "LotSize", "SL", "TP",
    "ExitTimeUTC", "ExitPrice", "Status",
    "Commission", "PnL_Currency", "BalanceAfterTrade"
]

# --- Strategy & Backtest Parameters ---
SYMBOLS_TO_BACKTEST = ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD", 
                           "AUDJPY",   "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF", "CADCHF", "CADJPY", "EURCAD",
                           "GBPCAD", "NZDCAD", "GBPAUD", "GBPNZD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD", "EURAUD", 
                       "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD","AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500", 
                       "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C",

                             ]


# "NVDA",  "AAPL", "AMD", "AMZN", "GOOGL" stocks to trade

TRADING_SESSIONS_UTC = { # (start_hour_inclusive, end_hour_exclusive)
                           "EURUSD":[(7, 16)], "USDCHF":[(7, 16)],   "GBPJPY": [ (7, 16)], "GBPUSD": [ (7, 16)], 
                           "AUDJPY":[(0, 4)],  "XAUUSD": [(7, 16)], "XAGUSD": [(7, 16)], "EURNZD": [(7, 16)], "NZDUSD": [(7, 16)], "AUDUSD": [ (7, 16)], "USDCAD": [(12, 16)],"USDJPY":[(0,4), (12, 16)], "EURJPY": [ (0,4) , (7, 16)],"EURCHF": [(7, 16)], "CADCHF": [  (7, 16)], "CADJPY": [ (0,4) , (12, 16)], "EURCAD":[(7, 16)],
                           "GBPCAD": [(7, 16)], "NZDCAD":[(12,16)], "GBPAUD":[(0,4), (7, 16)], "GBPNZD":[(7,16)], "GBPCHF":[(7,16)], "AUDCAD":[(0,4) , (12,16)], "AUDCHF":[(0,4) , (7,16)], "AUDNZD":[(0,4)], "EURAUD":[(0,4) , (7,16)], 
                           "AAPL": [(14, 17)] , "MSFT": [(14, 17)], "GOOGL": [(14, 17)], "AMZN": [(14, 17)], "NVDA": [(14, 17)], "META": [(14, 17)], "TSLA": [(14, 17)], "AMD": [(14, 17)], "NFLX": [(14, 17)], "US500": [(14, 17)], 
                           "USTEC": [(14, 17)],"INTC":[(14, 17)], "MO":[(14, 17)], "BABA":[(14, 17)], "ABT":[(14, 17)], "LI":[(14, 17)], "TME":[(14, 17)], "ADBE":[(14, 17)], "MMM":[(14, 17)], "WMT":[(14, 17)], "PFE":[(14, 17)], "EQIX":[(14, 17)], "F":[(14, 17)], "ORCL":[(14, 17)], "BA":[(14, 17)], "NKE":[(14, 17)], "C":[(14, 17)],
                          
                          }


TRADING_SESSIONS_UTC["USOIL"] = [(12, 16)]
TRADING_SESSIONS_UTC["UKOIL"] = [(7, 16)]
CRYPTO_SESSIONS_USER = {"BTCUSD":[(7, 16)], "BTCJPY":[(0, 16)], "BTCXAU":[(7, 16)], "ETHUSD":[(7, 16)]}
for crypto_sym, sess_val in CRYPTO_SESSIONS_USER.items():
    TRADING_SESSIONS_UTC[crypto_sym] = sess_val


INITIAL_ACCOUNT_BALANCE = 200.00
RISK_PER_TRADE_PERCENT = 0.01 # Risk 1% of current balance per trade
DAILY_RISK_LIMIT_PERCENT = 0.05 # Daily risk limit of 5% of balance at start of day

# --- NEW: Simulation Parameters ---
SPREAD_PIPS = 0.2  # Simulate a 1.5 pip spread
SLIPPAGE_PIPS = 0.5 # Simulate 0.5 pips of slippage on entry

# --- NEW: Commission Structure ---
# This dictionary holds the commission cost per trade for the minimum lot size.
# The bot currently only trades the minimum lot, so this value is applied directly.






COMMISSIONS = {

    "EURUSD":0.07, "USDCHF":0.10,   "GBPJPY":0.15, "GBPUSD":0.09, 
                           "AUDJPY":0.09,   "EURNZD":0.18, "NZDUSD":0.13, "AUDUSD":0.10, "USDCAD":0.10,"USDJPY":0.07, "EURJPY":0.11,"EURCHF":0.17, "CADCHF":0.09, "CADJPY":0.15, "EURCAD":0.13,
                           "GBPCAD":0.20, "NZDCAD":0.10, "GBPAUD":0.13, "GBPNZD":0.19, "GBPCHF":0.17, "AUDCAD":0.10, "AUDCHF":0.09, "AUDNZD":0.08, "EURAUD":0.13, 
                       "USOIL":0.16, "UKOIL":0.65, "XAUUSD":0.11, "XAGUSD":1.05,
                       "BTCUSD":0.16, "BTCJPY":0.21, "BTCXAU":0.20, "ETHUSD":0.30,"AAPL":0.05, "MSFT":0.17, 
                        "AAPL": 0.05, "MSFT":0.17, "GOOGL": 0.11, "AMZN": 0.05, "NVDA": 0.08, "META": 0.33, "TSLA": 0.07, 
                        "AMD":0.14, "NFLX":0.98 , "US500":0.03 , 
                        "USTEC":0.03,"INTC":0.07, "MO":0.05, "BABA":0.13, "ABT":0.08, "LI":0.04, "TME":0.05, "ADBE":0.20, "MMM":0.10, "WMT":0.08, "PFE":0.07, "EQIX":0.87, "F":0.09, "ORCL":0.17, "BA":0.33, "NKE":0.10, "C":0.07,
                      

                            }

# ✅ Step 3: 📌 Create a Function to Initialize the File
def initialize_trade_history_file():
    if not os.path.exists(TRADE_HISTORY_FILE):
        with open(TRADE_HISTORY_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)

# ✅ Step 4: ✍️ Create a Function to Log Each Trade
def log_backtest_trade_to_csv(trade):
    with open(TRADE_HISTORY_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            trade.get("symbol"),
            trade.get("type"),
            trade.get("entry_time"),
            trade.get("entry_price"),
            trade.get("lot_size"),
            trade.get("initial_sl"),
            trade.get("tp"),
            trade.get("exit_time"),
            trade.get("exit_price"),
            trade.get("status"),
            trade.get("commission"),
            trade.get("pnl_currency"),
            trade.get("balance_after_trade")
        ])

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
        if not RUN_BACKTEST: mt5.shutdown(); return False
        logger.warning("Could not get account info. Proceeding for backtest data access.")
    else:
        logger.info(f"Account Login: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")

    successfully_initialized_symbols = []
    temp_symbol_properties = {}
    for symbol_name in symbols_to_check:
        symbol_info_obj = mt5.symbol_info(symbol_name)
        if symbol_info_obj is None: logger.warning(f"Symbol {symbol_name} not found in MT5. Skipping."); continue
        if not symbol_info_obj.visible and not RUN_BACKTEST:
            logger.info(f"Symbol {symbol_name} not visible, attempting to select.")
            if not mt5.symbol_select(symbol_name, True): logger.warning(f"symbol_select({symbol_name}) failed. Skipping."); continue
            time.sleep(0.2); symbol_info_obj = mt5.symbol_info(symbol_name)
            if symbol_info_obj is None or not symbol_info_obj.visible: logger.warning(f"Symbol {symbol_name} still not available. Skipping."); continue

        if symbol_info_obj.point == 0: logger.warning(f"Symbol {symbol_name} point value is 0. Skipping."); continue
        if symbol_info_obj.trade_tick_size == 0: logger.warning(f"Symbol {symbol_name} trade_tick_size is 0. Skipping."); continue

        spread_points = symbol_info_obj.spread
        pip_value_std = 0.0001; pip_value_jpy = 0.01
        current_pip_value = pip_value_jpy if 'JPY' in symbol_name.upper() else pip_value_std

        if symbol_name.upper() in ["XAUUSD", "XAGUSD", "XPTUSD"]: current_pip_value = 0.01
        elif "OIL" in symbol_name.upper() or "USOIL" in symbol_name.upper() or "UKOIL" in symbol_name.upper(): current_pip_value = 0.01
        elif "BTC" in symbol_name.upper() or "ETH" in symbol_name.upper(): current_pip_value = 0.01

        spread_pips = (spread_points * symbol_info_obj.point) / current_pip_value if current_pip_value > 0 else 0

        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point, 'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
            'trade_tick_value': symbol_info_obj.trade_tick_value,
            'volume_min': symbol_info_obj.volume_min,
            'volume_step': symbol_info_obj.volume_step if symbol_info_obj.volume_step > 0 else 0.01,
            'volume_max': symbol_info_obj.volume_max,
            'trade_contract_size': symbol_info_obj.trade_contract_size,
            'spread_points': spread_points,
            'currency_profit': symbol_info_obj.currency_profit,
            'currency_margin': symbol_info_obj.currency_margin,
            'pip_value_calc': current_pip_value, # This is the value of one "pip" (e.g. 0.0001 for EURUSD)
            'spread_pips_on_init': spread_pips
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

# --- Helper Functions ---
def is_within_session(candle_time_utc, symbol_sessions):
    if not symbol_sessions: return True
    candle_hour = candle_time_utc.hour
    for start_hour, end_hour in symbol_sessions:
        if start_hour <= candle_hour < end_hour: return True
    return False

# ✅ Step 1: Define the Pullback Measurement Logic
def calculate_pullback_depth(impulse_start, impulse_end, current_price, trade_type):
    total_leg = abs(impulse_end - impulse_start)
    if total_leg == 0:
        return 0

    if trade_type == "BUY":
        pullback = impulse_end - current_price
    else:  # SELL
        pullback = current_price - impulse_end

    return max(0.0, pullback / total_leg)  # returns a % (e.g., 0.32 = 32%)

def calculate_fib_levels(swing_high, swing_low):
    fibs = {
        "0.0": swing_low,
        "0.236": swing_low + 0.236 * (swing_high - swing_low),
        "0.382": swing_low + 0.382 * (swing_high - swing_low),
        "0.5": swing_low + 0.5 * (swing_high - swing_low),
        "0.618": swing_low + 0.618 * (swing_high - swing_low),
        "0.786": swing_low + 0.786 * (swing_high - swing_low),
        "1.0": swing_high
    }
    return fibs

def calculate_lot_size(account_balance_for_risk_calc, risk_percent, sl_price_diff, symbol_props):
    if sl_price_diff <= 0: return 0
    risk_amount_currency = account_balance_for_risk_calc * risk_percent
    if symbol_props['trade_tick_size'] == 0 or symbol_props['trade_tick_value'] == 0: return 0
    sl_distance_ticks = sl_price_diff / symbol_props['trade_tick_size']
    sl_cost_per_lot = sl_distance_ticks * symbol_props['trade_tick_value']
    if sl_cost_per_lot <= 0: return 0
    lot_size = risk_amount_currency / sl_cost_per_lot
    lot_size = max(symbol_props['volume_min'], lot_size)
    lot_size = math.floor(lot_size / symbol_props['volume_step']) * symbol_props['volume_step']
    lot_size = min(symbol_props['volume_max'], lot_size)
    if lot_size < symbol_props['volume_min']:
        if symbol_props['volume_min'] * sl_cost_per_lot > risk_amount_currency * 1.5:
            return 0
        return symbol_props['volume_min']
    return round(lot_size, int(-math.log10(symbol_props['volume_step'])) if symbol_props['volume_step'] > 0 else 2)


# --- Performance Statistics Calculation ---
def calculate_performance_stats(trades_list, initial_balance_for_period):
    stats = {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "win_rate": 0,
        "gross_profit": 0.0, "gross_loss": 0.0, "net_profit": 0.0, "profit_factor": 0.0,
        "max_drawdown_abs": 0.0, "max_drawdown_pct": 0.0,
        "start_balance": float(initial_balance_for_period),
        "end_balance": float(initial_balance_for_period)
    }
    if not trades_list:
        return stats

    trades_df = pd.DataFrame(trades_list)
    if not trades_df.empty and 'entry_time' in trades_df.columns:
        trades_df.sort_values(by='entry_time', inplace=True)

    pnl_series = trades_df['pnl_currency'] if 'pnl_currency' in trades_df else pd.Series(dtype=float)


    stats["total_trades"] = len(trades_df)
    stats["winning_trades"] = len(trades_df[pnl_series > 0]) if not pnl_series.empty else 0
    stats["losing_trades"] = len(trades_df[pnl_series < 0]) if not pnl_series.empty else 0
    stats["win_rate"] = (stats["winning_trades"] / stats["total_trades"]) * 100 if stats["total_trades"] > 0 else 0

    stats["gross_profit"] = pnl_series[pnl_series > 0].sum() if not pnl_series.empty else 0.0
    stats["gross_loss"] = abs(pnl_series[pnl_series < 0].sum()) if not pnl_series.empty else 0.0
    stats["net_profit"] = pnl_series.sum() if not pnl_series.empty else 0.0
    stats["end_balance"] = initial_balance_for_period + stats["net_profit"]

    if stats["gross_loss"] > 0:
        stats["profit_factor"] = stats["gross_profit"] / stats["gross_loss"]
    elif stats["gross_profit"] > 0:
        stats["profit_factor"] = float('inf')
    else:
        stats["profit_factor"] = 0.0

    equity_curve = [initial_balance_for_period]
    current_equity = initial_balance_for_period
    if not pnl_series.empty:
        for pnl in pnl_series:
            current_equity += pnl
            equity_curve.append(current_equity)

    equity_series = pd.Series(equity_curve)
    rolling_max_equity = equity_series.cummax()
    absolute_drawdowns = equity_series - rolling_max_equity
    stats["max_drawdown_abs"] = abs(absolute_drawdowns.min()) if not absolute_drawdowns.empty else 0
    if stats["max_drawdown_abs"] > 0 and not rolling_max_equity.empty:
        mdd_end_index = absolute_drawdowns.idxmin()
        if mdd_end_index < len(rolling_max_equity) and mdd_end_index > 0:
            peak_at_mdd_start = rolling_max_equity.iloc[mdd_end_index]
            if peak_at_mdd_start > 0 :
                 stats["max_drawdown_pct"] = (stats["max_drawdown_abs"] / peak_at_mdd_start) * 100
            else:
                 stats["max_drawdown_pct"] = 0.0
        elif mdd_end_index == 0 and initial_balance_for_period > 0 :
            stats["max_drawdown_pct"] = (stats["max_drawdown_abs"] / initial_balance_for_period) * 100
        else:
            stats["max_drawdown_pct"] = 0.0
    else:
        stats["max_drawdown_pct"] = 0.0
    return stats

# --- FIX APPLIED IN THIS FUNCTION ---
def analyze_rr_distribution(closed_trades, symbol_properties_dict):
    """
    Analyzes the distribution of closed trades based on their Risk-to-Reward ratio.

    Args:
        closed_trades (list): A list of closed trade dictionaries.
        symbol_properties_dict (dict): A dictionary containing properties for all symbols.

    Returns:
        dict: A dictionary with RR buckets as keys and trade counts as values.
    """
    # CORRECTED: The dictionary keys now exactly match the strings used in the logic below.
    rr_buckets = {
        "Stop Loss (~ -1R)": 0,
        "Partial Loss (< 0R)": 0,
        "Break Even (0R to <1R>)": 0,
        "1R to <2R>": 0,
        "2R to <3R>": 0,
        "3R to <4R>": 0,
        "Take Profit (>= 4R)": 0,
        "Other/Error": 0
    }

    if not closed_trades:
        return rr_buckets

    for trade in closed_trades:
        symbol = trade.get('symbol')
        props = symbol_properties_dict.get(symbol)
        if not props:
            rr_buckets["Other/Error"] += 1
            continue

        pnl = trade.get('pnl_currency', 0.0)
        # Use 'initial_sl' to ensure we calculate against the original risk
        initial_risk_price_diff = abs(trade.get('entry_price', 0) - trade.get('initial_sl', 0))

        if initial_risk_price_diff <= 0 or props.get('trade_tick_size', 0) <= 0:
            rr_buckets["Other/Error"] += 1
            continue

        initial_risk_currency = (initial_risk_price_diff / props['trade_tick_size']) * props['trade_tick_value'] * trade.get('lot_size', 0)

        if initial_risk_currency <= 0:
            rr_buckets["Other/Error"] += 1
            continue

        rr_value = pnl / initial_risk_currency

        # Correctly categorize the RR value into buckets
        if rr_value >= 4.0:
            rr_buckets["Take Profit (>= 4R)"] += 1
        elif 3.0 <= rr_value < 4.0:
            rr_buckets["3R to <4R>"] += 1
        elif 2.0 <= rr_value < 3.0:
            rr_buckets["2R to <3R>"] += 1
        elif 1.0 <= rr_value < 2.0:
            rr_buckets["1R to <2R>"] += 1
        elif 0.0 <= rr_value < 1.0:
            rr_buckets["Break Even (0R to <1R>)"] += 1
        elif abs(rr_value + 1.0) < 0.05: # If RR is very close to -1
            rr_buckets["Stop Loss (~ -1R)"] += 1
        else: # Any other loss that wasn't a clean stop out (e.g., trailing stop hit for a loss)
            rr_buckets["Partial Loss (< 0R)"] += 1

    return rr_buckets

# --- Data Fetching and Preparation ---
def get_historical_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logger.warning(f"No data for {symbol} {timeframe_to_string(timeframe)} from {start_date} to {end_date}. Err: {mt5.last_error()}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def timeframe_to_string(tf_enum):
    map_tf = { mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
               mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
               mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1" }
    return map_tf.get(tf_enum, "UnknownTF")

def prepare_symbol_data(symbol, start_date, end_date, symbol_props):
    logger.info(f"Preparing data for {symbol} from {start_date} to {end_date}")

    # --- H1 Data ---
    df_h1 = get_historical_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    # Prepare H1 EMAs and Close
    if df_h1.empty:
        logger.warning(f"No H1 data for {symbol} (EMAs/Close).")
        df_h1_resampled_emas = pd.DataFrame(columns=['H1_Close_For_Bias', 'H1_EMA8', 'H1_EMA21'])
    else:
        df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
        df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
        df_h1_resampled_emas = df_h1[['close', 'H1_EMA8', 'H1_EMA21']].rename(columns={'close': 'H1_Close_For_Bias'})

    # Prepare H1 RSI
    if df_h1.empty: # df_h1 is the same as above
        logger.warning(f"No H1 data for {symbol} (RSI).")
        df_h1_resampled_rsi = pd.DataFrame(columns=['RSI_H1'])
        df_h1_resampled_rsi['RSI_H1'] = np.nan
    else:
        df_h1['RSI_H1'] = ta.rsi(df_h1['close'], length=14) # 🧮 For H1 RSI (User Step 1)
        df_h1_resampled_rsi = df_h1[['RSI_H1']].copy() # This is df_h1_resampled['RSI_H1'] = df_h1['RSI_H1'] implicitly

    # --- H4 Data ---
    df_h4 = get_historical_data(symbol, mt5.TIMEFRAME_H4, start_date, end_date)
    h4_data_available = not df_h4.empty
    # Prepare H4 EMAs
    if h4_data_available:
        df_h4['H4_EMA8'] = ta.ema(df_h4['close'], length=8)
        df_h4['H4_EMA21'] = ta.ema(df_h4['close'], length=21)
        df_h4_resampled_emas = df_h4[['H4_EMA8', 'H4_EMA21']].copy()
    else:
        logger.warning(f"No H4 data for {symbol} (EMAs).")
        df_h4_resampled_emas = pd.DataFrame(columns=['H4_EMA8', 'H4_EMA21'])
    # Prepare H4 RSI
    if h4_data_available: # df_h4 is the same as above
        df_h4['RSI_H4'] = ta.rsi(df_h4['close'], length=14) # 🧮 For H4 RSI (User Step 1)
        df_h4_resampled_rsi = df_h4[['RSI_H4']].copy() # This is df_h4_resampled['RSI_H4'] = df_h4['RSI_H4'] implicitly
    else:
        logger.warning(f"No H4 data for {symbol} (RSI).")
        df_h4_resampled_rsi = pd.DataFrame(columns=['RSI_H4'])
        df_h4_resampled_rsi['RSI_H4'] = np.nan

    # --- M5 Data --- (MODIFIED from M1 to M5)
    df_m5 = get_historical_data(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    if df_m5.empty: return pd.DataFrame()
    df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
    df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
    df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
    if len(df_m5) >= 14:
        df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
    else:
        df_m5['ATR'] = np.nan
    df_m5['RSI_M5'] = ta.rsi(df_m5['close'], length=14) # For M5 RSI

    # Initial merge for M5 data + H1 EMAs/Close
    combined_df = pd.merge_asof(df_m5.sort_index(), df_h1_resampled_emas.sort_index(),
                                left_index=True, right_index=True,
                                direction='backward', tolerance=pd.Timedelta(hours=1))

    # Merge H4 EMAs
    if h4_data_available:
        combined_df = pd.merge_asof(combined_df.sort_index(), df_h4_resampled_emas.sort_index(),
                                    left_index=True, right_index=True,
                                    direction='backward', tolerance=pd.Timedelta(hours=4))
    else: # If H4 EMAs data was not available
        combined_df['H4_EMA8'] = np.nan
        combined_df['H4_EMA21'] = np.nan

    # 🔁 STEP 2: Merge RSI values into combined_df
    # Merge H1 RSI
    combined_df = pd.merge_asof(combined_df.sort_index(), df_h1_resampled_rsi.sort_index(),
                                left_index=True, right_index=True,
                                direction='backward', tolerance=pd.Timedelta(hours=1))
    # Merge H4 RSI
    combined_df = pd.merge_asof(combined_df.sort_index(), df_h4_resampled_rsi.sort_index(),
                                left_index=True, right_index=True,
                                direction='backward', tolerance=pd.Timedelta(hours=4))

    # Note: If df_h1 was empty, df_h1_resampled_rsi has 'RSI_H1' as NaN. merge_asof will propagate these NaNs. Same for H4.
    # This is the desired behavior. The dropna below will handle rows with insufficient data.

    combined_df.dropna(inplace=True)
    return combined_df


# --- Main Execution ---
if __name__ == "__main__":
    start_datetime = datetime(2025, 6, 26)
    end_datetime = datetime(2025, 6, 27)
    
    # ✅ Call initialization function once
    initialize_trade_history_file()

    buffer_days = 15
    data_fetch_start_date = start_datetime - timedelta(days=buffer_days)

    if not initialize_mt5_interface(SYMBOLS_TO_BACKTEST):
        logger.error("Failed to initialize MT5 or critical symbols. Exiting.")
    else:
        shared_account_balance = INITIAL_ACCOUNT_BALANCE
        global_active_trade = None
        global_pending_order = None
        all_closed_trades_portfolio = []
        equity_curve_over_time = [] # To track equity for plotting

        # ✅ Step 1: Add a New Variable to Hold "Delayed Setups"
        delayed_setups_queue = []  # List of setups waiting for confirmation

        symbol_conceptual_start_balances = {}
        trades_per_symbol_map = {sym: [] for sym in SYMBOLS_AVAILABLE_FOR_TRADE}

        current_simulation_date = None
        daily_risk_allocated_on_current_date = 0.0
        max_daily_risk_budget_for_current_date = 0.0

        consecutive_losses_count = 0

        logger.info(f"Global Initial Account Balance: {shared_account_balance:.2f} USD")
        logger.info(f"Backtesting Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
        logger.info(f"Risk per trade: {RISK_PER_TRADE_PERCENT*100:.2f}%, Daily Risk Limit: {DAILY_RISK_LIMIT_PERCENT*100:.2f}%")
        logger.info(f"--- SIMULATION: Spread = {SPREAD_PIPS} pips, Slippage = {SLIPPAGE_PIPS} pips ---")
        logger.info("One trade at a time across the entire portfolio.")
        logger.info("Lot size will be fixed to minimum volume. Risk guard active for min lot.")
        logger.info("--- NEW: Trading will halt for the day after 5 consecutive losses.")
        logger.info("--- NEW: Setups are queued and confirmed 2 candles later before placing a pending order.")


        prepared_symbol_data = {}
        master_time_index_set = set()
        for sym in SYMBOLS_AVAILABLE_FOR_TRADE:
            props = ALL_SYMBOL_PROPERTIES[sym]
            df = prepare_symbol_data(sym, data_fetch_start_date, end_datetime, props)
            if not df.empty:
                df_filtered_for_index = df[(df.index >= pd.Timestamp(start_datetime, tz='UTC')) & (df.index <= pd.Timestamp(end_datetime, tz='UTC'))]
                if not df_filtered_for_index.empty:
                    prepared_symbol_data[sym] = df
                    master_time_index_set.update(df_filtered_for_index.index)
                else:
                    logger.warning(f"No data for {sym} within the backtest period {start_datetime} - {end_datetime} after initial buffer fetch. Skipping.")
            else:
                logger.warning(f"No data prepared for {sym}, it will be skipped in simulation.")

        if not master_time_index_set:
            logger.error("No data available for any symbol in the specified range for master time index. Exiting.")
            shutdown_mt5_interface()
            exit()

        master_time_index = sorted(list(master_time_index_set))
        logger.info(f"Master time index created with {len(master_time_index)} M5 candles to process.")

        for timestamp in master_time_index:
            candle_date = timestamp.date()

            if candle_date != current_simulation_date:
                current_simulation_date = candle_date
                daily_risk_allocated_on_current_date = 0.0
                max_daily_risk_budget_for_current_date = shared_account_balance * DAILY_RISK_LIMIT_PERCENT
                consecutive_losses_count = 0
                logger.debug(f"Portfolio New Day: {current_simulation_date}. Max Daily Risk: {max_daily_risk_budget_for_current_date:.2f} (Bal: {shared_account_balance:.2f}). Daily risk & loss counters reset.")

            if global_active_trade:
                trade_symbol = global_active_trade['symbol']
                props = ALL_SYMBOL_PROPERTIES[trade_symbol]

                if trade_symbol in prepared_symbol_data and timestamp in prepared_symbol_data[trade_symbol].index:
                    current_candle = prepared_symbol_data[trade_symbol].loc[timestamp]
                    
                    exit_price = 0
                    
                    # ===== SPREAD SIMULATION FOR EXITS =====
                    spread_price = SPREAD_PIPS * props['pip_value_calc']

                    if global_active_trade['type'] == "BUY":
                        # TP is hit by the BID price, which is chart price - spread (per user logic)
                        if (current_candle['high'] - spread_price) >= global_active_trade['tp']:
                            exit_price = global_active_trade['tp']
                            global_active_trade['status'] = "TP_HIT"
                        # SL is hit by the BID price (per user logic)
                        elif (current_candle['low'] - spread_price) <= global_active_trade['sl']:
                            exit_price = global_active_trade['sl']
                            global_active_trade['status'] = "SL_HIT"

                    elif global_active_trade['type'] == "SELL":
                        # TP is hit by the ASK price, which is chart price + spread (per user logic)
                        if (current_candle['low'] + spread_price) <= global_active_trade['tp']:
                            exit_price = global_active_trade['tp']
                            global_active_trade['status'] = "TP_HIT"
                        # SL is hit by the ASK price (per user logic)
                        elif (current_candle['high'] + spread_price) >= global_active_trade['sl']:
                            exit_price = global_active_trade['sl']
                            global_active_trade['status'] = "SL_HIT"
                    # ===== END SPREAD SIMULATION =====

                    if global_active_trade['status'] != "OPEN":
                        global_active_trade['exit_time'] = timestamp
                        global_active_trade['exit_price'] = exit_price
                        price_diff = (exit_price - global_active_trade['entry_price']) if global_active_trade['type'] == "BUY" else (global_active_trade['entry_price'] - exit_price)
                        if props['trade_tick_size'] > 0:
                            pnl_ticks = price_diff / props['trade_tick_size']
                        else:
                            pnl_ticks = 0
                            logger.error(f"[{trade_symbol}] trade_tick_size is zero or invalid at PNL calculation.")

                        raw_pnl = pnl_ticks * props['trade_tick_value'] * global_active_trade['lot_size']
                        commission_cost = COMMISSIONS.get(trade_symbol, 0.0)
                        global_active_trade['commission'] = commission_cost
                        global_active_trade['pnl_currency'] = raw_pnl - commission_cost

                        shared_account_balance += global_active_trade['pnl_currency']
                        equity_curve_over_time.append((timestamp, shared_account_balance))

                        if global_active_trade['pnl_currency'] < 0:
                            consecutive_losses_count += 1
                            if consecutive_losses_count >= 5:
                                logger.warning(f"STOP LOSS LIMIT HIT: {consecutive_losses_count} consecutive losses...")
                        else:
                            if consecutive_losses_count > 0:
                                logger.info(f"Winning/BE trade recorded. Resetting consecutive loss counter from {consecutive_losses_count} to 0.")
                            consecutive_losses_count = 0

                        global_active_trade['balance_after_trade'] = shared_account_balance
                        logger.info(f"[{trade_symbol}] {timestamp} Trade CLOSED ({global_active_trade['status']}): Net P&L: {global_active_trade['pnl_currency']:.2f}, New Bal: {shared_account_balance:.2f}")
                        all_closed_trades_portfolio.append(global_active_trade.copy())
                        trades_per_symbol_map[trade_symbol].append(global_active_trade.copy())
                        log_backtest_trade_to_csv(global_active_trade)
                        global_active_trade = None
                    
                    elif global_active_trade:
                        atr_val = current_candle.get('ATR', np.nan)
                        if not (pd.isna(atr_val) or atr_val <= 0):
                            move_from_entry = (
                                current_candle['high'] - global_active_trade['entry_price']
                                if global_active_trade['type'] == "BUY"
                                else global_active_trade['entry_price'] - current_candle['low']
                            )

                            atr_movement = move_from_entry / atr_val

                            if atr_movement >= global_active_trade['ts_next_atr_level']:
                                if not global_active_trade['trailing_active']:
                                    global_active_trade['trailing_active'] = True
                                    logger.info(f"[{trade_symbol}] {timestamp} Trailing Stop ACTIVATED at {global_active_trade['ts_next_atr_level']:.2f} ATR.")
                                else:
                                    logger.info(f"[{trade_symbol}] {timestamp} TSL Update Triggered at {global_active_trade['ts_next_atr_level']:.2f} ATR.")

                                symbol_df_for_tsl = prepared_symbol_data[trade_symbol]
                                try:
                                    current_idx = symbol_df_for_tsl.index.get_loc(timestamp)
                                    if current_idx >= 3:
                                        last_3 = symbol_df_for_tsl.iloc[max(0, current_idx - 3): current_idx]
                                        if not last_3.empty:
                                            new_sl = 0
                                            if global_active_trade['type'] == "BUY":
                                                new_sl = last_3['low'].min() - 2 * props['pip_value_calc']
                                                if new_sl > global_active_trade['sl']:
                                                    global_active_trade['sl'] = round(new_sl, props['digits'])
                                            else:
                                                new_sl = last_3['high'].max() + 2 * props['pip_value_calc']
                                                if new_sl < global_active_trade['sl']:
                                                    global_active_trade['sl'] = round(new_sl, props['digits'])

                                            logger.debug(f"[{trade_symbol}] SL updated to {global_active_trade['sl']:.{props['digits']}f} after {global_active_trade['ts_next_atr_level']:.2f} ATR move.")

                                except KeyError:
                                    logger.warning(f"{timestamp} not found in {trade_symbol} df for trailing update.")

                                global_active_trade['ts_next_atr_level'] += 0.5
            
            if not global_active_trade and global_pending_order:
                order_symbol = global_pending_order['symbol']
                props = ALL_SYMBOL_PROPERTIES[order_symbol]
                if order_symbol in prepared_symbol_data and timestamp in prepared_symbol_data[order_symbol].index:
                    current_candle_for_pending_order = prepared_symbol_data[order_symbol].loc[timestamp]
                    entry_price_pending = global_pending_order['entry_price']
                    sl_price_pending = global_pending_order['sl_price']
                    order_type_pending = global_pending_order['type']
                    lot_size_pending = global_pending_order['lot_size']
                    m5_ema21_for_invalidation = current_candle_for_pending_order['M5_EMA21']
                    setup_invalidated = False
                    if global_pending_order['setup_bias'] == "BUY" and current_candle_for_pending_order['close'] < m5_ema21_for_invalidation:
                        setup_invalidated = True
                        logger.info(f"[{order_symbol}] {timestamp} PENDING BUY order invalidated (Close < M5_EMA21 before trigger).")
                    elif global_pending_order['setup_bias'] == "SELL" and current_candle_for_pending_order['close'] > m5_ema21_for_invalidation:
                        setup_invalidated = True
                        logger.info(f"[{order_symbol}] {timestamp} PENDING SELL order invalidated (Close > M5_EMA21 before trigger).")
                    if setup_invalidated:
                        daily_risk_allocated_on_current_date -= global_pending_order['intended_risk_amount']
                        logger.debug(f"[{order_symbol}] Risk {global_pending_order['intended_risk_amount']:.2f} refunded due to pending order invalidation. Daily allocated: {daily_risk_allocated_on_current_date:.2f}")
                        global_pending_order = None
                    else:
                        triggered = False; actual_entry_price = 0
                        if order_type_pending == "BUY_STOP" and current_candle_for_pending_order['high'] >= entry_price_pending:
                            triggered = True
                        elif order_type_pending == "SELL_STOP" and current_candle_for_pending_order['low'] <= entry_price_pending:
                            triggered = True
                        
                        if triggered:
                            # --- ADD SLIPPAGE SIMULATION ---
                            slippage_price_adj = SLIPPAGE_PIPS * props['pip_value_calc']
                            if order_type_pending == "BUY_STOP":
                                actual_entry_price = entry_price_pending + slippage_price_adj
                            else: # SELL_STOP
                                actual_entry_price = entry_price_pending - slippage_price_adj
                            # --- END SIMULATION ---
                            
                            logger.info(f"[{order_symbol}] {timestamp} PENDING {order_type_pending} TRIGGERED. Original Entry: {entry_price_pending:.{props['digits']}f}, Actual (w/ slippage): {actual_entry_price:.{props['digits']}f} Lot:{lot_size_pending}")

                            risk_val_diff = abs(actual_entry_price - sl_price_pending)
                            if risk_val_diff <= 0 or lot_size_pending <= 0:
                                logger.warning(f"[{order_symbol}] Invalid risk (diff {risk_val_diff}) /lot ({lot_size_pending}) on trigger. Cancelling order and refunding risk.")
                                daily_risk_allocated_on_current_date -= global_pending_order['intended_risk_amount']
                                global_pending_order = None
                            else:
                                tp_price = actual_entry_price + (4 * risk_val_diff) if order_type_pending=="BUY_STOP" else actual_entry_price - (4 * risk_val_diff)
                                tp_price = round(tp_price, props['digits'])
                                global_active_trade = {
                                    "symbol": order_symbol, "type": "BUY" if order_type_pending=="BUY_STOP" else "SELL",
                                    "entry_time": timestamp, "entry_price": actual_entry_price, "sl": sl_price_pending,
                                    "initial_sl": sl_price_pending, "tp": tp_price, "r_value_price_diff": risk_val_diff,
                                    "status": "OPEN", "lot_size": lot_size_pending, "pnl_currency": 0.0,
                                    "commission": 0.0, "trailing_active": False,
                                    "ts_trigger_atr_multiple": 1.5,
                                    "ts_next_atr_level": 1.5,
                                }
                                logger.info(f"  [{order_symbol}] Trade OPEN: {global_active_trade['type']} @{global_active_trade['entry_price']:.{props['digits']}f}, SL:{global_active_trade['sl']:.{props['digits']}f}, TP:{global_active_trade['tp']:.{props['digits']}f}, R-dist: {risk_val_diff:.{props['digits']}f}")
                                if order_symbol not in symbol_conceptual_start_balances:
                                    symbol_conceptual_start_balances[order_symbol] = shared_account_balance
                                global_pending_order = None

            
            if not global_active_trade and not global_pending_order:
                
                # --- Step 1: Process the delayed setups queue first ---
                if delayed_setups_queue:
                    setups_to_process_now = []
                    setups_to_keep_for_later = []
                    
                    # Increment counters and split the queue
                    for setup in delayed_setups_queue:
                        setup["confirm_count"] += 1
                        if setup["confirm_count"] >= 2:
                            setups_to_process_now.append(setup)
                        else:
                            setups_to_keep_for_later.append(setup)
                            
                    delayed_setups_queue = setups_to_keep_for_later # Update queue with non-ready setups

                    # Now process the confirmed setups (if any)
                    if setups_to_process_now:
                        for setup in setups_to_process_now:
                            # Safety check, if a pending order was created by a previous item in this same tick's list, stop.
                            if global_pending_order:
                                delayed_setups_queue.append(setup) # Put back in queue for next tick
                                continue
                            
                            # === USER'S SPECIFIC CHANGE APPLIED HERE ===
                            # Confirmation Filter 1: Price-based invalidation is REMOVED as requested.
                            # We now trust the original signal.
                            
                            # Confirmation Filter 2: Check risk budget (Still important)
                            if daily_risk_allocated_on_current_date + setup["risk_amt"] > max_daily_risk_budget_for_current_date + 1e-9:
                                logger.info(f"[{setup['symbol']}] {timestamp} Delayed setup confirmed, but Portfolio Daily Risk Limit would be exceeded. Dropping setup.")
                                continue # Drop this setup and check the next one

                            # If we get here, the setup is confirmed and becomes a pending order
                            daily_risk_allocated_on_current_date += setup["risk_amt"]
                            logger.debug(f"[{setup['symbol']}] Risk {setup['risk_amt']:.2f} allocated for pending order. Daily total: {daily_risk_allocated_on_current_date:.2f}/{max_daily_risk_budget_for_current_date:.2f}")
                            
                            props_pending = ALL_SYMBOL_PROPERTIES[setup['symbol']]
                            order_type = "BUY_STOP" if setup['bias'] == "BUY" else "SELL_STOP"

                            global_pending_order = {
                                "symbol": setup['symbol'], "type": order_type, "entry_price": setup['entry_price'],
                                "sl_price": setup['sl_price'], "lot_size": setup['lot_size'], "setup_bias": setup['bias'],
                                "creation_time": timestamp, "intended_risk_amount": setup['risk_amt']
                            }
                            logger.info(f"[{setup['symbol']}] {timestamp} CONFIRMED setup converted to PENDING order. Type: {order_type}, Entry: {setup['entry_price']:.{props_pending['digits']}f}, SL: {setup['sl_price']:.{props_pending['digits']}f}")
                            
                            # Put remaining setups from this tick back in queue and exit this loop (one pending order at a time)
                            remaining_setups = [s for s in setups_to_process_now if s != setup]
                            delayed_setups_queue.extend(remaining_setups)
                            break 

                # --- Step 2: If no pending order was created, look for new setups ---
                if not global_pending_order and consecutive_losses_count < 5 and daily_risk_allocated_on_current_date < max_daily_risk_budget_for_current_date:
                    for sym_to_check_setup in SYMBOLS_AVAILABLE_FOR_TRADE:
                        
                        try:
                            current_idx = prepared_symbol_data[sym_to_check_setup].index.get_loc(timestamp)
                        except KeyError:
                            continue 
                        if current_idx < 1:
                            continue
                        previous_candle = prepared_symbol_data[sym_to_check_setup].iloc[current_idx - 1]

                        props_setup = ALL_SYMBOL_PROPERTIES[sym_to_check_setup]
                        if not is_within_session(timestamp, TRADING_SESSIONS_UTC.get(sym_to_check_setup,[])):
                            continue

                        h1_ema8 = previous_candle['H1_EMA8']; h1_ema21 = previous_candle['H1_EMA21']
                        h1_close = previous_candle['H1_Close_For_Bias']
                        if pd.isna(h1_ema8) or pd.isna(h1_ema21) or pd.isna(h1_close): continue
                        
                        h1_trend_bias_setup = None
                        if h1_ema8 > h1_ema21 and h1_close > h1_ema8 and h1_close > h1_ema21: h1_trend_bias_setup = "BUY"
                        elif h1_ema8 < h1_ema21 and h1_close < h1_ema8 and h1_close < h1_ema21: h1_trend_bias_setup = "SELL"
                        if h1_trend_bias_setup is None: continue

                        m5_ema8 = previous_candle['M5_EMA8']; m5_ema13 = previous_candle['M5_EMA13']; m5_ema21_val = previous_candle['M5_EMA21']
                        if pd.isna(m5_ema8) or pd.isna(m5_ema13) or pd.isna(m5_ema21_val): continue
                        
                        m5_fanned_buy = m5_ema8 > m5_ema13 or m5_ema8 > m5_ema21_val
                        m5_fanned_sell = m5_ema8 < m5_ema13 or m5_ema8 < m5_ema21_val
                        is_fanned_for_bias = (h1_trend_bias_setup == "BUY" and m5_fanned_buy) or (h1_trend_bias_setup == "SELL" and m5_fanned_sell)
                        if not is_fanned_for_bias: continue

                        m5_setup_bias_setup = h1_trend_bias_setup

                        h4_ema8 = previous_candle.get('H4_EMA8', np.nan)
                        h4_ema21 = previous_candle.get('H4_EMA21', np.nan)
                        if pd.isna(h4_ema8) or pd.isna(h4_ema21): continue
                        if (m5_setup_bias_setup == "BUY" and h4_ema8 < h4_ema21) or \
                           (m5_setup_bias_setup == "SELL" and h4_ema8 > h4_ema21): continue

                        rsi_m5 = previous_candle.get('RSI_M5', np.nan); rsi_h1 = previous_candle.get('RSI_H1', np.nan)
                        if pd.isna(rsi_m5) or pd.isna(rsi_h1): continue
                        if (m5_setup_bias_setup == "BUY" and not (rsi_m5 > 50 and rsi_h1 > 50)) or \
                           (m5_setup_bias_setup == "SELL" and not (rsi_m5 < 50 and rsi_h1 < 50)): continue

                        if (m5_setup_bias_setup == "BUY" and previous_candle['close'] < m5_ema21_val) or \
                           (m5_setup_bias_setup == "SELL" and previous_candle['close'] > m5_ema21_val): continue
                        
                        pullback = (m5_setup_bias_setup == "BUY" and previous_candle['low'] <= m5_ema8) or \
                                   (m5_setup_bias_setup == "SELL" and previous_candle['high'] >= m5_ema8)
                        if not pullback: continue
                        
                        if current_idx < 4: continue
                        
                        symbol_df = prepared_symbol_data[sym_to_check_setup]
                        
                        # =========================================================================
                        # ===== START OF MODIFIED BLOCK WITH NEW FILTERS =====
                        # =========================================================================
                        recent_candles_weakness = symbol_df.iloc[current_idx - 5 : current_idx - 1]
                        if len(recent_candles_weakness) < 4: continue
                        bullish_count = (recent_candles_weakness['close'] > recent_candles_weakness['open']).sum()
                        bearish_count = (recent_candles_weakness['close'] < recent_candles_weakness['open']).sum()
                        if (m5_setup_bias_setup == "BUY" and bullish_count > 2) or \
                           (m5_setup_bias_setup == "SELL" and bearish_count > 2): continue

                        # --- START OF NEW LOGIC ---
                        # Get ATR value early as it's needed for the confluence filter's tolerance
                        atr_val = previous_candle.get('ATR', np.nan)
                        if pd.isna(atr_val) or atr_val <= 0: continue

                        # Define the lookback window for the impulse leg (10 candles before the signal candle)
                        if current_idx < 12: continue # Ensure we have enough data for the lookback
                        lookback_window_for_swing = symbol_df.iloc[current_idx - 11 : current_idx - 1]
                        if lookback_window_for_swing.empty: continue

                        swing_high = lookback_window_for_swing['high'].max()
                        swing_low = lookback_window_for_swing['low'].min()

                        # 1. Pullback Depth Filter
                        # Define impulse start/end based on bias
                        impulse_start, impulse_end, price_for_pb = (swing_low, swing_high, previous_candle['low']) if m5_setup_bias_setup == "BUY" else (swing_high, swing_low, previous_candle['high'])
                        
                        # Check if pullback is at least 30%
                        if calculate_pullback_depth(impulse_start, impulse_end, price_for_pb, m5_setup_bias_setup) < 0.30:
                            continue

                        # 2. EMA-Fibonacci Confluence Filter
                        fib_levels = calculate_fib_levels(swing_high, swing_low)
                        tolerance = 0.5 * atr_val # Use 50% of ATR as tolerance
                        
                        m5_ema8_val = previous_candle['M5_EMA8']
                        m5_ema13_val = previous_candle['M5_EMA13']
                        
                        # Check if either EMA is close to any of the calculated Fib levels
                        fib_confluence_found = False
                        for fib_price in fib_levels.values():
                            if abs(m5_ema8_val - fib_price) <= tolerance or abs(m5_ema13_val - fib_price) <= tolerance:
                                fib_confluence_found = True
                                break
                        
                        if not fib_confluence_found:
                            continue
                        # --- END OF NEW LOGIC ---

                        lookback_df_for_entry = symbol_df.iloc[current_idx - 3 : current_idx]
                        
                        pip_adj_setup = 3 * props_setup['trade_tick_size']
                        # =========================================================================
                        # ===== END OF MODIFIED BLOCK =====
                        # =========================================================================
                        
                        sl_distance_atr = 2.5 * atr_val
                        entry_px, sl_px = (0, 0)
                        
                        if m5_setup_bias_setup == "BUY":
                            entry_px = lookback_df_for_entry['high'].max() + pip_adj_setup
                            sl_px = entry_px - sl_distance_atr
                        else: # SELL
                            entry_px = lookback_df_for_entry['low'].min() - pip_adj_setup
                            sl_px = entry_px + sl_distance_atr

                        entry_px = round(entry_px, props_setup['digits'])
                        sl_px = round(sl_px, props_setup['digits'])
                        
                        if abs(entry_px - sl_px) <= 0: continue

                        lot_size_fixed_min = props_setup.get("volume_min", 0.01)
                        estimated_risk_min_lot = lot_size_fixed_min * (abs(entry_px - sl_px) / props_setup['trade_tick_size']) * props_setup['trade_tick_value'] if props_setup['trade_tick_size'] > 0 else 0
                        if estimated_risk_min_lot <= 0: continue
                        
                        max_allowed_risk_per_trade = shared_account_balance * RISK_PER_TRADE_PERCENT
                        if estimated_risk_min_lot > max_allowed_risk_per_trade: continue
                        
                        delayed_setups_queue.append({
                            "symbol": sym_to_check_setup, "timestamp": timestamp, "bias": m5_setup_bias_setup,
                            "entry_price": entry_px, "sl_price": sl_px, "lot_size": lot_size_fixed_min,
                            "risk_amt": estimated_risk_min_lot, "confirm_count": 0
                        })
                        logger.info(f"[{sym_to_check_setup}] {timestamp} Setup QUEUED. Bias: {m5_setup_bias_setup}, Entry: {entry_px}")


        logger.info("\n\n===== All Symbol Simulations Complete. Generating Summaries. =====")

        for symbol_iter, symbol_trades_list in trades_per_symbol_map.items():
            logger.info(f"\n--- Performance Summary for Symbol: {symbol_iter} ---")
            logger.info(f"  Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")

            conceptual_start_bal_for_symbol = INITIAL_ACCOUNT_BALANCE
            if symbol_trades_list:
                symbol_trades_list.sort(key=lambda x: x['entry_time'])
                conceptual_start_bal_for_symbol = symbol_conceptual_start_balances.get(symbol_iter, INITIAL_ACCOUNT_BALANCE)

            if not symbol_trades_list:
                logger.info(f"  Starting Balance (conceptual for this symbol): {INITIAL_ACCOUNT_BALANCE:.2f} USD")
                logger.info(f"  Ending Balance (conceptual): {INITIAL_ACCOUNT_BALANCE:.2f} USD")
                logger.info(f"  No trades executed for {symbol_iter} during the backtest period.")
            else:
                symbol_stats = calculate_performance_stats(symbol_trades_list, conceptual_start_bal_for_symbol)
                logger.info(f"  Starting Balance (conceptual, portfolio balance when this symbol's first trade was initiated): {symbol_stats['start_balance']:.2f} USD")
                logger.info(f"  Ending Balance (conceptual, start_bal + PnL for this symbol's trades): {symbol_stats['end_balance']:.2f} USD")
                logger.info(f"  Total Trades: {symbol_stats['total_trades']}")
                logger.info(f"  Winning Trades: {symbol_stats['winning_trades']}")
                logger.info(f"  Losing Trades: {symbol_stats['losing_trades']}")
                logger.info(f"  Win Rate: {symbol_stats['win_rate']:.2f}%")
                logger.info(f"  Net Profit (for this symbol): {symbol_stats['net_profit']:.2f} USD")
                logger.info(f"  Profit Factor (for this symbol): {symbol_stats['profit_factor']:.2f}")
                logger.info(f"  Max Drawdown (based on this symbol's trades in isolation from its conceptual start): {symbol_stats['max_drawdown_abs']:.2f} USD ({symbol_stats['max_drawdown_pct']:.2f}%)")

        logger.info("\n\n===== Overall Backtest Performance Summary =====")
        if all_closed_trades_portfolio:
            all_closed_trades_portfolio.sort(key=lambda x: x['entry_time'])
            overall_stats = calculate_performance_stats(all_closed_trades_portfolio, INITIAL_ACCOUNT_BALANCE)
            rr_distribution = analyze_rr_distribution(all_closed_trades_portfolio, ALL_SYMBOL_PROPERTIES)
            logger.info(f"Tested Symbols: {SYMBOLS_AVAILABLE_FOR_TRADE}")
            logger.info(f"Overall Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
            logger.info(f"Overall Starting Balance: {overall_stats['start_balance']:.2f} USD")
            logger.info(f"Overall Ending Balance (from sorted trades PnL): {overall_stats['end_balance']:.2f} USD")
            logger.info(f"Overall Ending Balance (final portfolio balance variable): {shared_account_balance:.2f} USD")
            logger.info(f"Overall Total Trades: {overall_stats['total_trades']}")
            logger.info(f"Overall Winning Trades: {overall_stats['winning_trades']}")
            logger.info(f"Overall Losing Trades: {overall_stats['losing_trades']}")
            logger.info(f"Overall Win Rate: {overall_stats['win_rate']:.2f}%")
            logger.info(f"Overall Net Profit: {overall_stats['net_profit']:.2f} USD")
            logger.info(f"Overall Profit Factor: {overall_stats['profit_factor']:.2f}")
            logger.info(f"Overall Max Drawdown: {overall_stats['max_drawdown_abs']:.2f} USD ({overall_stats['max_drawdown_pct']:.2f}%)")
            logger.info("\n--- RR Distribution Summary ---")
            total_counted_trades = sum(rr_distribution.values())
            logger.info(f"  (Analysis based on {total_counted_trades} of {overall_stats['total_trades']} total trades)")
            for bucket, count in rr_distribution.items():
                if count > 0:
                    percentage = (count / total_counted_trades) * 100 if total_counted_trades > 0 else 0
                    logger.info(f"  {bucket:<25}: {count:<5} trades ({percentage:.2f}%)")
        else:
            logger.info("No trades were executed across any symbols during the backtest period.")
            logger.info(f"Overall Starting Balance: {INITIAL_ACCOUNT_BALANCE:.2f} USD")
            logger.info(f"Overall Ending Balance: {shared_account_balance:.2f} USD")

        shutdown_mt5_interface()

        # Create a DataFrame from recorded equity points
        if equity_curve_over_time:
            eq_df = pd.DataFrame(equity_curve_over_time, columns=["timestamp", "equity"])
            eq_df.set_index("timestamp", inplace=True)

            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(eq_df.index, eq_df["equity"], label="Equity Curve", color="blue")
            plt.title("Equity Curve Over Time")
            plt.xlabel("Time")
            plt.ylabel("Equity (USD)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.savefig("equity_curve.png") # Uncomment to save the plot to a file
            plt.show()
        else:
            logger.warning("No equity data to plot.")