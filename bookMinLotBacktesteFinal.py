import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta # For EMAs and ATR
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import math

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}
RUN_BACKTEST = True # Important for the provided MT5 init function

# --- Strategy & Backtest Parameters ---
SYMBOLS_TO_BACKTEST = ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                           "AUDJPY",  "XAUUSD",
                       "USOIL",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD"  ]

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


INITIAL_ACCOUNT_BALANCE = 200.00
RISK_PER_TRADE_PERCENT = 0.01 # Risk 1% of current balance per trade
DAILY_RISK_LIMIT_PERCENT = 0.05 # Daily risk limit of 5% of balance at start of day

# --- NEW: Commission Structure ---
# This dictionary holds the commission cost per trade for the minimum lot size.
# The bot currently only trades the minimum lot, so this value is applied directly.
COMMISSIONS = {
    "EURUSD": 0.07, "AUDUSD": 0.10, "USDCHF": 0.10, "USDCAD": 0.10,
    "NZDUSD": 0.13, "AUDJPY": 0.09, "EURNZD": 0.18, "USOIL": 0.16,
    "UKOIL": 0.65, "BTCUSD": 0.16, "BTCJPY": 0.21, "BTCXAU": 0.20,
    "ETHUSD": 0.30, "GBPUSD": 0.09, "USDJPY": 0.07, "GBPJPY": 0.15,
    "XAUUSD":0.11,
}


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

def analyze_rr_distribution(closed_trades, symbol_properties_dict):
    """
    Analyzes the distribution of closed trades based on their Risk-to-Reward ratio.

    Args:
        closed_trades (list): A list of closed trade dictionaries.
        symbol_properties_dict (dict): A dictionary containing properties for all symbols.

    Returns:
        dict: A dictionary with RR buckets as keys and trade counts as values.
    """
    rr_buckets = {
        "Stop Loss (~ -1R)": 0,
        "Partial Loss (< 0R)": 0,
        "Break Even (0R to <1R)": 0,
        "1R to <2R": 0,
        "2R to <3R": 0,
        "3R to <4R": 0,
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
            rr_buckets["3R to <4R"] += 1
        elif 2.0 <= rr_value < 3.0:
            rr_buckets["2R to <3R"] += 1
        elif 1.0 <= rr_value < 2.0:
            rr_buckets["1R to <2R"] += 1
        elif 0.0 <= rr_value < 1.0:
            rr_buckets["Break Even (0R to <1R)"] += 1
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

    # --- M5 Data ---
    df_m5 = get_historical_data(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    if df_m5.empty: return pd.DataFrame()
    df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
    df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
    df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
    if len(df_m5) >= 14:
        df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
    else:
        df_m5['ATR'] = np.nan
    df_m5['RSI_M5'] = ta.rsi(df_m5['close'], length=14) # 🧮 For M5 RSI (User Step 1)

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
    start_datetime = datetime(2024, 8, 1)
    end_datetime = datetime(2025, 5, 31)

    buffer_days = 15
    data_fetch_start_date = start_datetime - timedelta(days=buffer_days)

    if not initialize_mt5_interface(SYMBOLS_TO_BACKTEST):
        logger.error("Failed to initialize MT5 or critical symbols. Exiting.")
    else:
        shared_account_balance = INITIAL_ACCOUNT_BALANCE
        global_active_trade = None
        global_pending_order = None
        all_closed_trades_portfolio = []

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
                    current_candle_for_active_trade = prepared_symbol_data[trade_symbol].loc[timestamp]
                    closed_this_bar = False
                    exit_price = 0

                    if global_active_trade['type'] == "BUY" and current_candle_for_active_trade['low'] <= global_active_trade['sl']:
                        exit_price = global_active_trade['sl']; global_active_trade['status'] = "SL_HIT"
                    elif global_active_trade['type'] == "SELL" and current_candle_for_active_trade['high'] >= global_active_trade['sl']:
                        exit_price = global_active_trade['sl']; global_active_trade['status'] = "SL_HIT"

                    if global_active_trade['status'] == "OPEN":
                        if global_active_trade['type'] == "BUY" and current_candle_for_active_trade['high'] >= global_active_trade['tp']:
                            exit_price = global_active_trade['tp']; global_active_trade['status'] = "TP_HIT"
                        elif global_active_trade['type'] == "SELL" and current_candle_for_active_trade['low'] <= global_active_trade['tp']:
                            exit_price = global_active_trade['tp']; global_active_trade['status'] = "TP_HIT"

                    if global_active_trade['status'] != "OPEN":
                        closed_this_bar = True
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

                        if global_active_trade['pnl_currency'] < 0:
                            consecutive_losses_count += 1
                            logger.info(f"Loss recorded. Consecutive losses now: {consecutive_losses_count}.")
                            if consecutive_losses_count >= 5:
                                logger.warning(f"STOP LOSS LIMIT HIT: {consecutive_losses_count} consecutive losses. No new trades will be initiated for the rest of the day: {current_simulation_date}.")
                        else: # Win or Break-even
                            if consecutive_losses_count > 0:
                                logger.info(f"Winning/BE trade recorded. Resetting consecutive loss counter from {consecutive_losses_count} to 0.")
                            consecutive_losses_count = 0

                        global_active_trade['balance_after_trade'] = shared_account_balance
                        logger.info(f"[{trade_symbol}] {timestamp} Trade CLOSED ({global_active_trade['status']}): Raw P&L: {raw_pnl:.2f}, Comm: {commission_cost:.2f}, Net P&L: {global_active_trade['pnl_currency']:.2f}, New Portfolio Bal: {shared_account_balance:.2f}")
                        all_closed_trades_portfolio.append(global_active_trade.copy())
                        trades_per_symbol_map[trade_symbol].append(global_active_trade.copy())
                        global_active_trade = None

                    if global_active_trade and not closed_this_bar:
                        # === New ATR-Based Trailing Stop Logic ===
                        atr_val = current_candle_for_active_trade.get('ATR', np.nan)
                        if not (pd.isna(atr_val) or atr_val <= 0):
                            # Determine current move size from entry
                            move_from_entry = (
                                current_candle_for_active_trade['high'] - global_active_trade['entry_price']
                                if global_active_trade['type'] == "BUY"
                                else global_active_trade['entry_price'] - current_candle_for_active_trade['low']
                            )

                            # How many ATRs has the price moved?
                            atr_movement = move_from_entry / atr_val

                            # Check if it's time to trail
                            if atr_movement >= global_active_trade['ts_next_atr_level']:
                                if not global_active_trade['trailing_active']:
                                    global_active_trade['trailing_active'] = True
                                    logger.info(f"[{trade_symbol}] {timestamp} Trailing Stop ACTIVATED at {global_active_trade['ts_next_atr_level']:.2f} ATR.")
                                else:
                                    logger.info(f"[{trade_symbol}] {timestamp} TSL Update Triggered at {global_active_trade['ts_next_atr_level']:.2f} ATR.")

                                # 🔁 Adjust SL based on recent lows/highs
                                symbol_df_for_tsl = prepared_symbol_data[trade_symbol]
                                try:
                                    current_idx = symbol_df_for_tsl.index.get_loc(timestamp)
                                    if current_idx >= 2:
                                        last_3 = symbol_df_for_tsl.iloc[max(0, current_idx - 2): current_idx + 1]

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

                                # Increment next trailing level (e.g., 1.5 → 2.0 → 2.5)
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
                            actual_entry_price = entry_price_pending; triggered = True
                        elif order_type_pending == "SELL_STOP" and current_candle_for_pending_order['low'] <= entry_price_pending:
                            actual_entry_price = entry_price_pending; triggered = True
                        if triggered:
                            logger.info(f"[{order_symbol}] {timestamp} PENDING {order_type_pending} TRIGGERED @{actual_entry_price:.{props['digits']}f} Lot:{lot_size_pending}")
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
                                    # ✅ Step 1: Replace fixed ts_trigger_levels with dynamic ATR tracking
                                    "ts_trigger_atr_multiple": 1.5,  # start trailing when price moves 1.5×ATR
                                    "ts_next_atr_level": 1.5,        # track when next trail is due
                                }
                                logger.info(f"  [{order_symbol}] Trade OPEN: {global_active_trade['type']} @{global_active_trade['entry_price']:.{props['digits']}f}, SL:{global_active_trade['sl']:.{props['digits']}f}, TP:{global_active_trade['tp']:.{props['digits']}f}, R-dist: {risk_val_diff:.{props['digits']}f}")
                                if order_symbol not in symbol_conceptual_start_balances:
                                    symbol_conceptual_start_balances[order_symbol] = shared_account_balance
                                global_pending_order = None

            if not global_active_trade and not global_pending_order:
                # ✅ Step 3: In Each Time Step, Process the Delay Queue
                # Step through delayed setups
                new_queue = []
                for setup in delayed_setups_queue:
                    if setup['symbol'] not in prepared_symbol_data:
                        continue
                    if timestamp not in prepared_symbol_data[setup['symbol']].index:
                        # If current timestamp is missing for this symbol's data, keep it in the queue
                        # but don't increment its counter.
                        new_queue.append(setup)
                        continue

                    setup['confirm_count'] += 1

                    # Wait 2 candles (i.e., 10 minutes)
                    if setup['confirm_count'] < 2:
                        new_queue.append(setup)
                        continue

                    current_candle = prepared_symbol_data[setup['symbol']].loc[timestamp]

                    # Confirm the trend is still valid (reapply filters like M5 EMA21 check)
                    if setup['bias'] == "BUY" and current_candle['close'] < current_candle['M5_EMA21']:
                        logger.info(f"[{setup['symbol']}] {timestamp} Delayed BUY setup invalidated. Not placing order.")
                        continue # Invalidated, so don't add to new_queue
                    if setup['bias'] == "SELL" and current_candle['close'] > current_candle['M5_EMA21']:
                        logger.info(f"[{setup['symbol']}] {timestamp} Delayed SELL setup invalidated. Not placing order.")
                        continue # Invalidated, so don't add to new_queue

                    # Setup confirmed → place order
                    order_type = "BUY_STOP" if setup['bias'] == "BUY" else "SELL_STOP"

                    # Double-check daily risk budget before placing the order
                    if daily_risk_allocated_on_current_date + setup["risk_amt"] > max_daily_risk_budget_for_current_date + 1e-9:
                        logger.info(f"[{setup['symbol']}] {timestamp} Delayed setup confirmed, but Portfolio Daily Risk Limit would be exceeded. Cancelling.")
                        continue # Exceeds risk, so don't place and don't re-queue

                    global_pending_order = {
                        "symbol": setup["symbol"],
                        "type": order_type,
                        "entry_price": setup["entry_price"],
                        "sl_price": setup["sl_price"],
                        "created_time": timestamp,
                        "lot_size": setup["lot_size"],
                        "setup_bias": setup["bias"],
                        "intended_risk_amount": setup["risk_amt"]
                    }

                    daily_risk_allocated_on_current_date += setup["risk_amt"]
                    logger.info(f"[{setup['symbol']}] {timestamp} Delayed Setup Confirmed. Placing {order_type} pending order. Risk: {setup['risk_amt']:.2f}")
                    logger.debug(f"  Portfolio daily risk allocated: {daily_risk_allocated_on_current_date:.2f}/{max_daily_risk_budget_for_current_date:.2f}")

                    # Once one setup is promoted to a pending order, stop processing the queue for this tick.
                    # This maintains the "one trade at a time" rule.
                    # Any remaining setups in the old queue will be added to the new queue to be processed on the next tick.
                    new_queue.extend(delayed_setups_queue[delayed_setups_queue.index(setup)+1:])
                    break

                # Update queue
                delayed_setups_queue = new_queue

                if consecutive_losses_count >= 5:
                    continue

                # Only look for new setups if one was NOT just promoted from the queue
                if not global_pending_order:
                    for sym_to_check_setup in SYMBOLS_AVAILABLE_FOR_TRADE:
                        if sym_to_check_setup not in prepared_symbol_data or timestamp not in prepared_symbol_data[sym_to_check_setup].index:
                            continue

                        current_candle_for_setup = prepared_symbol_data[sym_to_check_setup].loc[timestamp]
                        props_setup = ALL_SYMBOL_PROPERTIES[sym_to_check_setup]

                        pip_adj_setup = 3 * props_setup['trade_tick_size']

                        h1_trend_bias_setup = None
                        m5_setup_bias_setup = None

                        if not is_within_session(timestamp, TRADING_SESSIONS_UTC.get(sym_to_check_setup,[])):
                            continue

                        h1_ema8 = current_candle_for_setup['H1_EMA8']; h1_ema21 = current_candle_for_setup['H1_EMA21']
                        h1_close = current_candle_for_setup['H1_Close_For_Bias']
                        if pd.isna(h1_ema8) or pd.isna(h1_ema21) or pd.isna(h1_close): continue

                        if h1_ema8>h1_ema21 and h1_close>h1_ema8 and h1_close>h1_ema21: h1_trend_bias_setup="BUY"
                        elif h1_ema8<h1_ema21 and h1_close<h1_ema8 and h1_close<h1_ema21: h1_trend_bias_setup="SELL"
                        if h1_trend_bias_setup is None: continue

                        m5_ema8 = current_candle_for_setup['M5_EMA8']; m5_ema13 = current_candle_for_setup['M5_EMA13']; m5_ema21_val = current_candle_for_setup['M5_EMA21']
                        if pd.isna(m5_ema8) or pd.isna(m5_ema13) or pd.isna(m5_ema21_val): continue

                        m5_fanned_buy = m5_ema8 > m5_ema13 or m5_ema8 > m5_ema21_val
                        m5_fanned_sell = m5_ema8 < m5_ema13 or m5_ema8 < m5_ema21_val
                        is_fanned_for_bias = (h1_trend_bias_setup == "BUY" and m5_fanned_buy) or (h1_trend_bias_setup == "SELL" and m5_fanned_sell)
                        if not is_fanned_for_bias:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} M5 EMA structure not aligned enough for {h1_trend_bias_setup} setup. Skipping.")
                            continue

                        m5_setup_bias_setup = h1_trend_bias_setup

                        h4_ema8 = current_candle_for_setup.get('H4_EMA8', np.nan)
                        h4_ema21 = current_candle_for_setup.get('H4_EMA21', np.nan)
                        if pd.isna(h4_ema8) or pd.isna(h4_ema21): continue

                        if m5_setup_bias_setup == "BUY" and h4_ema8 < h4_ema21: continue
                        if m5_setup_bias_setup == "SELL" and h4_ema8 > h4_ema21: continue

                        rsi_m5 = current_candle_for_setup.get('RSI_M5', np.nan)
                        rsi_h1 = current_candle_for_setup.get('RSI_H1', np.nan)
                        rsi_h4 = current_candle_for_setup.get('RSI_H4', np.nan)

                        if pd.isna(rsi_m5) or pd.isna(rsi_h1) or pd.isna(rsi_h4):
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} Missing RSI values. Skipping setup.")
                            continue

                        if m5_setup_bias_setup == "BUY" and not (rsi_m5 > 50 and rsi_h1 > 50):
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} RSI misalignment for BUY. M5:{rsi_m5:.1f} H1:{rsi_h1:.1f} H4:{rsi_h4:.1f}")
                            continue

                        if m5_setup_bias_setup == "SELL" and not (rsi_m5 < 50 and rsi_h1 < 50 ):
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} RSI misalignment for SELL. M5:{rsi_m5:.1f} H1:{rsi_h1:.1f} H4:{rsi_h4:.1f}")
                            continue

                        if (m5_setup_bias_setup=="BUY" and current_candle_for_setup['close'] < m5_ema21_val) or \
                           (m5_setup_bias_setup=="SELL" and current_candle_for_setup['close'] > m5_ema21_val):
                            continue

                        pullback = (m5_setup_bias_setup=="BUY" and current_candle_for_setup['low']<=m5_ema8) or \
                                   (m5_setup_bias_setup=="SELL" and current_candle_for_setup['high']>=m5_ema8)
                        if not pullback: continue

                        symbol_df_for_weakness_filter = prepared_symbol_data[sym_to_check_setup]
                        try:
                            current_idx_for_weakness_filter = symbol_df_for_weakness_filter.index.get_loc(timestamp)
                        except KeyError:
                            logger.warning(f"[{sym_to_check_setup}] {timestamp} Timestamp not found in DataFrame for weakness filter. Skipping setup.")
                            continue

                        current_idx_in_symbol_df_for_lookback = current_idx_for_weakness_filter # Use the same index

                        if current_idx_for_weakness_filter < 4:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} Not enough preceding M5 candles for weakness filter ({current_idx_for_weakness_filter} total candles up to current). Skipping setup.")
                            continue
                        recent_candles = symbol_df_for_weakness_filter.iloc[current_idx_for_weakness_filter - 4 : current_idx_for_weakness_filter]
                        if len(recent_candles) < 4:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} Sliced less than 4 preceding M5 candles for weakness filter ({len(recent_candles)} found). Skipping setup.")
                            continue
                        bullish_count = (recent_candles['close'] > recent_candles['open']).sum()
                        bearish_count = (recent_candles['close'] < recent_candles['open']).sum()

                        if m5_setup_bias_setup == "BUY" and bullish_count > 2:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} Preceding Bar Weakness Filter: Too many prior bullish M5 candles ({bullish_count}/4). Skipping BUY setup.")
                            continue

                        if m5_setup_bias_setup == "SELL" and bearish_count > 2:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} Preceding Bar Weakness Filter: Too many prior bearish M5 candles ({bearish_count}/4). Skipping SELL setup.")
                            continue

                        atr_val = current_candle_for_setup.get('ATR', np.nan)
                        if pd.isna(atr_val) or atr_val <= 0:
                            continue

                        # Identify last swing leg (simple: last 10 candles)
                        lookback_window = 10
                        if current_idx_in_symbol_df_for_lookback >= lookback_window:
                            recent_candles = prepared_symbol_data[sym_to_check_setup].iloc[current_idx_in_symbol_df_for_lookback - lookback_window: current_idx_in_symbol_df_for_lookback]
                            swing_high = recent_candles['high'].max()
                            swing_low = recent_candles['low'].min()
                        else:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} Not enough data for Swing/Pullback analysis.")
                            continue

                        # --- NEW PULLBACK FILTER LOGIC START ---
                        # ✅ Step 2: Identify the Impulse Leg
                        if m5_setup_bias_setup == "BUY":
                            impulse_start = swing_low
                            impulse_end = swing_high
                            current_price = current_candle_for_setup['low']
                        else: # SELL
                            impulse_start = swing_high
                            impulse_end = swing_low
                            current_price = current_candle_for_setup['high']

                        # ✅ Step 3: Calculate Pullback Depth and Apply Filter
                        pullback_depth = calculate_pullback_depth(impulse_start, impulse_end, current_price, m5_setup_bias_setup)
                        min_required_pullback = 0.30  # 30%

                        if pullback_depth < min_required_pullback:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} Pullback too shallow ({pullback_depth*100:.1f}%). Skipping.")
                            continue
                        # --- NEW PULLBACK FILTER LOGIC END ---

                        #Fibonacci Logic - must come AFTER swing is identified
                        if m5_setup_bias_setup == "BUY":
                            fib_levels = calculate_fib_levels(swing_high, swing_low)
                        else:
                            fib_levels = calculate_fib_levels(swing_low, swing_high)  # reversed for SELL

                        # Check for Confluence with EMA8 or EMA13
                        ema8 = current_candle_for_setup['M5_EMA8']
                        ema13 = current_candle_for_setup['M5_EMA13']
                        tolerance = 0.5 * atr_val # Define how close is "confluent"

                        confluent = False
                        for level_key in ['0.382', '0.5', '0.618']:
                            fib_price = fib_levels[level_key]
                            if abs(ema8 - fib_price) <= tolerance or abs(ema13 - fib_price) <= tolerance:
                                confluent = True
                                break

                        if not confluent:
                            logger.debug(f"[{sym_to_check_setup}] {timestamp} No EMA-Fib confluence. Skipping.")
                            continue

                        sl_distance_atr = 1.5 * atr_val

                        symbol_df_for_lookback = prepared_symbol_data[sym_to_check_setup]
                        if current_idx_in_symbol_df_for_lookback < 4:
                            continue

                        lookback_df_for_entry = symbol_df_for_lookback.iloc[current_idx_in_symbol_df_for_lookback-2 : current_idx_in_symbol_df_for_lookback+1]
                        entry_px, sl_px, order_type_setup = (0,0,"")
                        if m5_setup_bias_setup=="BUY":
                            entry_px = lookback_df_for_entry['high'].max() + pip_adj_setup
                            sl_px = entry_px - sl_distance_atr
                        else: # SELL
                            entry_px = lookback_df_for_entry['low'].min() - pip_adj_setup
                            sl_px = entry_px + sl_distance_atr
                        entry_px=round(entry_px, props_setup['digits'])
                        sl_px=round(sl_px, props_setup['digits'])
                        sl_diff = abs(entry_px - sl_px)
                        if sl_diff <= 0:
                            continue

                        lot_size_fixed_min = props_setup.get("volume_min", 0.01)
                        tick_val_setup = props_setup["trade_tick_value"]
                        tick_size_setup = props_setup["trade_tick_size"]
                        estimated_risk_min_lot = 0.0
                        if tick_size_setup > 0 and tick_val_setup > 0:
                            estimated_risk_min_lot = lot_size_fixed_min * (sl_diff / tick_size_setup) * tick_val_setup
                        else:
                            logger.warning(f"[{sym_to_check_setup}] {timestamp} Invalid tick_size ({tick_size_setup}) or tick_value ({tick_val_setup}) for risk estimation. Skipping trade.")
                            continue
                        if estimated_risk_min_lot <= 0:
                            logger.warning(f"[{sym_to_check_setup}] {timestamp} Estimated risk with min lot is zero or negative ({estimated_risk_min_lot:.2f}). SL diff: {sl_diff:.{props_setup['digits']}f}. Skipping trade.")
                            continue
                        max_allowed_risk_per_trade = shared_account_balance * RISK_PER_TRADE_PERCENT
                        if estimated_risk_min_lot > max_allowed_risk_per_trade:
                            logger.info(f"[{sym_to_check_setup}] {timestamp} SKIP PENDING: Min lot ({lot_size_fixed_min}) risk ({estimated_risk_min_lot:.2f}) for SL {sl_diff:.{props_setup['digits']}f} exceeds {RISK_PER_TRADE_PERCENT*100:.2f}% of balance ({max_allowed_risk_per_trade:.2f}).")
                            continue

                        # Modify Setup Detection to Queue, Not Place Order
                        delayed_setups_queue.append({
                            "symbol": sym_to_check_setup,
                            "timestamp": timestamp,
                            "bias": m5_setup_bias_setup,
                            "entry_price": entry_px,
                            "sl_price": sl_px,
                            "lot_size": lot_size_fixed_min,
                            "risk_amt": estimated_risk_min_lot,
                            "confirm_count": 0  # tracks how many candles passed
                        })
                        logger.info(f"[{sym_to_check_setup}] {timestamp} Setup QUEUED for delayed confirmation. Bias: {m5_setup_bias_setup}, Entry: {entry_px:.{props_setup['digits']}f}")

                        # Break after finding and queuing one setup
                        break

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