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
SYMBOLS_TO_BACKTEST = ["ETHUSD", "BTCXAU", "BTCJPY", "BTCUSD", "UKOIL", "USOIL",  "AUDJPY",  "USDCAD"    ]

#  ["ETHUSD", "BTCXAU", "BTCJPY", "BTCUSD", "UKOIL", "USOIL", "AUDJPY", "EURCHF", "EURUSD" ,   ]



                    # ["EURUSD", "AUDUSD", "USDCHF", "USDCAD",
                    #    "GBPJPY",  "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                    #    "USOIL", "CADJPY", "XAGUSD", "XPTUSD", "UKOIL",
                    #    "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD", "XAGGBP", "XAGEUR", "XAGAUD", "BTCXAG" ]


TRADING_SESSIONS_UTC = { # (start_hour_inclusive, end_hour_exclusive)
    "EURUSD": [(7, 16)], "GBPUSD": [(7, 16)], "AUDUSD": [(0, 4), (7, 16)],
    "USDCHF": [(7, 16)], "USDCAD": [(12, 16)], "USDJPY": [(0, 4), (12, 16)],
    "EURJPY": [(0, 4), (7, 12)], "GBPJPY": [(7, 16)], "NZDUSD": [(0, 4), (7, 16)],
    "EURCHF": [(7, 12)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 16)],
    "EURNZD": [(0, 4), (7, 12)], "GBPNZD": [(7, 12)], "XAUUSD": [(7, 16)],
    "XAGUSD": [(7, 16)], "XPTUSD": [(7, 16)], "XAGGBP":[(7, 16)], "XAGEUR":[(7,16)], "XAGAUD": [(0,4), (7,10)], "BTCXAG":[(7,16)]
}
TRADING_SESSIONS_UTC["USOIL"] = [(12, 16)]
TRADING_SESSIONS_UTC["UKOIL"] = [(7, 16)]
CRYPTO_SESSIONS_USER = {"BTCUSD":[(7, 16)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 16)], "ETHUSD":[(7, 16)]}
for crypto_sym, sess_val in CRYPTO_SESSIONS_USER.items():
    TRADING_SESSIONS_UTC[crypto_sym] = sess_val


INITIAL_ACCOUNT_BALANCE = 200.00
RISK_PER_TRADE_PERCENT = 0.01 # Risk 1% of current balance per trade
DAILY_RISK_LIMIT_PERCENT = 0.05 # Daily risk limit of 5% of balance at start of day


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

        symbol_conceptual_start_balances = {}
        trades_per_symbol_map = {sym: [] for sym in SYMBOLS_AVAILABLE_FOR_TRADE}

        current_simulation_date = None
        daily_risk_allocated_on_current_date = 0.0
        max_daily_risk_budget_for_current_date = 0.0

        logger.info(f"Global Initial Account Balance: {shared_account_balance:.2f} USD")
        logger.info(f"Backtesting Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
        logger.info(f"Risk per trade: {RISK_PER_TRADE_PERCENT*100:.2f}%, Daily Risk Limit: {DAILY_RISK_LIMIT_PERCENT*100:.2f}%")
        logger.info("One trade at a time across the entire portfolio.")
        logger.info("Lot size will be fixed to minimum volume. Risk guard active for min lot.")


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
                logger.debug(f"Portfolio New Day: {current_simulation_date}. Max Daily Risk: {max_daily_risk_budget_for_current_date:.2f} (Bal: {shared_account_balance:.2f}). Daily allocated risk reset.")

            if global_active_trade:
                trade_symbol = global_active_trade['symbol']
                props = ALL_SYMBOL_PROPERTIES[trade_symbol]
                pip_adj_tsl = 3 * props['pip_value_calc']


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

                        global_active_trade['pnl_currency'] = pnl_ticks * props['trade_tick_value'] * global_active_trade['lot_size']
                        shared_account_balance += global_active_trade['pnl_currency']
                        global_active_trade['balance_after_trade'] = shared_account_balance

                        logger.info(f"[{trade_symbol}] {timestamp} Trade CLOSED ({global_active_trade['status']}): P&L: {global_active_trade['pnl_currency']:.2f}, New Portfolio Bal: {shared_account_balance:.2f}")
                        all_closed_trades_portfolio.append(global_active_trade.copy())
                        trades_per_symbol_map[trade_symbol].append(global_active_trade.copy())
                        global_active_trade = None

                    if global_active_trade and not closed_this_bar:
                        current_price_for_ts = current_candle_for_active_trade['high'] if global_active_trade['type'] == "BUY" else current_candle_for_active_trade['low']
                        entry_price_ts = global_active_trade['entry_price']
                        r_diff_ts = global_active_trade['r_value_price_diff']
                        ts_levels_ts = global_active_trade['ts_trigger_levels']
                        current_ts_level_idx = global_active_trade['next_ts_level_idx']

                        if current_ts_level_idx < len(ts_levels_ts):
                            r_multiple_target = ts_levels_ts[current_ts_level_idx]

                            target_price_for_ts_level = 0
                            if global_active_trade['type'] == "BUY":
                                target_price_for_ts_level = entry_price_ts + (r_multiple_target * r_diff_ts)
                            else:
                                target_price_for_ts_level = entry_price_ts - (r_multiple_target * r_diff_ts)

                            price_hit_ts_level = False
                            if global_active_trade['type'] == "BUY" and current_price_for_ts >= target_price_for_ts_level:
                                price_hit_ts_level = True
                            elif global_active_trade['type'] == "SELL" and current_price_for_ts <= target_price_for_ts_level:
                                price_hit_ts_level = True

                            if price_hit_ts_level:
                                if not global_active_trade['trailing_active']:
                                    global_active_trade['trailing_active'] = True
                                    logger.info(f"[{trade_symbol}] {timestamp} Trailing Stop ACTIVATED at {r_multiple_target}R. SL was {global_active_trade['sl']:.{props['digits']}f}")
                                else:
                                    logger.info(f"[{trade_symbol}] {timestamp} Trailing Stop Update Triggered at {r_multiple_target}R. SL was {global_active_trade['sl']:.{props['digits']}f}")

                                symbol_df_for_tsl = prepared_symbol_data[trade_symbol]
                                try:
                                    current_idx_in_symbol_df_tsl = symbol_df_for_tsl.index.get_loc(timestamp)
                                    if current_idx_in_symbol_df_tsl >= 2:
                                        last_3_candles_tsl = symbol_df_for_tsl.iloc[max(0, current_idx_in_symbol_df_tsl - 2) : current_idx_in_symbol_df_tsl + 1]

                                        new_sl_ts = 0
                                        if global_active_trade['type'] == "BUY":
                                            new_sl_ts = last_3_candles_tsl['low'].min() - pip_adj_tsl
                                            if new_sl_ts > global_active_trade['sl']:
                                                global_active_trade['sl'] = round(new_sl_ts, props['digits'])
                                                logger.debug(f"  [{trade_symbol}] {timestamp} TSL Updated BUY: New SL {global_active_trade['sl']:.{props['digits']}f}")
                                        else:
                                            new_sl_ts = last_3_candles_tsl['high'].max() + pip_adj_tsl
                                            if new_sl_ts < global_active_trade['sl']:
                                                global_active_trade['sl'] = round(new_sl_ts, props['digits'])
                                                logger.debug(f"  [{trade_symbol}] {timestamp} TSL Updated SELL: New SL {global_active_trade['sl']:.{props['digits']}f}")
                                    else:
                                        logger.debug(f"[{trade_symbol}] {timestamp} Not enough preceding candles for TSL adjustment at {r_multiple_target}R.")

                                except KeyError:
                                    logger.warning(f"Timestamp {timestamp} not found in {trade_symbol} df for TSL, skipping TSL update for level {r_multiple_target}R.")

                                global_active_trade['next_ts_level_idx'] += 1


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
                                    "symbol": order_symbol,
                                    "type": "BUY" if order_type_pending=="BUY_STOP" else "SELL",
                                    "entry_time": timestamp,
                                    "entry_price": actual_entry_price,
                                    "sl": sl_price_pending,
                                    "initial_sl": sl_price_pending,
                                    "tp": tp_price,
                                    "r_value_price_diff": risk_val_diff,
                                    "status": "OPEN",
                                    "lot_size": lot_size_pending,
                                    "pnl_currency": 0.0,
                                    "trailing_active": False,
                                    "ts_trigger_levels": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                                    "next_ts_level_idx": 0,
                                }
                                logger.info(f"  [{order_symbol}] Trade OPEN: {global_active_trade['type']} @{global_active_trade['entry_price']:.{props['digits']}f}, SL:{global_active_trade['sl']:.{props['digits']}f}, TP:{global_active_trade['tp']:.{props['digits']}f}, R-dist: {risk_val_diff:.{props['digits']}f}")
                                if order_symbol not in symbol_conceptual_start_balances:
                                    symbol_conceptual_start_balances[order_symbol] = shared_account_balance
                                global_pending_order = None

            if not global_active_trade and not global_pending_order:
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

                    m5_fanned_buy = m5_ema8>m5_ema13 and m5_ema13>m5_ema21_val
                    m5_fanned_sell = m5_ema8<m5_ema13 and m5_ema13<m5_ema21_val
                    is_fanned_for_bias = (h1_trend_bias_setup=="BUY" and m5_fanned_buy) or (h1_trend_bias_setup=="SELL" and m5_fanned_sell)
                    if not is_fanned_for_bias: continue

                    m5_setup_bias_setup = h1_trend_bias_setup

                    h4_ema8 = current_candle_for_setup.get('H4_EMA8', np.nan)
                    h4_ema21 = current_candle_for_setup.get('H4_EMA21', np.nan)
                    if pd.isna(h4_ema8) or pd.isna(h4_ema21): continue

                    if m5_setup_bias_setup == "BUY" and h4_ema8 < h4_ema21: continue
                    if m5_setup_bias_setup == "SELL" and h4_ema8 > h4_ema21: continue

                    # ✅ STEP 3: Add RSI Alignment Check to Entry Logic (Existing)
                    rsi_m5 = current_candle_for_setup.get('RSI_M5', np.nan)
                    rsi_h1 = current_candle_for_setup.get('RSI_H1', np.nan)
                    rsi_h4 = current_candle_for_setup.get('RSI_H4', np.nan)

                    if pd.isna(rsi_m5) or pd.isna(rsi_h1) or pd.isna(rsi_h4):
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} Missing RSI values. Skipping setup.")
                        continue

                    if m5_setup_bias_setup == "BUY" and not (rsi_m5 > 50 and rsi_h1 > 50 and rsi_h4 > 50):
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} RSI misalignment for BUY. M5:{rsi_m5:.1f} H1:{rsi_h1:.1f} H4:{rsi_h4:.1f}")
                        continue

                    if m5_setup_bias_setup == "SELL" and not (rsi_m5 < 50 and rsi_h1 < 50 and rsi_h4 < 50):
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} RSI misalignment for SELL. M5:{rsi_m5:.1f} H1:{rsi_h1:.1f} H4:{rsi_h4:.1f}")
                        continue
                    # End of RSI Alignment Check

                    if (m5_setup_bias_setup=="BUY" and current_candle_for_setup['close'] < m5_ema21_val) or \
                       (m5_setup_bias_setup=="SELL" and current_candle_for_setup['close'] > m5_ema21_val):
                        continue

                    pullback = (m5_setup_bias_setup=="BUY" and current_candle_for_setup['low']<=m5_ema8) or \
                               (m5_setup_bias_setup=="SELL" and current_candle_for_setup['high']>=m5_ema8)
                    if not pullback: continue

                    # --- START: Preceding Bar Weakness Filter ---
                    # ✅ STEP 2: Get the Last 4 Candles
                    symbol_df_for_weakness_filter = prepared_symbol_data[sym_to_check_setup]
                    try:
                        current_idx_for_weakness_filter = symbol_df_for_weakness_filter.index.get_loc(timestamp)
                    except KeyError:
                        logger.warning(f"[{sym_to_check_setup}] {timestamp} Timestamp not found in DataFrame for weakness filter. Skipping setup.")
                        continue

                    if current_idx_for_weakness_filter < 4:
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} Not enough preceding M5 candles for weakness filter ({current_idx_for_weakness_filter} total candles up to current). Skipping setup.")
                        continue

                    # Slice the last 4 full candles BEFORE the current one.
                    # e.g. if current_idx is 4 (5th candle, index 0,1,2,3,4), slice is iloc[0:4] giving candles 0,1,2,3
                    recent_candles = symbol_df_for_weakness_filter.iloc[current_idx_for_weakness_filter - 4 : current_idx_for_weakness_filter]

                    if len(recent_candles) < 4: # Should be redundant due to check above, but as a safeguard
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} Sliced less than 4 preceding M5 candles for weakness filter ({len(recent_candles)} found). Skipping setup.")
                        continue

                    # ✅ STEP 3: Apply the Directional Count Filter
                    bullish_count = (recent_candles['close'] > recent_candles['open']).sum()
                    bearish_count = (recent_candles['close'] < recent_candles['open']).sum()

                    # For BUY setup
                    if m5_setup_bias_setup == "BUY" and bullish_count > 2:
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} Preceding Bar Weakness Filter: Too many prior bullish M5 candles ({bullish_count}/4). Skipping BUY setup.")
                        continue

                    # For SELL setup
                    if m5_setup_bias_setup == "SELL" and bearish_count > 2:
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} Preceding Bar Weakness Filter: Too many prior bearish M5 candles ({bearish_count}/4). Skipping SELL setup.")
                        continue
                    # --- END: Preceding Bar Weakness Filter ---


                    atr_val = current_candle_for_setup.get('ATR', np.nan)
                    if pd.isna(atr_val) or atr_val <= 0:
                        continue

                    sl_distance_atr = 1.5 * atr_val

                    symbol_df_for_lookback = prepared_symbol_data[sym_to_check_setup]
                    try:
                        current_idx_in_symbol_df_for_lookback = symbol_df_for_lookback.index.get_loc(timestamp)
                    except KeyError: continue

                    if current_idx_in_symbol_df_for_lookback < 4:
                        continue

                    lookback_df_for_entry = symbol_df_for_lookback.iloc[current_idx_in_symbol_df_for_lookback-4 : current_idx_in_symbol_df_for_lookback+1]

                    entry_px, sl_px, order_type_setup = (0,0,"")

                    if m5_setup_bias_setup=="BUY":
                        entry_px = lookback_df_for_entry['high'].max() + pip_adj_setup
                        sl_px = entry_px - sl_distance_atr
                        order_type_setup="BUY_STOP"
                    else: # SELL
                        entry_px = lookback_df_for_entry['low'].min() - pip_adj_setup
                        sl_px = entry_px + sl_distance_atr
                        order_type_setup="SELL_STOP"

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

                    if daily_risk_allocated_on_current_date + estimated_risk_min_lot > max_daily_risk_budget_for_current_date + 1e-9:
                        logger.info(f"[{sym_to_check_setup}] {timestamp} Portfolio Daily Risk Limit Reached for {current_simulation_date}. "
                                    f"Max: {max_daily_risk_budget_for_current_date:.2f}, "
                                    f"Allocated: {daily_risk_allocated_on_current_date:.2f}, "
                                    f"Intended for this trade (min lot): {estimated_risk_min_lot:.2f}. Skipping pending order.")
                        continue

                    global_pending_order = {"symbol":sym_to_check_setup, "type":order_type_setup,
                                            "entry_price":entry_px, "sl_price":sl_px,
                                            "created_time":timestamp, "lot_size":lot_size_fixed_min,
                                            "setup_bias": m5_setup_bias_setup,
                                            "intended_risk_amount": estimated_risk_min_lot}

                    daily_risk_allocated_on_current_date += estimated_risk_min_lot
                    logger.info(f"[{sym_to_check_setup}] {timestamp} GLOBAL PENDING {order_type_setup} Order Set: Entry {entry_px:.{props_setup['digits']}f}, SL {sl_px:.{props_setup['digits']}f} (ATR: {atr_val:.{props_setup['digits']}f}), Lot {lot_size_fixed_min} (Min Vol). Estimated Risk: {estimated_risk_min_lot:.2f}")
                    logger.debug(f"  Portfolio daily risk allocated: {daily_risk_allocated_on_current_date:.2f}/{max_daily_risk_budget_for_current_date:.2f}")
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
            
            # --- NEW: Call the RR analysis function ---
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

            # --- NEW: Print the RR distribution summary ---
            logger.info("\n--- RR Distribution Summary ---")
            total_counted_trades = sum(rr_distribution.values())
            logger.info(f"  (Analysis based on {total_counted_trades} of {overall_stats['total_trades']} total trades)")
            for bucket, count in rr_distribution.items():
                if count > 0: # Only display buckets that have trades in them
                    percentage = (count / total_counted_trades) * 100 if total_counted_trades > 0 else 0
                    logger.info(f"  {bucket:<25}: {count:<5} trades ({percentage:.2f}%)")
        else:
            logger.info("No trades were executed across any symbols during the backtest period.")
            logger.info(f"Overall Starting Balance: {INITIAL_ACCOUNT_BALANCE:.2f} USD")
            logger.info(f"Overall Ending Balance: {shared_account_balance:.2f} USD")

        shutdown_mt5_interface()