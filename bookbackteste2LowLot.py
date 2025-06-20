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
SYMBOLS_TO_BACKTEST = ["EURUSD", "AUDUSD", "USDCHF", "USDCAD",
                       "GBPJPY",  "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                       "USOIL", "CADJPY", "XAGUSD", "XPTUSD", "UKOIL",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD", "XAGGBP", "XAGEUR", "XAGAUD", "BTCXAG" ]

TRADING_SESSIONS_UTC = { # (start_hour_inclusive, end_hour_exclusive)
    "EURUSD": [(7, 16)], "GBPUSD": [(7, 16)], "AUDUSD": [(0, 4), (7, 16)],
    "USDCHF": [(7, 16)], "USDCAD": [(12, 17)], "USDJPY": [(0, 4), (12, 17)],
    "EURJPY": [(0, 4), (7, 12)], "GBPJPY": [(7, 16)], "NZDUSD": [(0, 4), (7, 16)],
    "EURCHF": [(7, 12)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 17)],
    "EURNZD": [(0, 4), (7, 12)], "GBPNZD": [(7, 12)], "XAUUSD": [(7, 16)],
    "XAGUSD": [(7, 16)], "XPTUSD": [(7, 16)], "XAGGBP":[(7, 16)], "XAGEUR":[(7,16)], "XAGAUD": [(0,4), (7,10)], "BTCXAG":[(7,16)]
}
TRADING_SESSIONS_UTC["USOIL"] = [(12, 17)]
TRADING_SESSIONS_UTC["UKOIL"] = [(7, 16)]
CRYPTO_SESSIONS_USER = {"BTCUSD":[(0, 16)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 16)], "ETHUSD":[(0, 16)]}
for crypto_sym, sess_val in CRYPTO_SESSIONS_USER.items():
    TRADING_SESSIONS_UTC[crypto_sym] = sess_val


INITIAL_ACCOUNT_BALANCE = 200.00
RISK_PER_TRADE_PERCENT = 0.01 # Max Risk 1% of current balance per trade (used for new risk filter upper bound)
MIN_RISK_PER_TRADE_PERCENT_FOR_MIN_LOT = 0.000 # Min Risk 0.7% for min lot trade (new risk filter lower bound)
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
            'volume_min': symbol_info_obj.volume_min if symbol_info_obj.volume_min > 0 else 0.01, # Ensure volume_min is not zero
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

# --- calculate_lot_size is no longer used by the new trade setup logic, but kept for potential future use or other strategies ---
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
    df_h1 = get_historical_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if df_h1.empty:
        logger.warning(f"No H1 data for {symbol}. H1 dependent indicators will be NaN.")
        df_h1_resampled = pd.DataFrame(columns=['H1_Close_For_Bias', 'H1_EMA8', 'H1_EMA21']) 
    else:
        df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
        df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
        df_h1_resampled = df_h1[['close', 'H1_EMA8', 'H1_EMA21']].rename(columns={'close': 'H1_Close_For_Bias'})

    df_h4 = get_historical_data(symbol, mt5.TIMEFRAME_H4, start_date, end_date)
    h4_data_available = not df_h4.empty
    if h4_data_available:
        df_h4['H4_EMA8'] = ta.ema(df_h4['close'], length=8)
        df_h4['H4_EMA21'] = ta.ema(df_h4['close'], length=21)
        df_h4_resampled = df_h4[['H4_EMA8', 'H4_EMA21']].copy()
    else:
        logger.warning(f"No H4 data for {symbol}. H4 dependent indicators will be NaN.")

    df_m5 = get_historical_data(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    if df_m5.empty: return pd.DataFrame() 
    df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
    df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
    df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
    if len(df_m5) >= 14: 
        df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
    else:
        df_m5['ATR'] = np.nan 

    combined_df = pd.merge_asof(df_m5.sort_index(), df_h1_resampled.sort_index(),
                                left_index=True, right_index=True,
                                direction='backward', tolerance=pd.Timedelta(hours=1))

    if h4_data_available:
        combined_df = pd.merge_asof(combined_df.sort_index(), df_h4_resampled.sort_index(),
                                    left_index=True, right_index=True,
                                    direction='backward', tolerance=pd.Timedelta(hours=4))
    else:
        combined_df['H4_EMA8'] = np.nan
        combined_df['H4_EMA21'] = np.nan
        
    combined_df.dropna(subset=['open', 'high', 'low', 'close', # M5 Base
                                'M5_EMA8', 'M5_EMA13', 'M5_EMA21', 'ATR', # M5 Indicators
                                'H1_Close_For_Bias', 'H1_EMA8', 'H1_EMA21', # H1 Indicators
                                'H4_EMA8', 'H4_EMA21'], # H4 Indicators
                       how='any', inplace=True) # Drop if any of these critical indicators are NaN
    return combined_df


# --- Main Execution ---
if __name__ == "__main__":
    start_datetime = datetime(2025, 5, 1) 
    end_datetime = datetime(2025, 5, 31) 
    
    buffer_days = 30 # Increased buffer for ATR and longer EMAs to stabilize
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
        logger.info(f"Risk Filter for Min Lot: {MIN_RISK_PER_TRADE_PERCENT_FOR_MIN_LOT*100:.2f}% - {RISK_PER_TRADE_PERCENT*100:.2f}%, Daily Risk Limit: {DAILY_RISK_LIMIT_PERCENT*100:.2f}%")
        logger.info("One trade at a time across the entire portfolio.")

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
                    logger.warning(f"No data for {sym} within the backtest period {start_datetime} - {end_datetime} after initial buffer fetch and NaN drop. Skipping.")
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
                            else: # SELL
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
                                        else: # SELL
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
                            
                            r_value_diff_for_active_trade = global_pending_order['r_value_price_diff_initial_calc']
                            
                            tp_price_active = 0
                            if order_type_pending == "BUY_STOP": # Actual trade is BUY
                                tp_price_active = actual_entry_price + (4 * r_value_diff_for_active_trade)
                            else: # SELL_STOP, actual trade is SELL
                                tp_price_active = actual_entry_price - (4 * r_value_diff_for_active_trade)
                            tp_price_active = round(tp_price_active, props['digits'])

                            final_sl_for_active_trade = sl_price_pending 

                            if (order_type_pending == "BUY_STOP" and actual_entry_price <= final_sl_for_active_trade) or \
                               (order_type_pending == "SELL_STOP" and actual_entry_price >= final_sl_for_active_trade) or \
                               r_value_diff_for_active_trade <= (props['trade_tick_size'] * 2): # Min 2 ticks SL distance
                                logger.warning(f"[{order_symbol}] Invalid SL/Entry ({actual_entry_price:.{props['digits']}f} vs {final_sl_for_active_trade:.{props['digits']}f}) or R-diff ({r_value_diff_for_active_trade:.{props['digits']}f}) on trigger. Cancelling order and refunding risk.")
                                daily_risk_allocated_on_current_date -= global_pending_order['intended_risk_amount'] 
                                global_pending_order = None
                            else:
                                global_active_trade = {
                                    "symbol": order_symbol,
                                    "type": "BUY" if order_type_pending=="BUY_STOP" else "SELL",
                                    "entry_time": timestamp,
                                    "entry_price": actual_entry_price,
                                    "sl": final_sl_for_active_trade, 
                                    "initial_sl": final_sl_for_active_trade,
                                    "tp": tp_price_active, 
                                    "r_value_price_diff": r_value_diff_for_active_trade, 
                                    "status": "OPEN",
                                    "lot_size": lot_size_pending,
                                    "pnl_currency": 0.0,
                                    "trailing_active": False,
                                    "ts_trigger_levels": global_pending_order['ts_trigger_levels_pending'],
                                    "next_ts_level_idx": global_pending_order['next_ts_level_idx_pending'],
                                }
                                logger.info(f"  [{order_symbol}] Trade OPEN: {global_active_trade['type']} @{global_active_trade['entry_price']:.{props['digits']}f}, SL:{global_active_trade['sl']:.{props['digits']}f}, TP:{global_active_trade['tp']:.{props['digits']}f}, R-dist: {global_active_trade['r_value_price_diff']:.{props['digits']}f}, Lot: {global_active_trade['lot_size']}")
                                if order_symbol not in symbol_conceptual_start_balances:
                                    symbol_conceptual_start_balances[order_symbol] = shared_account_balance 
                                global_pending_order = None

            if not global_active_trade and not global_pending_order: 
                for sym_to_check_setup in SYMBOLS_AVAILABLE_FOR_TRADE:
                    if sym_to_check_setup not in prepared_symbol_data or timestamp not in prepared_symbol_data[sym_to_check_setup].index:
                        continue 

                    current_candle_for_setup = prepared_symbol_data[sym_to_check_setup].loc[timestamp]
                    props_setup = ALL_SYMBOL_PROPERTIES[sym_to_check_setup]
                    
                    # Used for entry trigger adjustment (e.g., 3 ticks beyond high/low)
                    pip_adj_entry_trigger = 3 * props_setup['trade_tick_size'] 

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
                    
                    m5_setup_bias_setup = h1_trend_bias_setup # This is the trade_direction

                    h4_ema8 = current_candle_for_setup.get('H4_EMA8', np.nan) 
                    h4_ema21 = current_candle_for_setup.get('H4_EMA21', np.nan)
                    if pd.isna(h4_ema8) or pd.isna(h4_ema21): continue 
                    
                    if m5_setup_bias_setup == "BUY" and h4_ema8 < h4_ema21: continue
                    if m5_setup_bias_setup == "SELL" and h4_ema8 > h4_ema21: continue

                    if (m5_setup_bias_setup=="BUY" and current_candle_for_setup['close'] < m5_ema21_val) or \
                       (m5_setup_bias_setup=="SELL" and current_candle_for_setup['close'] > m5_ema21_val):
                        continue 

                    pullback = (m5_setup_bias_setup=="BUY" and current_candle_for_setup['low']<=m5_ema8) or \
                               (m5_setup_bias_setup=="SELL" and current_candle_for_setup['high']>=m5_ema8)
                    if not pullback: continue
                    
                    atr_val = current_candle_for_setup.get('ATR', np.nan)
                    if pd.isna(atr_val) or atr_val <= 0:
                        continue
                    
                    symbol_df_for_lookback = prepared_symbol_data[sym_to_check_setup]
                    try:
                        current_idx_in_symbol_df_for_lookback = symbol_df_for_lookback.index.get_loc(timestamp)
                    except KeyError: continue 

                    if current_idx_in_symbol_df_for_lookback < 4: # Need 5 candles for lookback_df (0 to 4)
                        continue
                    
                    # lookback_df_for_entry includes the current candle_for_setup
                    lookback_df_for_entry = symbol_df_for_lookback.iloc[max(0, current_idx_in_symbol_df_for_lookback - 4) : current_idx_in_symbol_df_for_lookback + 1]
                    if len(lookback_df_for_entry) < 5: continue # Ensure 5 candles for swing high/low

                    # === START OF NEW LOGIC INTEGRATION ===
                    # === 1. Get symbol properties ===
                    volume_min_lot = props_setup.get("volume_min", 0.01)
                    lot_size_trade = volume_min_lot # Fixed lot size
                    digits_trade = props_setup["digits"]
                    tick_val_trade = props_setup["trade_tick_value"]
                    tick_size_trade = props_setup["trade_tick_size"]
                    pip_value_trade = props_setup["pip_value_calc"]

                    trade_direction_setup = m5_setup_bias_setup # "BUY" or "SELL"

                    # Calculate entry trigger price (for stop order)
                    entry_px_trigger = 0
                    order_type_setup = ""
                    if trade_direction_setup == "BUY":
                        entry_px_trigger = lookback_df_for_entry['high'].max() + pip_adj_entry_trigger
                        order_type_setup = "BUY_STOP"
                    elif trade_direction_setup == "SELL":
                        entry_px_trigger = lookback_df_for_entry['low'].min() - pip_adj_entry_trigger
                        order_type_setup = "SELL_STOP"
                    else: # Should not happen due to earlier checks
                        logger.warning(f"[{sym_to_check_setup}] {timestamp} Unknown trade direction '{trade_direction_setup}'. Skipping.")
                        continue
                    entry_px_trigger = round(entry_px_trigger, digits_trade)

                    # === 2. SL Distance from ATR ===
                    atr_multiplier = 1.5
                    atr_sl_price_diff = atr_multiplier * atr_val

                    # === 3. Swing High/Low based SL Component ===
                    sl_buffer_pips = 3 
                    sl_buffer_price_units = sl_buffer_pips * pip_value_trade # Buffer in price terms

                    raw_sl_price_trade = 0
                    sl_price_diff_trade = 0 # This is the R-value distance
                    # tp_price_trade will be calculated after sl_price_diff_trade

                    if trade_direction_setup == "BUY":
                        swing_sl_price = lookback_df_for_entry['low'].min() - sl_buffer_price_units
                        atr_sl_price = entry_px_trigger - atr_sl_price_diff
                        raw_sl_price_trade = min(atr_sl_price, swing_sl_price) # Wider stop
                        sl_price_diff_trade = entry_px_trigger - raw_sl_price_trade
                    elif trade_direction_setup == "SELL":
                        swing_sl_price = lookback_df_for_entry['high'].max() + sl_buffer_price_units
                        atr_sl_price = entry_px_trigger + atr_sl_price_diff
                        raw_sl_price_trade = max(atr_sl_price, swing_sl_price) # Wider stop
                        sl_price_diff_trade = raw_sl_price_trade - entry_px_trigger
                    
                    if sl_price_diff_trade <= (tick_size_trade * 2): # Min SL distance, e.g. 2 ticks
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} SL distance too small ({sl_price_diff_trade:.{digits_trade}f}) for {trade_direction_setup}. Entry: {entry_px_trigger:.{digits_trade}f}, RawSL: {raw_sl_price_trade:.{digits_trade}f}. Skipping.")
                        continue

                    tp_price_trade = 0
                    if trade_direction_setup == "BUY":
                        tp_price_trade = entry_px_trigger + 4 * sl_price_diff_trade
                    elif trade_direction_setup == "SELL":
                        tp_price_trade = entry_px_trigger - 4 * sl_price_diff_trade


                    # === 4. Risk Estimation (using min_lot) ===
                    account_balance_for_risk_est = shared_account_balance
                    risk_usd_trade = 0
                    if tick_size_trade > 0 and tick_val_trade > 0:
                        risk_usd_trade = lot_size_trade * (sl_price_diff_trade / tick_size_trade) * tick_val_trade
                    else:
                        logger.warning(f"[{sym_to_check_setup}] {timestamp} Tick size or value is zero. Cannot calculate risk. Skipping.")
                        continue
                    
                    risk_pct_trade = risk_usd_trade / account_balance_for_risk_est if account_balance_for_risk_est > 0 else float('inf')

                    # === 5. Risk Filters ===
                    if risk_pct_trade > RISK_PER_TRADE_PERCENT:
                        logger.info(f"[{sym_to_check_setup}] {timestamp} SKIP PENDING: Risk with min lot ({risk_pct_trade:.2%}) > max_risk_target ({RISK_PER_TRADE_PERCENT:.2%}). SL Diff: {sl_price_diff_trade:.{digits_trade}f}, Lot: {lot_size_trade}")
                        continue
                    elif risk_pct_trade < MIN_RISK_PER_TRADE_PERCENT_FOR_MIN_LOT:
                        logger.info(f"[{sym_to_check_setup}] {timestamp} SKIP PENDING: Risk with min lot ({risk_pct_trade:.2%}) < min_risk_target ({MIN_RISK_PER_TRADE_PERCENT_FOR_MIN_LOT:.2%}). SL Diff: {sl_price_diff_trade:.{digits_trade}f}, Lot: {lot_size_trade}")
                        continue
                    
                    # Daily risk check using the calculated risk_usd_trade
                    if daily_risk_allocated_on_current_date + risk_usd_trade > max_daily_risk_budget_for_current_date + 1e-9: 
                        logger.info(f"[{sym_to_check_setup}] {timestamp} Portfolio Daily Risk Limit Reached for {current_simulation_date}. "
                                    f"Max: {max_daily_risk_budget_for_current_date:.2f}, "
                                    f"Allocated: {daily_risk_allocated_on_current_date:.2f}, "
                                    f"Intended for this trade: {risk_usd_trade:.2f}. Skipping trade.")
                        continue 

                    # === 6. Round SL/TP ===
                    # entry_px_trigger is already rounded
                    final_sl_price = round(raw_sl_price_trade, digits_trade)
                    final_tp_price = round(tp_price_trade, digits_trade)

                    # Final check: SL should not have rounded through entry
                    if trade_direction_setup == "BUY" and final_sl_price >= entry_px_trigger:
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} BUY SL ({final_sl_price:.{digits_trade}f}) rounded to be >= Entry ({entry_px_trigger:.{digits_trade}f}). Skipping.")
                        continue
                    if trade_direction_setup == "SELL" and final_sl_price <= entry_px_trigger:
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} SELL SL ({final_sl_price:.{digits_trade}f}) rounded to be <= Entry ({entry_px_trigger:.{digits_trade}f}). Skipping.")
                        continue
                    
                    # Final check: TP should not be at or worse than entry
                    if trade_direction_setup == "BUY" and final_tp_price <= entry_px_trigger:
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} BUY TP ({final_tp_price:.{digits_trade}f}) rounded to be <= Entry ({entry_px_trigger:.{digits_trade}f}). Skipping.")
                        continue
                    if trade_direction_setup == "SELL" and final_tp_price >= entry_px_trigger:
                        logger.debug(f"[{sym_to_check_setup}] {timestamp} SELL TP ({final_tp_price:.{digits_trade}f}) rounded to be >= Entry ({entry_px_trigger:.{digits_trade}f}). Skipping.")
                        continue


                    # === 7. Place Trade (Create Pending Order) ===
                    global_pending_order = {
                        "symbol": sym_to_check_setup,
                        "type": order_type_setup, # BUY_STOP or SELL_STOP
                        "entry_price": entry_px_trigger, 
                        "sl_price": final_sl_price,
                        "tp_price_initial_calc": final_tp_price, # For active trade creation
                        "lot_size": lot_size_trade, 
                        "r_value_price_diff_initial_calc": sl_price_diff_trade, # For active trade
                        "created_time": timestamp,
                        "setup_bias": trade_direction_setup, # "BUY" or "SELL"
                        "intended_risk_amount": risk_usd_trade,
                        "ts_trigger_levels_pending": [1.5, 2.0, 2.5, 3.0, 3.5], # From snippet
                        "next_ts_level_idx_pending": 0 # From snippet
                    }
                    
                    daily_risk_allocated_on_current_date += risk_usd_trade
                    logger.info(f"[{sym_to_check_setup}] {timestamp} GLOBAL PENDING {order_type_setup} Order Set: Entry {entry_px_trigger:.{digits_trade}f}, SL {final_sl_price:.{digits_trade}f}, TP (calc) {final_tp_price:.{digits_trade}f}, Lot {lot_size_trade}, Risk {risk_pct_trade:.2%} ({risk_usd_trade:.2f} USD)")
                    logger.debug(f"  Portfolio daily risk allocated: {daily_risk_allocated_on_current_date:.2f}/{max_daily_risk_budget_for_current_date:.2f}")
                    break # Found a trade setup, exit symbol loop for this timestamp
                    # === END OF NEW LOGIC INTEGRATION ===
        
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
        else:
            logger.info("No trades were executed across any symbols during the backtest period.")
            logger.info(f"Overall Starting Balance: {INITIAL_ACCOUNT_BALANCE:.2f} USD")
            logger.info(f"Overall Ending Balance: {shared_account_balance:.2f} USD")
 
        shutdown_mt5_interface()