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


INITIAL_ACCOUNT_BALANCE_PER_SYMBOL = 200.00
RISK_PER_TRADE_PERCENT = 0.01 # Risk 1% of current symbol's balance per trade


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
            'pip_value_calc': current_pip_value, 
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
        
    # No aggressive dropna here; will be handled after filtering for backtest period
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
        logger.info(f"Parallel Individual Symbol Backtesting Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
        logger.info(f"Initial Balance per Symbol: {INITIAL_ACCOUNT_BALANCE_PER_SYMBOL:.2f} USD")
        logger.info(f"Risk per trade: {RISK_PER_TRADE_PERCENT*100:.2f}% of current symbol's balance.")

        # Initialize per-symbol states
        symbol_account_balances = {sym: INITIAL_ACCOUNT_BALANCE_PER_SYMBOL for sym in SYMBOLS_AVAILABLE_FOR_TRADE}
        symbol_active_trades = {sym: None for sym in SYMBOLS_AVAILABLE_FOR_TRADE}
        symbol_pending_orders = {sym: None for sym in SYMBOLS_AVAILABLE_FOR_TRADE}
        symbol_closed_trades_map = {sym: [] for sym in SYMBOLS_AVAILABLE_FOR_TRADE}

        # Prepare data for all symbols and create master time index
        full_prepared_data_map = {}
        backtest_data_for_simulation_map = {}
        master_time_index_set = set()

        for sym in SYMBOLS_AVAILABLE_FOR_TRADE:
            props = ALL_SYMBOL_PROPERTIES[sym]
            # Fetch full data including buffer
            df_full = prepare_symbol_data(sym, data_fetch_start_date, end_datetime, props)
            if not df_full.empty:
                full_prepared_data_map[sym] = df_full
                
                # Filter for actual backtest period and drop initial NaNs for critical indicators
                df_for_simulation = df_full[
                    (df_full.index >= pd.Timestamp(start_datetime, tz='UTC')) &
                    (df_full.index <= pd.Timestamp(end_datetime, tz='UTC'))
                ].copy()
                df_for_simulation.dropna(subset=['M5_EMA8', 'M5_EMA13', 'M5_EMA21', 'ATR'], inplace=True)

                if not df_for_simulation.empty:
                    backtest_data_for_simulation_map[sym] = df_for_simulation
                    master_time_index_set.update(df_for_simulation.index)
                else:
                    logger.warning(f"No usable data for {sym} in backtest period after NaN drop. It will be skipped.")
            else:
                logger.warning(f"No data prepared for {sym} at all. It will be skipped.")
        
        if not master_time_index_set:
            logger.error("No data available for any symbol in the specified range for master time index. Exiting.")
            shutdown_mt5_interface()
            exit()

        master_time_index = sorted(list(master_time_index_set))
        logger.info(f"Master time index created with {len(master_time_index)} M5 candles to process across all symbols.")

        # --- Main Simulation Loop (Iterating through master_time_index) ---
        for timestamp in master_time_index:
            for sym_to_process in SYMBOLS_AVAILABLE_FOR_TRADE:
                # Skip if symbol has no data for simulation or no data at this specific timestamp
                if sym_to_process not in backtest_data_for_simulation_map or \
                   timestamp not in backtest_data_for_simulation_map[sym_to_process].index:
                    continue

                current_candle_data = backtest_data_for_simulation_map[sym_to_process].loc[timestamp]
                props = ALL_SYMBOL_PROPERTIES[sym_to_process]
                
                # Get current state for this symbol
                current_symbol_balance = symbol_account_balances[sym_to_process]
                active_trade_for_sym = symbol_active_trades[sym_to_process]
                pending_order_for_sym = symbol_pending_orders[sym_to_process]

                # --- Manage Active Trade for the current symbol ---
                if active_trade_for_sym:
                    pip_adj_tsl = 3 * props['pip_value_calc'] 
                    closed_this_bar = False
                    exit_price = 0

                    if active_trade_for_sym['type'] == "BUY" and current_candle_data['low'] <= active_trade_for_sym['sl']:
                        exit_price = active_trade_for_sym['sl']; active_trade_for_sym['status'] = "SL_HIT"
                    elif active_trade_for_sym['type'] == "SELL" and current_candle_data['high'] >= active_trade_for_sym['sl']:
                        exit_price = active_trade_for_sym['sl']; active_trade_for_sym['status'] = "SL_HIT"
                    
                    if active_trade_for_sym['status'] == "OPEN": 
                        if active_trade_for_sym['type'] == "BUY" and current_candle_data['high'] >= active_trade_for_sym['tp']:
                            exit_price = active_trade_for_sym['tp']; active_trade_for_sym['status'] = "TP_HIT"
                        elif active_trade_for_sym['type'] == "SELL" and current_candle_data['low'] <= active_trade_for_sym['tp']:
                            exit_price = active_trade_for_sym['tp']; active_trade_for_sym['status'] = "TP_HIT"
                    
                    if active_trade_for_sym['status'] != "OPEN":
                        closed_this_bar = True
                        active_trade_for_sym['exit_time'] = timestamp
                        active_trade_for_sym['exit_price'] = exit_price
                        price_diff = (exit_price - active_trade_for_sym['entry_price']) if active_trade_for_sym['type'] == "BUY" else (active_trade_for_sym['entry_price'] - exit_price)
                        
                        if props['trade_tick_size'] > 0:
                            pnl_ticks = price_diff / props['trade_tick_size']
                        else: 
                            pnl_ticks = 0
                            logger.error(f"[{sym_to_process}] trade_tick_size is zero or invalid at PNL calculation.")

                        active_trade_for_sym['pnl_currency'] = pnl_ticks * props['trade_tick_value'] * active_trade_for_sym['lot_size']
                        symbol_account_balances[sym_to_process] += active_trade_for_sym['pnl_currency']
                        active_trade_for_sym['balance_after_trade'] = symbol_account_balances[sym_to_process]
                        
                        logger.info(f"[{sym_to_process}] {timestamp} Trade CLOSED ({active_trade_for_sym['status']}): P&L: {active_trade_for_sym['pnl_currency']:.2f}, New Bal ({sym_to_process}): {symbol_account_balances[sym_to_process]:.2f}")
                        symbol_closed_trades_map[sym_to_process].append(active_trade_for_sym.copy())
                        symbol_active_trades[sym_to_process] = None
                        active_trade_for_sym = None # Update local copy
                    
                    # --- Trailing Stop Logic for the current symbol's active trade ---
                    if active_trade_for_sym and not closed_this_bar: # Re-check active_trade_for_sym as it might have been closed
                        current_price_for_ts = current_candle_data['high'] if active_trade_for_sym['type'] == "BUY" else current_candle_data['low']
                        entry_price_ts = active_trade_for_sym['entry_price']
                        r_diff_ts = active_trade_for_sym['r_value_price_diff']
                        ts_levels_ts = active_trade_for_sym['ts_trigger_levels']
                        current_ts_level_idx = active_trade_for_sym['next_ts_level_idx']

                        if current_ts_level_idx < len(ts_levels_ts):
                            r_multiple_target = ts_levels_ts[current_ts_level_idx]
                            target_price_for_ts_level = 0
                            if active_trade_for_sym['type'] == "BUY":
                                target_price_for_ts_level = entry_price_ts + (r_multiple_target * r_diff_ts)
                            else: # SELL
                                target_price_for_ts_level = entry_price_ts - (r_multiple_target * r_diff_ts)

                            price_hit_ts_level = False
                            if active_trade_for_sym['type'] == "BUY" and current_price_for_ts >= target_price_for_ts_level:
                                price_hit_ts_level = True
                            elif active_trade_for_sym['type'] == "SELL" and current_price_for_ts <= target_price_for_ts_level:
                                price_hit_ts_level = True

                            if price_hit_ts_level:
                                if not active_trade_for_sym['trailing_active']:
                                    active_trade_for_sym['trailing_active'] = True
                                    logger.info(f"[{sym_to_process}] {timestamp} Trailing Stop ACTIVATED at {r_multiple_target}R. SL was {active_trade_for_sym['sl']:.{props['digits']}f}")
                                else:
                                    logger.info(f"[{sym_to_process}] {timestamp} Trailing Stop Update Triggered at {r_multiple_target}R. SL was {active_trade_for_sym['sl']:.{props['digits']}f}")
                                
                                # Use the full prepared data (including buffer) for TSL lookback
                                symbol_df_for_tsl_full = full_prepared_data_map[sym_to_process]
                                try:
                                    current_idx_in_symbol_df_tsl = symbol_df_for_tsl_full.index.get_loc(timestamp)
                                    if current_idx_in_symbol_df_tsl >= 2: 
                                        last_3_candles_tsl = symbol_df_for_tsl_full.iloc[max(0, current_idx_in_symbol_df_tsl - 2) : current_idx_in_symbol_df_tsl + 1]
                                        new_sl_ts = 0
                                        if active_trade_for_sym['type'] == "BUY":
                                            new_sl_ts = last_3_candles_tsl['low'].min() - pip_adj_tsl
                                            if new_sl_ts > active_trade_for_sym['sl']:
                                                active_trade_for_sym['sl'] = round(new_sl_ts, props['digits'])
                                                logger.debug(f"  [{sym_to_process}] {timestamp} TSL Updated BUY: New SL {active_trade_for_sym['sl']:.{props['digits']}f}")
                                        else: # SELL
                                            new_sl_ts = last_3_candles_tsl['high'].max() + pip_adj_tsl
                                            if new_sl_ts < active_trade_for_sym['sl']:
                                                active_trade_for_sym['sl'] = round(new_sl_ts, props['digits'])
                                                logger.debug(f"  [{sym_to_process}] {timestamp} TSL Updated SELL: New SL {active_trade_for_sym['sl']:.{props['digits']}f}")
                                    else:
                                        logger.debug(f"[{sym_to_process}] {timestamp} Not enough preceding candles for TSL adjustment at {r_multiple_target}R.")
                                except KeyError:
                                    logger.warning(f"Timestamp {timestamp} not found in {sym_to_process} full df for TSL, skipping TSL update for level {r_multiple_target}R.")
                                active_trade_for_sym['next_ts_level_idx'] += 1
                
                # --- Manage Pending Order for the current symbol ---
                # Check 'not symbol_active_trades[sym_to_process]' because an order might trigger and become active in the same candle tick
                if not symbol_active_trades[sym_to_process] and pending_order_for_sym:
                    entry_price_pending = pending_order_for_sym['entry_price'] 
                    sl_price_pending = pending_order_for_sym['sl_price']       
                    order_type_pending = pending_order_for_sym['type']         
                    lot_size_pending = pending_order_for_sym['lot_size']       
                    
                    m5_ema21_for_invalidation = current_candle_data['M5_EMA21'] 
                    
                    setup_invalidated = False
                    if pending_order_for_sym['setup_bias'] == "BUY" and current_candle_data['close'] < m5_ema21_for_invalidation:
                        setup_invalidated = True
                        logger.info(f"[{sym_to_process}] {timestamp} PENDING BUY order invalidated (Close < M5_EMA21 before trigger).")
                    elif pending_order_for_sym['setup_bias'] == "SELL" and current_candle_data['close'] > m5_ema21_for_invalidation:
                        setup_invalidated = True
                        logger.info(f"[{sym_to_process}] {timestamp} PENDING SELL order invalidated (Close > M5_EMA21 before trigger).")

                    if setup_invalidated:
                        symbol_pending_orders[sym_to_process] = None 
                    else:
                        triggered = False; actual_entry_price = 0
                        if order_type_pending == "BUY_STOP" and current_candle_data['high'] >= entry_price_pending:
                            actual_entry_price = entry_price_pending; triggered = True 
                        elif order_type_pending == "SELL_STOP" and current_candle_data['low'] <= entry_price_pending:
                            actual_entry_price = entry_price_pending; triggered = True 
                        
                        if triggered:
                            logger.info(f"[{sym_to_process}] {timestamp} PENDING {order_type_pending} TRIGGERED @{actual_entry_price:.{props['digits']}f} Lot:{lot_size_pending}")
                            risk_val_diff = abs(actual_entry_price - sl_price_pending) 
                            
                            if risk_val_diff <= 0 or lot_size_pending <= 0:
                                logger.warning(f"[{sym_to_process}] Invalid risk (diff {risk_val_diff}) /lot ({lot_size_pending}) on trigger. Cancelling order.")
                                symbol_pending_orders[sym_to_process] = None
                            else:
                                tp_price = actual_entry_price + (4 * risk_val_diff) if order_type_pending=="BUY_STOP" else actual_entry_price - (4 * risk_val_diff)
                                tp_price = round(tp_price, props['digits'])

                                symbol_active_trades[sym_to_process] = {
                                    "symbol": sym_to_process,
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
                                    "ts_trigger_levels": [1.5, 2.0, 2.5, 3.0, 3.5],
                                    "next_ts_level_idx": 0,
                                }
                                logger.info(f"  [{sym_to_process}] Trade OPEN: {symbol_active_trades[sym_to_process]['type']} @{symbol_active_trades[sym_to_process]['entry_price']:.{props['digits']}f}, SL:{symbol_active_trades[sym_to_process]['sl']:.{props['digits']}f}, TP:{symbol_active_trades[sym_to_process]['tp']:.{props['digits']}f}, R-dist: {risk_val_diff:.{props['digits']}f}")
                                symbol_pending_orders[sym_to_process] = None
                
                # --- Check for New Setups for the current symbol ---
                if not symbol_active_trades[sym_to_process] and not symbol_pending_orders[sym_to_process]: 
                    pip_adj_setup = 3 * props['trade_tick_size'] 
                    h1_trend_bias_setup = None
                    m5_setup_bias_setup = None 
                    
                    if not is_within_session(timestamp, TRADING_SESSIONS_UTC.get(sym_to_process,[])):
                        continue # Next symbol for this timestamp

                    h1_ema8 = current_candle_data['H1_EMA8']; h1_ema21 = current_candle_data['H1_EMA21']
                    h1_close = current_candle_data['H1_Close_For_Bias']
                    if pd.isna(h1_ema8) or pd.isna(h1_ema21) or pd.isna(h1_close): continue 

                    if h1_ema8>h1_ema21 and h1_close>h1_ema8 and h1_close>h1_ema21: h1_trend_bias_setup="BUY"
                    elif h1_ema8<h1_ema21 and h1_close<h1_ema8 and h1_close<h1_ema21: h1_trend_bias_setup="SELL"
                    if h1_trend_bias_setup is None: continue

                    m5_ema8 = current_candle_data['M5_EMA8']; m5_ema13 = current_candle_data['M5_EMA13']; m5_ema21_val = current_candle_data['M5_EMA21']
                    if pd.isna(m5_ema8) or pd.isna(m5_ema13) or pd.isna(m5_ema21_val): continue 

                    m5_fanned_buy = m5_ema8>m5_ema13 and m5_ema13>m5_ema21_val
                    m5_fanned_sell = m5_ema8<m5_ema13 and m5_ema13<m5_ema21_val
                    is_fanned_for_bias = (h1_trend_bias_setup=="BUY" and m5_fanned_buy) or (h1_trend_bias_setup=="SELL" and m5_fanned_sell)
                    if not is_fanned_for_bias: continue
                    
                    m5_setup_bias_setup = h1_trend_bias_setup 

                    h4_ema8 = current_candle_data.get('H4_EMA8', np.nan) 
                    h4_ema21 = current_candle_data.get('H4_EMA21', np.nan)
                    if pd.isna(h4_ema8) or pd.isna(h4_ema21): continue 
                    
                    if m5_setup_bias_setup == "BUY" and h4_ema8 < h4_ema21: continue
                    if m5_setup_bias_setup == "SELL" and h4_ema8 > h4_ema21: continue

                    if (m5_setup_bias_setup=="BUY" and current_candle_data['close'] < m5_ema21_val) or \
                       (m5_setup_bias_setup=="SELL" and current_candle_data['close'] > m5_ema21_val):
                        continue 

                    pullback = (m5_setup_bias_setup=="BUY" and current_candle_data['low']<=m5_ema8) or \
                               (m5_setup_bias_setup=="SELL" and current_candle_data['high']>=m5_ema8)
                    if not pullback: continue
                    
                    atr_val = current_candle_data.get('ATR', np.nan)
                    if pd.isna(atr_val) or atr_val <= 0: continue
                    
                    sl_distance_atr = 1.5 * atr_val 

                    symbol_df_for_lookback_full = full_prepared_data_map[sym_to_process]
                    try:
                        current_idx_in_symbol_df_for_lookback = symbol_df_for_lookback_full.index.get_loc(timestamp)
                    except KeyError: continue 

                    if current_idx_in_symbol_df_for_lookback < 4: continue
                    
                    lookback_df_for_entry = symbol_df_for_lookback_full.iloc[current_idx_in_symbol_df_for_lookback-4 : current_idx_in_symbol_df_for_lookback+1]
                    
                    entry_px, sl_px, order_type_setup = (0,0,"")

                    if m5_setup_bias_setup=="BUY":
                        entry_px = lookback_df_for_entry['high'].max() + pip_adj_setup 
                        sl_px = entry_px - sl_distance_atr                          
                        order_type_setup="BUY_STOP"
                    else: # SELL
                        entry_px = lookback_df_for_entry['low'].min() - pip_adj_setup 
                        sl_px = entry_px + sl_distance_atr                         
                        order_type_setup="SELL_STOP"
                    
                    entry_px=round(entry_px, props['digits'])
                    sl_px=round(sl_px, props['digits'])
                    sl_diff = abs(entry_px - sl_px) 
                    if sl_diff <= 0: continue
                    
                    calc_lot = calculate_lot_size(current_symbol_balance, RISK_PER_TRADE_PERCENT, sl_diff, props)
                    if calc_lot <= 0: continue
                    
                    symbol_pending_orders[sym_to_process] = {"symbol":sym_to_process, "type":order_type_setup, 
                                            "entry_price":entry_px, "sl_price":sl_px, 
                                            "created_time":timestamp, "lot_size":calc_lot,
                                            "setup_bias": m5_setup_bias_setup} 
                    
                    logger.info(f"[{sym_to_process}] {timestamp} PENDING {order_type_setup} Order Set: Entry {entry_px:.{props['digits']}f}, SL {sl_px:.{props['digits']}f} (ATR: {atr_val:.{props['digits']}f}), Lot {calc_lot}, Bal: {current_symbol_balance:.2f}")

        # --- End of Main Simulation Loop ---

        # --- Reporting Phase (After Master Loop) ---
        logger.info("\n\n===== Symbol Backtest Performance Summaries =====")
        for sym_report in SYMBOLS_AVAILABLE_FOR_TRADE:
            if sym_report not in full_prepared_data_map: # Symbol was skipped entirely due to no data
                logger.info(f"\n--- Performance Summary for Symbol: {sym_report} ---")
                logger.info(f"  Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
                logger.info(f"  Starting Balance: {INITIAL_ACCOUNT_BALANCE_PER_SYMBOL:.2f} USD")
                logger.info(f"  Ending Balance: {INITIAL_ACCOUNT_BALANCE_PER_SYMBOL:.2f} USD")
                logger.info(f"  No data available for simulation for {sym_report}.")
                continue

            trades_list_for_sym = symbol_closed_trades_map[sym_report]
            logger.info(f"\n--- Performance Summary for Symbol: {sym_report} ---")
            logger.info(f"  Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
            
            if not trades_list_for_sym: 
                logger.info(f"  Starting Balance: {INITIAL_ACCOUNT_BALANCE_PER_SYMBOL:.2f} USD")
                logger.info(f"  Ending Balance (running symbol balance): {symbol_account_balances[sym_report]:.2f} USD")
                logger.info(f"  No trades executed for {sym_report} during the backtest period.")
            else:
                symbol_stats = calculate_performance_stats(trades_list_for_sym, INITIAL_ACCOUNT_BALANCE_PER_SYMBOL)
                logger.info(f"  Starting Balance: {symbol_stats['start_balance']:.2f} USD")
                logger.info(f"  Ending Balance (from trades PnL): {symbol_stats['end_balance']:.2f} USD")
                logger.info(f"  Ending Balance (running symbol balance): {symbol_account_balances[sym_report]:.2f} USD") # Final actual balance
                logger.info(f"  Total Trades: {symbol_stats['total_trades']}")
                logger.info(f"  Winning Trades: {symbol_stats['winning_trades']}")
                logger.info(f"  Losing Trades: {symbol_stats['losing_trades']}")
                logger.info(f"  Win Rate: {symbol_stats['win_rate']:.2f}%")
                logger.info(f"  Net Profit: {symbol_stats['net_profit']:.2f} USD")
                logger.info(f"  Profit Factor: {symbol_stats['profit_factor']:.2f}")
                logger.info(f"  Max Drawdown: {symbol_stats['max_drawdown_abs']:.2f} USD ({symbol_stats['max_drawdown_pct']:.2f}%)")

        logger.info("\n\n===== All Symbol Independent Parallel Backtests Complete. =====")
 
        shutdown_mt5_interface()