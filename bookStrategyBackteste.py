import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta # For EMAs
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
                       "GBPJPY", "XAUUSD", "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                       "USOIL", "CADJPY", "XAGUSD", "XPTUSD", "UKOIL",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD"]

TRADING_SESSIONS_UTC = { # (start_hour_inclusive, end_hour_exclusive)
    "EURUSD": [(7, 16)], "GBPUSD": [(7, 16)], "AUDUSD": [(0, 4), (7, 16)],
    "USDCHF": [(7, 16)], "USDCAD": [(12, 17)], "USDJPY": [(0, 4), (12, 17)],
    "EURJPY": [(0, 4), (7, 12)], "GBPJPY": [(7, 16)], "NZDUSD": [(0, 4), (7, 16)],
    "EURCHF": [(7, 12)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 17)],
    "EURNZD": [(0, 4), (7, 12)], "GBPNZD": [(7, 12)], "XAUUSD": [(7, 16)],
    "XAGUSD": [(7, 16)], "XPTUSD": [(7, 16)]
}
TRADING_SESSIONS_UTC["USOIL"] = [(12, 17)]
TRADING_SESSIONS_UTC["UKOIL"] = [(7, 16)]
CRYPTO_SESSIONS_USER = {"BTCUSD":[(0, 16)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 16)], "ETHUSD":[(0, 16)]}
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

    # Ensure trades_list is sorted by exit_time or entry_time if pnl is calculated sequentially
    # For drawdown, the order of P&L matters.
    # If trades_list comes from all_trades_for_overall_analysis, it's already sorted by entry_time.
    # For per-symbol, if its trades are interleaved, sorting by entry_time is also correct for its isolated view.
    
    # Create a DataFrame from the trades list
    trades_df = pd.DataFrame(trades_list)
    if not trades_df.empty and 'entry_time' in trades_df.columns:
        trades_df.sort_values(by='entry_time', inplace=True) # Ensure chronological order for equity curve

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
        # Ensure mdd_end_index is valid for rolling_max_equity
        if mdd_end_index < len(rolling_max_equity):
            peak_at_mdd_start = rolling_max_equity.iloc[mdd_end_index]
            stats["max_drawdown_pct"] = (stats["max_drawdown_abs"] / peak_at_mdd_start) * 100 if peak_at_mdd_start > 0 else 0
        else: # Should not happen if logic is correct
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
    if df_h1.empty: return pd.DataFrame()
    df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
    df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
    df_h1_resampled = df_h1[['close', 'H1_EMA8', 'H1_EMA21']].rename(columns={'close': 'H1_Close_For_Bias'})

    df_m5 = get_historical_data(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    if df_m5.empty: return pd.DataFrame()
    df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
    df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
    df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)

    combined_df = pd.merge_asof(df_m5.sort_index(), df_h1_resampled.sort_index(),
                                left_index=True, right_index=True,
                                direction='backward', tolerance=pd.Timedelta(hours=1))
    combined_df.dropna(inplace=True)
    return combined_df


# --- Main Execution ---
if __name__ == "__main__":
    start_datetime = datetime(2024, 7, 1) 
    end_datetime = datetime(2025, 5, 31)
    h1_ema_buffer_days = 5 
    data_fetch_start_date = start_datetime - timedelta(days=h1_ema_buffer_days * 2)

    if not initialize_mt5_interface(SYMBOLS_TO_BACKTEST):
        logger.error("Failed to initialize MT5 or critical symbols. Exiting.")
    else:
        # --- Global State Variables for Portfolio Simulation ---
        shared_account_balance = INITIAL_ACCOUNT_BALANCE
        global_active_trade = None
        global_pending_order = None
        all_closed_trades_portfolio = [] # Stores all trades from all symbols, chronologically
        
        # For per-symbol conceptual start balances for individual reporting
        symbol_conceptual_start_balances = {} 
        # To store trades per symbol for individual reporting
        trades_per_symbol_map = {sym: [] for sym in SYMBOLS_AVAILABLE_FOR_TRADE}


        # Daily risk tracking
        current_simulation_date = None
        daily_risk_allocated_on_current_date = 0.0
        max_daily_risk_budget_for_current_date = 0.0

        logger.info(f"Global Initial Account Balance: {shared_account_balance:.2f} USD")
        logger.info(f"Backtesting Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
        logger.info(f"Risk per trade: {RISK_PER_TRADE_PERCENT*100:.2f}%, Daily Risk Limit: {DAILY_RISK_LIMIT_PERCENT*100:.2f}%")
        logger.info("One trade at a time across the entire portfolio.")

        # 1. Prepare data for all symbols first
        prepared_symbol_data = {}
        master_time_index_set = set()
        for sym in SYMBOLS_AVAILABLE_FOR_TRADE:
            props = ALL_SYMBOL_PROPERTIES[sym]
            df = prepare_symbol_data(sym, data_fetch_start_date, end_datetime, props)
            if not df.empty:
                # Filter data to be within the actual backtest period for the master index
                df_filtered_for_index = df[(df.index >= pd.Timestamp(start_datetime, tz='UTC')) & (df.index <= pd.Timestamp(end_datetime, tz='UTC'))]
                prepared_symbol_data[sym] = df # Store full df for lookups
                master_time_index_set.update(df_filtered_for_index.index)
            else:
                logger.warning(f"No data prepared for {sym}, it will be skipped in simulation.")
        
        if not master_time_index_set:
            logger.error("No data available for any symbol in the specified range. Exiting.")
            shutdown_mt5_interface()
            exit()

        master_time_index = sorted(list(master_time_index_set))
        logger.info(f"Master time index created with {len(master_time_index)} M5 candles to process.")

        # 2. Main Simulation Loop (Chronological M5 candles across all symbols)
        for timestamp in master_time_index:
            candle_date = timestamp.date()

            # --- Daily Risk Management Reset ---
            if candle_date != current_simulation_date:
                current_simulation_date = candle_date
                daily_risk_allocated_on_current_date = 0.0
                max_daily_risk_budget_for_current_date = shared_account_balance * DAILY_RISK_LIMIT_PERCENT
                logger.debug(f"Portfolio New Day: {current_simulation_date}. Max Daily Risk: {max_daily_risk_budget_for_current_date:.2f} (Bal: {shared_account_balance:.2f}). Daily allocated risk reset.")

            # --- Manage Global Active Trade ---
            if global_active_trade:
                trade_symbol = global_active_trade['symbol']
                props = ALL_SYMBOL_PROPERTIES[trade_symbol]
                pip_adj = 3 * props['pip_value_calc']

                if trade_symbol in prepared_symbol_data and timestamp in prepared_symbol_data[trade_symbol].index:
                    current_candle_for_active_trade = prepared_symbol_data[trade_symbol].loc[timestamp]
                    closed_this_bar = False
                    exit_price = 0

                    # SL Check
                    if global_active_trade['type'] == "BUY" and current_candle_for_active_trade['low'] <= global_active_trade['sl']:
                        exit_price = global_active_trade['sl']; global_active_trade['status'] = "SL_HIT"
                    elif global_active_trade['type'] == "SELL" and current_candle_for_active_trade['high'] >= global_active_trade['sl']:
                        exit_price = global_active_trade['sl']; global_active_trade['status'] = "SL_HIT"
                    # TP Check
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
                        pnl_ticks = price_diff / props['trade_tick_size']
                        global_active_trade['pnl_currency'] = pnl_ticks * props['trade_tick_value'] * global_active_trade['lot_size']
                        shared_account_balance += global_active_trade['pnl_currency']
                        global_active_trade['balance_after_trade'] = shared_account_balance
                        
                        logger.info(f"[{trade_symbol}] {timestamp} Trade CLOSED ({global_active_trade['status']}): P&L: {global_active_trade['pnl_currency']:.2f}, New Portfolio Bal: {shared_account_balance:.2f}")
                        all_closed_trades_portfolio.append(global_active_trade.copy())
                        trades_per_symbol_map[trade_symbol].append(global_active_trade.copy())
                        global_active_trade = None
                    
                    # Trailing SL for active trade (if not closed)
                    if global_active_trade and not closed_this_bar:
                        if not global_active_trade['trailing_active']:
                            if (global_active_trade['type']=="BUY" and current_candle_for_active_trade['high']>=global_active_trade['ts_activation_price']) or \
                               (global_active_trade['type']=="SELL" and current_candle_for_active_trade['low']<=global_active_trade['ts_activation_price']):
                                global_active_trade['trailing_active']=True; logger.info(f"[{trade_symbol}] {timestamp} Trailing Stop ACTIVATED")
                        
                        if global_active_trade['trailing_active']:
                            # Need to find the index of 'timestamp' in this symbol's specific data to look back
                            symbol_df = prepared_symbol_data[trade_symbol]
                            try:
                                current_idx_in_symbol_df = symbol_df.index.get_loc(timestamp)
                                if current_idx_in_symbol_df >= 3: # Need 3 previous COMPLETED candles
                                    last_3_candles = symbol_df.iloc[current_idx_in_symbol_df-3 : current_idx_in_symbol_df] # Slices up to, not including, current
                                    if global_active_trade['type']=="BUY":
                                        new_sl = last_3_candles['low'].min() - pip_adj
                                        if new_sl > global_active_trade['sl']: global_active_trade['sl'] = new_sl
                                    else: # SELL
                                        new_sl = last_3_candles['high'].max() + pip_adj
                                        if new_sl < global_active_trade['sl']: global_active_trade['sl'] = new_sl
                            except KeyError:
                                logger.warning(f"Timestamp {timestamp} not found in {trade_symbol} df for trailing SL, skipping TSL update.")


            # --- Manage Global Pending Order (if no active trade) ---
            if not global_active_trade and global_pending_order:
                order_symbol = global_pending_order['symbol']
                props = ALL_SYMBOL_PROPERTIES[order_symbol]

                if order_symbol in prepared_symbol_data and timestamp in prepared_symbol_data[order_symbol].index:
                    current_candle_for_pending_order = prepared_symbol_data[order_symbol].loc[timestamp]
                    entry_price = global_pending_order['entry_price']
                    sl_price = global_pending_order['sl_price']
                    order_type = global_pending_order['type']
                    lot_size = global_pending_order['lot_size']
                    
                    # Invalidation Check (M5 close beyond M5 21 EMA) for the pending order's setup
                    # This needs the m5_setup_bias and m5_ema21 at the time of this candle for order_symbol
                    m5_ema21_for_invalidation = current_candle_for_pending_order['M5_EMA21'] # From the current candle of the order's symbol
                    
                    setup_invalidated = False
                    if global_pending_order['setup_bias'] == "BUY" and current_candle_for_pending_order['close'] < m5_ema21_for_invalidation:
                        setup_invalidated = True
                        logger.info(f"[{order_symbol}] {timestamp} PENDING BUY order invalidated (Close < M5_EMA21 before trigger).")
                    elif global_pending_order['setup_bias'] == "SELL" and current_candle_for_pending_order['close'] > m5_ema21_for_invalidation:
                        setup_invalidated = True
                        logger.info(f"[{order_symbol}] {timestamp} PENDING SELL order invalidated (Close > M5_EMA21 before trigger).")

                    if setup_invalidated:
                        # If invalidated, the daily allocated risk should be 'refunded'
                        daily_risk_allocated_on_current_date -= global_pending_order['intended_risk_amount']
                        logger.debug(f"[{order_symbol}] Risk {global_pending_order['intended_risk_amount']:.2f} refunded due to pending order invalidation. Daily allocated: {daily_risk_allocated_on_current_date:.2f}")
                        global_pending_order = None # Cancel it
                    else:
                        # Trigger Check
                        triggered = False; actual_entry_price = 0
                        if order_type == "BUY_STOP" and current_candle_for_pending_order['high'] >= entry_price:
                            actual_entry_price = entry_price; triggered = True
                        elif order_type == "SELL_STOP" and current_candle_for_pending_order['low'] <= entry_price:
                            actual_entry_price = entry_price; triggered = True
                        
                        if triggered:
                            logger.info(f"[{order_symbol}] {timestamp} PENDING {order_type} TRIGGERED @{actual_entry_price:.{props['digits']}f} Lot:{lot_size}")
                            risk_val_diff = abs(actual_entry_price - sl_price) # R-value in price
                            if risk_val_diff <= 0 or lot_size <= 0:
                                logger.warning(f"[{order_symbol}] Invalid risk/lot on trigger. Cancelling order and refunding risk.")
                                daily_risk_allocated_on_current_date -= global_pending_order['intended_risk_amount'] # Refund
                                global_pending_order = None
                            else:
                                tp_price = actual_entry_price + 3*risk_val_diff if order_type=="BUY_STOP" else actual_entry_price - 3*risk_val_diff
                                global_active_trade = {"symbol":order_symbol,"type":"BUY" if order_type=="BUY_STOP" else "SELL",
                                                      "entry_time":timestamp,"entry_price":actual_entry_price,
                                                      "sl":sl_price,"initial_sl":sl_price,"tp":tp_price,
                                                      "r_value_price_diff":risk_val_diff,"trailing_active":False,
                                                      "ts_activation_price": actual_entry_price + 1.5*risk_val_diff if order_type=="BUY_STOP" else actual_entry_price-1.5*risk_val_diff,
                                                      "status":"OPEN","lot_size":lot_size,"pnl_currency":0.0}
                                logger.info(f"  [{order_symbol}] Trade OPEN: {global_active_trade['type']} @{global_active_trade['entry_price']:.{props['digits']}f}, SL:{global_active_trade['sl']:.{props['digits']}f}, TP:{global_active_trade['tp']:.{props['digits']}f}")
                                # Record conceptual start balance for this symbol if it's its first trade
                                if order_symbol not in symbol_conceptual_start_balances:
                                    symbol_conceptual_start_balances[order_symbol] = shared_account_balance # Balance before this trade's P&L
                                global_pending_order = None


            # --- Look for New Trade Setup (if no global active trade AND no global pending order) ---
            if not global_active_trade and not global_pending_order:
                # This part iterates through symbols to find ONE setup if allowed
                # State variables for setup search, reset for each symbol at each timestamp
                # These are temporary for the inner loop for setup finding
                
                # We must iterate SYMBOLS_AVAILABLE_FOR_TRADE in a defined order to ensure determinism
                # if multiple setups occur at the same time.
                for sym_to_check_setup in SYMBOLS_AVAILABLE_FOR_TRADE:
                    if sym_to_check_setup not in prepared_symbol_data or timestamp not in prepared_symbol_data[sym_to_check_setup].index:
                        continue # Skip if no data for this symbol at this timestamp

                    current_candle_for_setup = prepared_symbol_data[sym_to_check_setup].loc[timestamp]
                    props_setup = ALL_SYMBOL_PROPERTIES[sym_to_check_setup]
                    pip_adj_setup = 3 * props_setup['pip_value_calc']

                    # Symbol-specific setup state (reset per symbol check per timestamp)
                    h1_trend_bias_setup = None
                    m5_setup_active_setup = False
                    m5_setup_bias_setup = None
                    trigger_bar_candle_data_setup = None
                    # We use the H1 data associated with `current_candle_for_setup`
                    
                    if not is_within_session(timestamp, TRADING_SESSIONS_UTC.get(sym_to_check_setup,[])):
                        continue

                    # Step 1: H1 Trend
                    h1_ema8 = current_candle_for_setup['H1_EMA8']; h1_ema21 = current_candle_for_setup['H1_EMA21']
                    h1_close = current_candle_for_setup['H1_Close_For_Bias']
                    if pd.isna(h1_ema8) or pd.isna(h1_ema21) or pd.isna(h1_close): continue # Skip if H1 data missing

                    if h1_ema8>h1_ema21 and h1_close>h1_ema8 and h1_close>h1_ema21: h1_trend_bias_setup="BUY"
                    elif h1_ema8<h1_ema21 and h1_close<h1_ema8 and h1_close<h1_ema21: h1_trend_bias_setup="SELL"
                    if h1_trend_bias_setup is None: continue

                    # Step 2: M5 Fanning
                    m5_ema8 = current_candle_for_setup['M5_EMA8']; m5_ema13 = current_candle_for_setup['M5_EMA13']; m5_ema21_val = current_candle_for_setup['M5_EMA21']
                    if pd.isna(m5_ema8) or pd.isna(m5_ema13) or pd.isna(m5_ema21_val): continue

                    m5_fanned_buy = m5_ema8>m5_ema13 and m5_ema13>m5_ema21_val
                    m5_fanned_sell = m5_ema8<m5_ema13 and m5_ema13<m5_ema21_val
                    is_fanned_for_bias = (h1_trend_bias_setup=="BUY" and m5_fanned_buy) or (h1_trend_bias_setup=="SELL" and m5_fanned_sell)
                    if not is_fanned_for_bias: continue
                    
                    m5_setup_active_setup = True; m5_setup_bias_setup = h1_trend_bias_setup
                    # logger.debug(f"[{sym_to_check_setup}] {timestamp} M5 Setup Potentially Active for {m5_setup_bias_setup}")

                    # Step 5: Invalidation (during pullback search on current candle)
                    if (m5_setup_bias_setup=="BUY" and current_candle_for_setup['close'] < m5_ema21_val) or \
                       (m5_setup_bias_setup=="SELL" and current_candle_for_setup['close'] > m5_ema21_val):
                        # logger.debug(f"[{sym_to_check_setup}] {timestamp} M5 Setup Invalidated (Close vs M5_EMA21).")
                        continue # Invalidated for this symbol, try next symbol or next timestamp

                    # Pullback & Trigger Bar
                    pullback = (m5_setup_bias_setup=="BUY" and current_candle_for_setup['low']<=m5_ema8) or \
                               (m5_setup_bias_setup=="SELL" and current_candle_for_setup['high']>=m5_ema8)
                    if not pullback: continue
                    
                    trigger_bar_candle_data_setup = current_candle_for_setup.copy()
                    # logger.debug(f"[{sym_to_check_setup}] {timestamp} TRIGGER BAR for {m5_setup_bias_setup}")
                    
                    # Index of current candle within this symbol's specific dataframe
                    symbol_df_for_lookback = prepared_symbol_data[sym_to_check_setup]
                    try:
                        current_idx_in_symbol_df_for_lookback = symbol_df_for_lookback.index.get_loc(timestamp)
                    except KeyError: continue # Should not happen if we are here

                    if current_idx_in_symbol_df_for_lookback < 4: 
                        # logger.warning(f"[{sym_to_check_setup}] Not enough lookback for entry. Skip.")
                        continue
                    
                    lookback_df = symbol_df_for_lookback.iloc[current_idx_in_symbol_df_for_lookback-4 : current_idx_in_symbol_df_for_lookback+1]
                    entry_px, sl_px, order_type_setup = (0,0,"")

                    if m5_setup_bias_setup=="BUY":
                        entry_px=lookback_df['high'].max()+pip_adj_setup
                        sl_px=trigger_bar_candle_data_setup['low']-pip_adj_setup
                        order_type_setup="BUY_STOP"
                    else: # SELL
                        entry_px=lookback_df['low'].min()-pip_adj_setup
                        sl_px=trigger_bar_candle_data_setup['high']+pip_adj_setup
                        order_type_setup="SELL_STOP"
                    
                    entry_px=round(entry_px,props_setup['digits']); sl_px=round(sl_px,props_setup['digits'])
                    sl_diff=abs(entry_px-sl_px)

                    if (m5_setup_bias_setup=="BUY" and sl_px>=entry_px) or \
                       (m5_setup_bias_setup=="SELL" and sl_px<=entry_px) or sl_diff==0:
                        # logger.warning(f"[{sym_to_check_setup}] Illogical SL/Entry for setup. Skip.")
                        continue
                    
                    # Daily Risk Check for Portfolio
                    intended_risk_amount_this_trade = shared_account_balance * RISK_PER_TRADE_PERCENT
                    if daily_risk_allocated_on_current_date + intended_risk_amount_this_trade > max_daily_risk_budget_for_current_date + 1e-9:
                        logger.info(f"[{sym_to_check_setup}] {timestamp} Portfolio Daily Risk Limit Reached for {current_simulation_date}. "
                                    f"Max: {max_daily_risk_budget_for_current_date:.2f}, "
                                    f"Allocated: {daily_risk_allocated_on_current_date:.2f}, "
                                    f"Intended for this trade: {intended_risk_amount_this_trade:.2f}. Skipping trade.")
                        continue # Cannot take this trade due to portfolio daily limit

                    calc_lot = calculate_lot_size(shared_account_balance, RISK_PER_TRADE_PERCENT, sl_diff, props_setup)
                    if calc_lot <= 0:
                        # logger.warning(f"[{sym_to_check_setup}] Lot size 0 for setup. Skip. Bal:{shared_account_balance:.2f}")
                        continue
                    
                    # If we reach here, a valid setup is found for sym_to_check_setup, risk is okay.
                    # Set global pending order and break from checking other symbols for this timestamp.
                    global_pending_order = {"symbol":sym_to_check_setup, "type":order_type_setup, 
                                            "entry_price":entry_px, "sl_price":sl_px, 
                                            "created_time":timestamp, "lot_size":calc_lot,
                                            "setup_bias": m5_setup_bias_setup, # Store for invalidation check
                                            "intended_risk_amount": intended_risk_amount_this_trade} # Store for potential refund
                    
                    daily_risk_allocated_on_current_date += intended_risk_amount_this_trade
                    logger.info(f"[{sym_to_check_setup}] {timestamp} GLOBAL PENDING {order_type_setup} Order Set: Entry {entry_px:.{props_setup['digits']}f}, SL {sl_px:.{props_setup['digits']}f}, Lot {calc_lot}")
                    logger.debug(f"  Portfolio daily risk allocated: {daily_risk_allocated_on_current_date:.2f}/{max_daily_risk_budget_for_current_date:.2f}")
                    break # IMPORTANT: Only one pending order across portfolio at a time
        
        # --- END OF MAIN SIMULATION LOOP ---

        logger.info("\n\n===== All Symbol Simulations Complete. Generating Summaries. =====")

        # --- Individual Symbol Summaries ---
        for symbol_iter, symbol_trades_list in trades_per_symbol_map.items():
            logger.info(f"\n--- Performance Summary for Symbol: {symbol_iter} ---")
            logger.info(f"  Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
            
            # Use the conceptual start balance recorded when the symbol's first trade was initiated
            # If no trades, it means it never got a chance or had no setups.
            conceptual_start_bal = symbol_conceptual_start_balances.get(symbol_iter, shared_account_balance if not all_closed_trades_portfolio else INITIAL_ACCOUNT_BALANCE)
            # If a symbol never traded, its "start balance" for its summary is less defined.
            # Default to INITIAL_ACCOUNT_BALANCE if it never traded to show it started from that pool.
            # Or, if it traded, use its specific recorded start balance.

            if not symbol_trades_list: # If symbol had no trades
                 # If no trades for this symbol, its "conceptual start balance" for ITS summary could be the global initial if no trades AT ALL happened yet,
                 # or if other trades happened, it's harder to define.
                 # Let's use the value from symbol_conceptual_start_balances if it exists (meaning it was about to trade but didn't complete one)
                 # or the initial global balance if it never even got to place a pending order.
                no_trade_start_bal = symbol_conceptual_start_balances.get(symbol_iter, INITIAL_ACCOUNT_BALANCE)
                logger.info(f"  Starting Balance (conceptual for this symbol's activity): {no_trade_start_bal:.2f} USD")
                logger.info(f"  Ending Balance (conceptual): {no_trade_start_bal:.2f} USD")
                logger.info(f"  No trades executed for {symbol_iter} during the backtest period.")
            else:
                # Sort this symbol's trades by entry time for its isolated stats calculation
                symbol_trades_list.sort(key=lambda x: x['entry_time'])
                symbol_stats = calculate_performance_stats(symbol_trades_list, conceptual_start_bal)
                logger.info(f"  Starting Balance (conceptual, when first trade for this symbol was initiated): {symbol_stats['start_balance']:.2f} USD")
                logger.info(f"  Ending Balance (conceptual, start_bal + PnL for this symbol): {symbol_stats['end_balance']:.2f} USD")
                logger.info(f"  Total Trades: {symbol_stats['total_trades']}")
                logger.info(f"  Winning Trades: {symbol_stats['winning_trades']}")
                logger.info(f"  Losing Trades: {symbol_stats['losing_trades']}")
                logger.info(f"  Win Rate: {symbol_stats['win_rate']:.2f}%")
                logger.info(f"  Net Profit (for this symbol): {symbol_stats['net_profit']:.2f} USD")
                logger.info(f"  Profit Factor (for this symbol): {symbol_stats['profit_factor']:.2f}")
                logger.info(f"  Max Drawdown (based on this symbol's trades in isolation): {symbol_stats['max_drawdown_abs']:.2f} USD ({symbol_stats['max_drawdown_pct']:.2f}%)")

        # --- Overall Performance Summary ---
        logger.info("\n\n===== Overall Backtest Performance Summary =====")
        if all_closed_trades_portfolio:
            # all_closed_trades_portfolio is already naturally chronological by M5 candle processing.
            # Re-sorting by entry_time ensures strictness if any sub-M5 event order was important (not the case here but good practice).
            all_closed_trades_portfolio.sort(key=lambda x: x['entry_time']) 
            overall_stats = calculate_performance_stats(all_closed_trades_portfolio, INITIAL_ACCOUNT_BALANCE)
            logger.info(f"Tested Symbols: {SYMBOLS_AVAILABLE_FOR_TRADE}")
            logger.info(f"Overall Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")
            logger.info(f"Overall Starting Balance: {overall_stats['start_balance']:.2f} USD")
            logger.info(f"Overall Ending Balance (from sorted trades): {overall_stats['end_balance']:.2f} USD")
            logger.info(f"Overall Ending Balance (final shared_account_balance variable): {shared_account_balance:.2f} USD") 
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