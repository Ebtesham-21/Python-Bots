import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta # For EMAs
import numpy as np
import time
from datetime import datetime, timedelta, date, timezone # Added timezone
import logging
import math

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}
RUN_BACKTEST = False # <<<< IMPORTANT FOR LIVE TRADING
BOT_MAGIC_NUMBER = 12345 # Unique identifier for this bot's trades

# --- Strategy & Trading Parameters ---
SYMBOLS_TO_TRADE = ["EURUSD", "AUDUSD", "USDCHF", "USDCAD",
                    "GBPJPY", "XAUUSD", "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                    "USOIL", "CADJPY", "XAGUSD", "XPTUSD", "UKOIL",
                    "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD"] # Symbols to consider

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

RISK_PER_TRADE_PERCENT = 0.01 # Risk 1% of current balance per trade
DAILY_RISK_LIMIT_PERCENT = 0.05 # Daily risk limit of 5% of balance at start of day
MAX_SPREAD_PIPS_ALLOWED = 5.0 # Max spread in pips to consider a trade

H1_CANDLES_TO_FETCH = 50 # For EMA calculation
M5_CANDLES_TO_FETCH = 50 # For EMA calculation
LOOP_SLEEP_SECONDS = 30 # How often to check for new signals (e.g., 30-60 seconds)

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
            time.sleep(0.5); symbol_info_obj = mt5.symbol_info(symbol_name) # Re-fetch after select
            if symbol_info_obj is None or not symbol_info_obj.visible: logger.warning(f"Symbol {symbol_name} still not available. Skipping."); continue

        if symbol_info_obj.point == 0: logger.warning(f"Symbol {symbol_name} point value is 0. Skipping."); continue
        if symbol_info_obj.trade_tick_size == 0: logger.warning(f"Symbol {symbol_name} trade_tick_size is 0. Skipping."); continue

        pip_value_std = 0.0001; pip_value_jpy = 0.01
        current_pip_value = pip_value_jpy if 'JPY' in symbol_name.upper() else pip_value_std
        if symbol_name.upper() in ["XAUUSD", "XAGUSD", "XPTUSD"]: current_pip_value = 0.01
        elif "OIL" in symbol_name.upper() or "USOIL" in symbol_name.upper() or "UKOIL" in symbol_name.upper(): current_pip_value = 0.01
        elif "BTC" in symbol_name.upper() or "ETH" in symbol_name.upper(): current_pip_value = 0.01

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
def is_within_session(symbol_sessions):
    if not symbol_sessions: return True
    now_utc = datetime.now(timezone.utc) # Corrected: Use timezone-aware UTC time
    current_hour = now_utc.hour
    for start_hour, end_hour in symbol_sessions:
        if start_hour <= current_hour < end_hour: return True
    return False

def calculate_lot_size(account_balance_for_risk_calc, risk_percent, sl_price_diff, symbol_props):
    if sl_price_diff <= 0: return 0
    risk_amount_currency = account_balance_for_risk_calc * risk_percent
    if symbol_props['trade_tick_size'] == 0 or symbol_props['trade_tick_value'] == 0:
        logger.error(f"Symbol {symbol_props.get('description', '')} has zero tick_size or tick_value.")
        return 0
    
    sl_distance_ticks = sl_price_diff / symbol_props['trade_tick_size']
    sl_cost_per_lot = sl_distance_ticks * symbol_props['trade_tick_value']

    if sl_cost_per_lot <= 1e-9:
        logger.warning(f"Calculated SL cost per lot is too low or zero for {symbol_props.get('description', '')}: {sl_cost_per_lot}. SL diff: {sl_price_diff}")
        return 0
        
    lot_size = risk_amount_currency / sl_cost_per_lot
    lot_size = max(symbol_props['volume_min'], lot_size)
    lot_size = math.floor(lot_size / symbol_props['volume_step']) * symbol_props['volume_step']
    lot_size = min(symbol_props['volume_max'], lot_size)

    if lot_size < symbol_props['volume_min']:
        if symbol_props['volume_min'] * sl_cost_per_lot > risk_amount_currency * 1.5:
            logger.warning(f"Min lot size for {symbol_props.get('description','')} exceeds risk. Lot: {lot_size}, Min Vol: {symbol_props['volume_min']}")
            return 0
        return symbol_props['volume_min']
    
    precision = int(-math.log10(symbol_props['volume_step'])) if symbol_props['volume_step'] > 0 else 2
    return round(lot_size, precision)


def get_current_spread_pips(symbol, symbol_props):
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info:
        spread_points = tick_info.ask - tick_info.bid
        spread_pips = spread_points / symbol_props['pip_value_calc'] if symbol_props['pip_value_calc'] > 0 else float('inf')
        return spread_pips
    return float('inf')

# --- Data Fetching for Live Trading ---
def get_live_data_with_emas(symbol, h1_candles_req, m5_candles_req):
    latest_h1_data_series = None
    latest_m5_data_series = None
    m5_lookback_df_out = None

    # H1 Data
    if h1_candles_req > 0:
        rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, h1_candles_req)
        if rates_h1 is not None and len(rates_h1) > 0:
            df_h1 = pd.DataFrame(rates_h1)
            df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s', utc=True)
            # Calculate EMAs if enough data, otherwise they will be NaN or missing in the series
            if len(df_h1) >= 8:  df_h1['H1_EMA8'] = ta.ema(df_h1['close'], length=8)
            if len(df_h1) >= 21: df_h1['H1_EMA21'] = ta.ema(df_h1['close'], length=21)
            latest_h1_data_series = df_h1.iloc[-1].copy() # Use .copy()
        else:
            logger.warning(f"H1 data for {symbol}: requested {h1_candles_req}, "
                           f"got {len(rates_h1) if rates_h1 is not None else 0}. 'latest_h1_data_series' will be None or incomplete.")
    else:
        logger.debug(f"H1 data not requested for {symbol} (h1_candles_req={h1_candles_req}).")

    # M5 Data (Strategy critically depends on M5 EMAs up to 21)
    if m5_candles_req > 0:
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, m5_candles_req)
        # Check for enough data for M5 EMAs. If not, M5 data will be None.
        if rates_m5 is not None and len(rates_m5) >= 21: 
            df_m5 = pd.DataFrame(rates_m5)
            df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s', utc=True)
            df_m5['M5_EMA8'] = ta.ema(df_m5['close'], length=8)
            df_m5['M5_EMA13'] = ta.ema(df_m5['close'], length=13)
            df_m5['M5_EMA21'] = ta.ema(df_m5['close'], length=21)
            latest_m5_data_series = df_m5.iloc[-1].copy() # Use .copy()
            m5_lookback_df_out = df_m5.iloc[-5:].copy()   # Use .copy()
        else:
            logger.warning(f"Not enough M5 data for EMA calculation for {symbol} "
                           f"(got {len(rates_m5) if rates_m5 is not None else 0}, requested {m5_candles_req}, min 21 for EMAs needed). "
                           f"'latest_m5_data_series' and 'm5_lookback_df_out' will be None.")
            # latest_m5_data_series and m5_lookback_df_out remain None
    else:
         logger.debug(f"M5 data not requested for {symbol} (m5_candles_req={m5_candles_req}).")

    return latest_h1_data_series, latest_m5_data_series, m5_lookback_df_out


# --- Trade Execution and Management ---
def place_pending_order(symbol, order_type, price, sl, lot_size, comment=""):
    props = ALL_SYMBOL_PROPERTIES[symbol]
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": round(price, props['digits']),
        "sl": round(sl, props['digits']),
        "magic": BOT_MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    
    if order_type == mt5.ORDER_TYPE_BUY_STOP:
        tp = round(price + 3 * abs(price - sl), props['digits'])
    elif order_type == mt5.ORDER_TYPE_SELL_STOP:
        tp = round(price - 3 * abs(price - sl), props['digits'])
    else:
        logger.error(f"Unsupported order type for TP calculation: {order_type}")
        return None
    request["tp"] = tp

    logger.info(f"Attempting to place PENDING order: {request}")
    result = mt5.order_send(request)
    if result is None:
        logger.error(f"order_send failed, error code = {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE and result.retcode != mt5.TRADE_RETCODE_PLACED:
        logger.error(f"order_send failed, retcode={result.retcode}, comment: {result.comment}")
        logger.error(f"Request was: {result.request}") # Log the request MT5 made
        return None
    
    logger.info(f"Pending order PLACED successfully: Ticket {result.order}, Symbol {symbol}, Type {order_type}, Price {price}, SL {sl}, TP {tp}, Lot {lot_size}")
    return result.order

def cancel_pending_order(ticket_id):
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": ticket_id,
        "magic": BOT_MAGIC_NUMBER
    }
    logger.info(f"Attempting to CANCEL pending order ticket: {ticket_id}")
    result = mt5.order_send(request)
    if result is None:
        logger.error(f"cancel_pending_order (ticket {ticket_id}) failed, error code = {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"cancel_pending_order (ticket {ticket_id}) failed, retcode={result.retcode}, comment: {result.comment}")
        return False
    logger.info(f"Pending order ticket {ticket_id} CANCELED successfully.")
    return True

def manage_trailing_sl(position, symbol_props, m5_lookback_df_for_tsl):
    if not position or position.magic != BOT_MAGIC_NUMBER:
        return

    pip_adj = 3 * symbol_props['pip_value_calc']
    ts_activation_price_calc = 0
    if position.type == mt5.ORDER_TYPE_BUY:
        ts_activation_price_calc = position.price_open + 1.5 * abs(position.price_open - position.sl)
    elif position.type == mt5.ORDER_TYPE_SELL:
        ts_activation_price_calc = position.price_open - 1.5 * abs(position.price_open - position.sl)

    current_tick = mt5.symbol_info_tick(position.symbol)
    if not current_tick:
        logger.warning(f"Could not get tick for {position.symbol} to manage TSL.")
        return

    trailing_active = False
    if position.type == mt5.ORDER_TYPE_BUY and current_tick.ask >= ts_activation_price_calc:
        trailing_active = True
    elif position.type == mt5.ORDER_TYPE_SELL and current_tick.bid <= ts_activation_price_calc:
        trailing_active = True
    
    if trailing_active:
        logger.info(f"TSL activated for position {position.ticket} on {position.symbol}.")
        new_sl = 0
        if len(m5_lookback_df_for_tsl) >= 4: # Need at least 4 rows to get last 3 completed candles ([-4:-1])
            last_3_closed_candles = m5_lookback_df_for_tsl.iloc[-4:-1] 
            if len(last_3_closed_candles) < 3:
                 logger.warning(f"Not enough candles in m5_lookback_df_for_tsl for TSL ({len(last_3_closed_candles)} from {-4--1}), skipping TSL for {position.ticket}")
                 return

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl_candidate = last_3_closed_candles['low'].min() - pip_adj
                if new_sl_candidate > position.sl: 
                    new_sl = round(new_sl_candidate, symbol_props['digits'])
            elif position.type == mt5.ORDER_TYPE_SELL:
                new_sl_candidate = last_3_closed_candles['high'].max() + pip_adj
                if new_sl_candidate < position.sl: 
                    new_sl = round(new_sl_candidate, symbol_props['digits'])
            
            if new_sl != 0 and new_sl != position.sl: # Check if a valid new SL was calculated and it's different
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp, 
                    "magic": BOT_MAGIC_NUMBER
                }
                logger.info(f"Attempting to TRAIL SL for position {position.ticket} to {new_sl}")
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"SL for position {position.ticket} trailed successfully to {new_sl}.")
                else:
                    err_code = mt5.last_error()
                    logger.error(f"Failed to trail SL for position {position.ticket}. Retcode: {result.retcode if result else 'None'}, MT5 Error: {err_code}, Details: {result.comment if result else 'N/A'}")
        else:
            logger.warning(f"m5_lookback_df_for_tsl for {position.symbol} has only {len(m5_lookback_df_for_tsl)} rows, need at least 4 for TSL.")


def get_todays_realized_pnl():
    # Get the start of today in UTC
    start_of_today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    from_timestamp_utc = int(start_of_today_utc.timestamp())
    to_timestamp_utc = int(datetime.now(timezone.utc).timestamp())
    
    deals = mt5.history_deals_get(from_timestamp_utc, to_timestamp_utc)
    if deals is None:
        logger.error(f"Failed to get history deals: {mt5.last_error()}")
        return 0.0

    pnl_today = 0.0
    for deal in deals:
        if deal.magic == BOT_MAGIC_NUMBER and deal.entry == mt5.DEAL_ENTRY_OUT:
            pnl_today += deal.profit
    return pnl_today


# --- Main Execution ---
if __name__ == "__main__":
    if not initialize_mt5_interface(SYMBOLS_TO_TRADE):
        logger.error("Failed to initialize MT5 or critical symbols. Exiting.")
        exit()

    balance_at_start_of_day = mt5.account_info().balance
    todays_realized_pnl = 0.0 
    
    logger.info(f"Bot starting with Magic Number: {BOT_MAGIC_NUMBER}")
    logger.info(f"Risk per trade: {RISK_PER_TRADE_PERCENT*100:.2f}%, Daily Risk Limit: {DAILY_RISK_LIMIT_PERCENT*100:.2f}%")
    logger.info("One trade/pending order at a time across the entire portfolio.")

    current_day_for_risk_reset = date.today()

    try:
        while True:
            current_utc_time = datetime.now(timezone.utc) # Corrected: Use timezone-aware UTC time
            
            if current_utc_time.date() != current_day_for_risk_reset:
                balance_at_start_of_day = mt5.account_info().balance
                todays_realized_pnl = get_todays_realized_pnl() 
                current_day_for_risk_reset = current_utc_time.date()
                logger.info(f"New Day: {current_day_for_risk_reset}. Balance at start of day: {balance_at_start_of_day:.2f}. Today's realized P&L (bot trades): {todays_realized_pnl:.2f}")

            current_account_balance = mt5.account_info().balance
            max_daily_loss_allowed = balance_at_start_of_day * DAILY_RISK_LIMIT_PERCENT
            
            open_positions = mt5.positions_get(magic=BOT_MAGIC_NUMBER)
            pending_orders = mt5.orders_get(magic=BOT_MAGIC_NUMBER)

            num_bot_positions = len(open_positions) if open_positions else 0
            num_bot_pending_orders = len(pending_orders) if pending_orders else 0
            
            if num_bot_positions > 0:
                for pos in open_positions: 
                    props = ALL_SYMBOL_PROPERTIES.get(pos.symbol)
                    if props:
                        _, _, m5_lookback_active_trade = get_live_data_with_emas(pos.symbol, 0, M5_CANDLES_TO_FETCH) # H1 not needed for TSL
                        if m5_lookback_active_trade is not None and not m5_lookback_active_trade.empty:
                            manage_trailing_sl(pos, props, m5_lookback_active_trade)
                        else:
                            logger.warning(f"Could not get M5 data for TSL on {pos.symbol}")
                time.sleep(LOOP_SLEEP_SECONDS)
                continue 

            if num_bot_pending_orders > 0:
                for order in pending_orders: 
                    props = ALL_SYMBOL_PROPERTIES.get(order.symbol)
                    if not props: continue

                    # For invalidation, only M5 data is strictly needed. H1 data is not used for this check.
                    _, latest_m5_data_order, _ = get_live_data_with_emas(order.symbol, 0, M5_CANDLES_TO_FETCH) 
                    
                    if latest_m5_data_order is None:
                        logger.warning(f"Could not get M5 data for pending order invalidation check on {order.symbol}. Skipping check for now.")
                        continue

                    # Ensure M5_EMA21 and close are available
                    if 'M5_EMA21' not in latest_m5_data_order or pd.isna(latest_m5_data_order['M5_EMA21']) or \
                       'close' not in latest_m5_data_order or pd.isna(latest_m5_data_order['close']):
                        logger.warning(f"M5_EMA21 or close price missing in M5 data for {order.symbol}. Skipping invalidation check.")
                        continue

                    m5_ema21_for_invalidation = latest_m5_data_order['M5_EMA21']
                    current_m5_close = latest_m5_data_order['close']
                    
                    setup_bias_for_order = "BUY" if order.type == mt5.ORDER_TYPE_BUY_STOP else "SELL"
                    invalidated = False
                    if setup_bias_for_order == "BUY" and current_m5_close < m5_ema21_for_invalidation:
                        invalidated = True
                        logger.info(f"Pending BUY order {order.ticket} on {order.symbol} invalidated (Close < M5_EMA21).")
                    elif setup_bias_for_order == "SELL" and current_m5_close > m5_ema21_for_invalidation:
                        invalidated = True
                        logger.info(f"Pending SELL order {order.ticket} on {order.symbol} invalidated (Close > M5_EMA21).")
                    
                    if invalidated:
                        cancel_pending_order(order.ticket)
                
                time.sleep(LOOP_SLEEP_SECONDS) 
                continue 

            if num_bot_positions == 0 and num_bot_pending_orders == 0:
                logger.debug(f"Scanning for new trade setups. Current Balance: {current_account_balance:.2f}")
                for sym_to_check_setup in SYMBOLS_AVAILABLE_FOR_TRADE:
                    props_setup = ALL_SYMBOL_PROPERTIES[sym_to_check_setup]
                    
                    if not is_within_session(TRADING_SESSIONS_UTC.get(sym_to_check_setup, [])):
                        logger.debug(f"Symbol {sym_to_check_setup} outside trading session. UTC Hour: {current_utc_time.hour}")
                        continue

                    current_spread = get_current_spread_pips(sym_to_check_setup, props_setup)
                    if current_spread > MAX_SPREAD_PIPS_ALLOWED:
                        logger.info(f"Spread for {sym_to_check_setup} ({current_spread:.1f} pips) too high. Max allowed: {MAX_SPREAD_PIPS_ALLOWED:.1f} pips. Skipping.")
                        continue

                    latest_h1, latest_m5, m5_lookback_df_setup = get_live_data_with_emas(sym_to_check_setup, H1_CANDLES_TO_FETCH, M5_CANDLES_TO_FETCH)

                    if latest_h1 is None or latest_m5 is None or m5_lookback_df_setup is None or m5_lookback_df_setup.empty:
                        logger.debug(f"Insufficient data (H1, M5, or lookback) for {sym_to_check_setup}. Skipping.")
                        continue
                    
                    pip_adj_setup = 3 * props_setup['pip_value_calc']

                    # Step 1: H1 Trend
                    h1_trend_bias_setup = None
                    # Check if EMAs are present in H1 data
                    if 'H1_EMA8' not in latest_h1 or 'H1_EMA21' not in latest_h1 or 'close' not in latest_h1 or \
                       pd.isna(latest_h1['H1_EMA8']) or pd.isna(latest_h1['H1_EMA21']) or pd.isna(latest_h1['close']):
                        logger.warning(f"H1 EMAs or close price missing for {sym_to_check_setup}. Skipping trend check.")
                        continue
                    h1_ema8 = latest_h1['H1_EMA8']; h1_ema21 = latest_h1['H1_EMA21']; h1_close = latest_h1['close']

                    if h1_ema8 > h1_ema21 and h1_close > h1_ema8 and h1_close > h1_ema21: h1_trend_bias_setup = "BUY"
                    elif h1_ema8 < h1_ema21 and h1_close < h1_ema8 and h1_close < h1_ema21: h1_trend_bias_setup = "SELL"
                    if h1_trend_bias_setup is None: continue

                    # Step 2: M5 Fanning
                    # Check if EMAs are present in M5 data
                    if 'M5_EMA8' not in latest_m5 or 'M5_EMA13' not in latest_m5 or 'M5_EMA21' not in latest_m5 or \
                       pd.isna(latest_m5['M5_EMA8']) or pd.isna(latest_m5['M5_EMA13']) or pd.isna(latest_m5['M5_EMA21']):
                        logger.warning(f"M5 EMAs missing for {sym_to_check_setup}. Skipping fanning check.")
                        continue
                    m5_ema8 = latest_m5['M5_EMA8']; m5_ema13 = latest_m5['M5_EMA13']; m5_ema21_val = latest_m5['M5_EMA21']

                    m5_fanned_buy = m5_ema8 > m5_ema13 and m5_ema13 > m5_ema21_val
                    m5_fanned_sell = m5_ema8 < m5_ema13 and m5_ema13 < m5_ema21_val
                    is_fanned_for_bias = (h1_trend_bias_setup == "BUY" and m5_fanned_buy) or \
                                         (h1_trend_bias_setup == "SELL" and m5_fanned_sell)
                    if not is_fanned_for_bias: continue
                    
                    m5_setup_bias_setup = h1_trend_bias_setup

                    if 'close' not in latest_m5 or pd.isna(latest_m5['close']): # Check M5 close for invalidation step
                        logger.warning(f"M5 close price missing for {sym_to_check_setup} for invalidation check. Skipping.")
                        continue
                    if (m5_setup_bias_setup == "BUY" and latest_m5['close'] < m5_ema21_val) or \
                       (m5_setup_bias_setup == "SELL" and latest_m5['close'] > m5_ema21_val):
                        logger.debug(f"[{sym_to_check_setup}] M5 Setup Invalidated (Close vs M5_EMA21 on latest M5 candle).")
                        continue

                    if 'low' not in latest_m5 or pd.isna(latest_m5['low']) or \
                       'high' not in latest_m5 or pd.isna(latest_m5['high']): # Check M5 low/high for pullback
                        logger.warning(f"M5 low/high price missing for {sym_to_check_setup} for pullback check. Skipping.")
                        continue
                    pullback = (m5_setup_bias_setup == "BUY" and latest_m5['low'] <= m5_ema8) or \
                               (m5_setup_bias_setup == "SELL" and latest_m5['high'] >= m5_ema8)
                    if not pullback: continue
                    
                    logger.info(f"Potential Setup Found for {sym_to_check_setup}: {m5_setup_bias_setup}")
                    trigger_bar_candle_data_setup = latest_m5
                    
                    if len(m5_lookback_df_setup) < 5 :
                        logger.warning(f"Not enough M5 lookback candles for entry/SL on {sym_to_check_setup} ({len(m5_lookback_df_setup)}). Skipping.")
                        continue

                    entry_px, sl_px, order_type_mt5 = (0,0,0)
                    if m5_setup_bias_setup == "BUY":
                        entry_px = m5_lookback_df_setup['high'].max() + pip_adj_setup
                        sl_px = trigger_bar_candle_data_setup['low'] - pip_adj_setup
                        order_type_mt5 = mt5.ORDER_TYPE_BUY_STOP
                    else: 
                        entry_px = m5_lookback_df_setup['low'].min() - pip_adj_setup
                        sl_px = trigger_bar_candle_data_setup['high'] + pip_adj_setup
                        order_type_mt5 = mt5.ORDER_TYPE_SELL_STOP
                    
                    entry_px = round(entry_px, props_setup['digits'])
                    sl_px = round(sl_px, props_setup['digits'])
                    sl_diff = abs(entry_px - sl_px)

                    if (m5_setup_bias_setup == "BUY" and sl_px >= entry_px) or \
                       (m5_setup_bias_setup == "SELL" and sl_px <= entry_px) or sl_diff < props_setup['trade_tick_size']:
                        logger.warning(f"[{sym_to_check_setup}] Illogical SL/Entry ({entry_px}/{sl_px}). Bias: {m5_setup_bias_setup}. Skipping.")
                        continue
                    
                    potential_loss_this_trade = current_account_balance * RISK_PER_TRADE_PERCENT
                    if todays_realized_pnl - potential_loss_this_trade < -max_daily_loss_allowed:
                        logger.info(f"DAILY RISK LIMIT HIT or would be exceeded for {sym_to_check_setup}. "
                                    f"Today's PNL: {todays_realized_pnl:.2f}, Potential Loss: {potential_loss_this_trade:.2f}, "
                                    f"Max Daily Loss Allowed: -{max_daily_loss_allowed:.2f}. Skipping trade.")
                        continue 

                    calc_lot = calculate_lot_size(current_account_balance, RISK_PER_TRADE_PERCENT, sl_diff, props_setup)
                    if calc_lot <= 0:
                        logger.warning(f"[{sym_to_check_setup}] Lot size 0 for setup. SL diff:{sl_diff:.{props_setup['digits']+1}f}. Balance:{current_account_balance:.2f}. Skipping.")
                        continue
                    
                    order_comment = f"{m5_setup_bias_setup}_SETUP_{sym_to_check_setup}"
                    order_ticket = place_pending_order(sym_to_check_setup, order_type_mt5, entry_px, sl_px, calc_lot, comment=order_comment)
                    
                    if order_ticket:
                        logger.info(f"Successfully placed PENDING {order_type_mt5} for {sym_to_check_setup} @ {entry_px}, SL {sl_px}, Lot {calc_lot}. Ticket: {order_ticket}")
                        break 
                    else:
                        logger.error(f"Failed to place pending order for {sym_to_check_setup}. Check logs.")

            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("Bot stopping due to user request (KeyboardInterrupt).")
    except Exception as e:
        logger.exception(f"An unhandled exception occurred in the main loop: {e}")
    finally:
        logger.info("Shutting down bot...")
        open_positions = mt5.positions_get(magic=BOT_MAGIC_NUMBER)
        if open_positions:
            logger.info(f"Found {len(open_positions)} open positions by this bot. Manual review recommended.")

        pending_orders = mt5.orders_get(magic=BOT_MAGIC_NUMBER)
        if pending_orders:
            logger.info(f"Found {len(pending_orders)} pending orders by this bot. Cancelling them...")
            for order in pending_orders:
                cancel_pending_order(order.ticket)
        
        shutdown_mt5_interface()
        logger.info("Bot has shut down.")