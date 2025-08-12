import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
import pytz
import logging

# --- 1. LIVE BOT CONFIGURATION ---
CONFIG = {
    # Trading symbols. WARNING: Start with a smaller list (5-10 symbols) for better performance.
    "SYMBOLS": ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                           "AUDJPY",   "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF", "CADCHF", "CADJPY", "EURCAD",
                           "GBPCAD", "NZDCAD", "GBPAUD", "GBPNZD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD", "EURAUD","NZDJPY",  "CHFJPY", "EURGBP",  "USDCNH","USDHKD", "USDMXN", 
                       "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD","AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500",
                       "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C"], # Reduced list for safety/performance
    # MT5 chart timeframe
    "TIMEFRAME": mt5.TIMEFRAME_M5,
    # New York Timezone for session filtering
    "NY_TIMEZONE": pytz.timezone("America/New_York"),
    # Opening Range (OR) times (NY Time)
    "OR_START_HOUR": 9, "OR_START_MIN": 30,
    "OR_END_HOUR": 9, "OR_END_MIN": 45,
    # Trading window (NY Time) - No new trades outside this window
    "TRADE_START_HOUR": 9, "TRADE_START_MIN": 45,
    "TRADE_END_HOUR": 12, "TRADE_END_MIN": 0,
    # Live Trade Management
    "MAX_CONCURRENT_TRADES": 5,
    "MAX_DAILY_DRAWDOWN_PERCENT": 5.0, # Set to 0 to disable this check
    # Strategy parameters (from your backtest)
    "ATR_PERIOD_STRENGTH": 14,
    "ATR_PERIOD_SL": 5,
    "IMPULSE_STRENGTH_FACTOR": 0.5,
    "SL_BUFFER_ATR_FACTOR": 1.5,
    "MIN_RR_RATIO": 2.0,
    "VOLATILITY_EXIT_FACTOR": 0.6,
    # Technical Settings
    "LOOP_SLEEP_SECONDS": 10,       # How often the main loop runs
    "SLIPPAGE": 5,                  # Allowed slippage for order execution
    "MAGIC_NUMBER": 123456,         # Unique ID for this bot's trades
}

# --- 2. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- 3. GLOBAL STATE VARIABLES ---
ALL_SYMBOL_PROPERTIES = {}
SYMBOLS_AVAILABLE_FOR_TRADE = []
session_start_balance = 0.0
symbol_states = {} # Dictionary to hold the state for each symbol

# --- 4. MT5 CONNECTION & UTILS ---

def initialize_mt5_interface():
    """Initializes the MT5 interface and prepares symbols for trading."""
    global SYMBOLS_AVAILABLE_FOR_TRADE, ALL_SYMBOL_PROPERTIES, session_start_balance, symbol_states
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    logger.info("MetaTrader 5 Initialized")

    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to get account info, error code = {mt5.last_error()}"); mt5.shutdown(); return False
    
    logger.info(f"Account Login: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
    session_start_balance = account_info.balance

    successfully_initialized_symbols = []
    temp_symbol_properties = {}
    for symbol_name in CONFIG["SYMBOLS"]:
        symbol_info_obj = mt5.symbol_info(symbol_name)
        if symbol_info_obj is None: logger.warning(f"Symbol {symbol_name} not found. Skipping."); continue
        if not symbol_info_obj.visible:
            if not mt5.symbol_select(symbol_name, True): logger.warning(f"symbol_select({symbol_name}) failed. Skipping."); continue
            time.sleep(0.5); symbol_info_obj = mt5.symbol_info(symbol_name)
        
        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point, 'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size, 'trade_tick_value': symbol_info_obj.trade_tick_value,
            'volume_min': symbol_info_obj.volume_min, 'volume_step': symbol_info_obj.volume_step if symbol_info_obj.volume_step > 0 else 0.01,
        }
        successfully_initialized_symbols.append(symbol_name)
    
    if not successfully_initialized_symbols: logger.error("No symbols were successfully initialized."); return False

    SYMBOLS_AVAILABLE_FOR_TRADE = successfully_initialized_symbols
    ALL_SYMBOL_PROPERTIES = temp_symbol_properties
    for symbol in SYMBOLS_AVAILABLE_FOR_TRADE: symbol_states[symbol] = reset_daily_state()
    logger.info(f"Properties recorded for: {SYMBOLS_AVAILABLE_FOR_TRADE}")
    return True

def shutdown_mt5_interface():
    mt5.shutdown()
    logger.info("MetaTrader 5 Shutdown")

def get_historical_data(symbol, timeframe, count=150):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0: return None
        df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}"); return None

# --- 5. ORDER & POSITION MANAGEMENT ---

def get_open_positions(): return mt5.positions_get(magic=CONFIG["MAGIC_NUMBER"]) or []
def get_pending_orders():
    orders = mt5.orders_get(magic=CONFIG["MAGIC_NUMBER"]) or []
    return [o for o in orders if o.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]]

def calculate_position_size(symbol):
    """MODIFIED: Returns the minimum allowed lot size for the symbol."""
    return ALL_SYMBOL_PROPERTIES.get(symbol, {}).get('volume_min', 0)

def place_limit_order(symbol, order_type, price, sl, tp, volume, entry_atr):
    request = {"action": mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": volume, "type": order_type,
               "price": price, "sl": sl, "tp": tp, "deviation": CONFIG["SLIPPAGE"], "magic": CONFIG["MAGIC_NUMBER"],
               "comment": f"ATR_EXIT_VAL:{entry_atr:.{ALL_SYMBOL_PROPERTIES[symbol]['digits']}f}",
               "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Successfully placed {order_type} for {symbol}: Ticket #{result.order}"); return result.order
    logger.error(f"Order Send failed for {symbol}: {result.comment if result else 'No response'}"); return None

def close_trade_by_ticket(pos, comment=""):
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume, "type": order_type,
               "position": pos.ticket, "deviation": CONFIG["SLIPPAGE"], "magic": CONFIG["MAGIC_NUMBER"], "comment": comment}
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE: logger.info(f"Successfully closed #{pos.ticket}. Reason: {comment}")
    else: logger.error(f"Failed to close #{pos.ticket}. Error: {mt5.last_error()}")

def cancel_pending_order(ticket):
    request = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE: logger.info(f"Successfully cancelled pending order #{ticket}.")
    else: logger.error(f"Failed to cancel pending order #{ticket}. Error: {mt5.last_error()}")

# --- 6. CORE LOGIC ---

def reset_daily_state():
    return {"daily_bias": None, "or_high": None, "or_low": None, "zone_defined": False,
            "order_placed_ticket": None, "trade_taken_today": False}

def run_bot():
    """Main execution loop for the live trading bot."""
    last_processed_day = None
    stop_trading_for_day_flag = False
    
    while True:
        try:
            ny_time = datetime.now(CONFIG["NY_TIMEZONE"])
            if last_processed_day != ny_time.day:
                logger.info(f"New trading day: {ny_time.strftime('%Y-%m-%d')}. Resetting states.")
                for symbol in SYMBOLS_AVAILABLE_FOR_TRADE: symbol_states[symbol] = reset_daily_state()
                last_processed_day = ny_time.day
                account_info = mt5.account_info(); session_start_balance = account_info.balance if account_info else 0
                stop_trading_for_day_flag = False

            if not stop_trading_for_day_flag and CONFIG["MAX_DAILY_DRAWDOWN_PERCENT"] > 0:
                account_info = mt5.account_info()
                if account_info and account_info.balance < session_start_balance:
                    drawdown = (session_start_balance - account_info.balance) / session_start_balance * 100
                    if drawdown >= CONFIG["MAX_DAILY_DRAWDOWN_PERCENT"]:
                        stop_trading_for_day_flag = True
                        logger.warning(f"Max daily drawdown hit. Stopping new trades for today. Cancelling pending orders.")
                        for order in get_pending_orders(): cancel_pending_order(order.ticket)

            for pos in get_open_positions():
                df_pos = get_historical_data(pos.symbol, CONFIG['TIMEFRAME'], 20)
                if df_pos is None: continue
                df_pos.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True)
                current_atr = df_pos[f'ATRr_{CONFIG["ATR_PERIOD_SL"]}'].iloc[-1]
                try:
                    entry_atr = float(pos.comment.split(":")[-1])
                    if current_atr < (entry_atr * CONFIG['VOLATILITY_EXIT_FACTOR']):
                        logger.info(f"[{pos.symbol}] Volatility exit triggered. Closing position #{pos.ticket}.")
                        close_trade_by_ticket(pos, "Volatility Exit")
                except (ValueError, IndexError): pass

            trade_start = ny_time.replace(hour=CONFIG["TRADE_START_HOUR"], minute=CONFIG["TRADE_START_MIN"], second=0)
            trade_end = ny_time.replace(hour=CONFIG["TRADE_END_HOUR"], minute=CONFIG["TRADE_END_MIN"], second=0)

            if ny_time > trade_end:
                 for order in get_pending_orders(): cancel_pending_order(order.ticket)
            
            if not (trade_start <= ny_time < trade_end) or stop_trading_for_day_flag:
                time.sleep(CONFIG["LOOP_SLEEP_SECONDS"]); continue

            for symbol in SYMBOLS_AVAILABLE_FOR_TRADE:
                state = symbol_states[symbol]
                if state['trade_taken_today']: continue
                
                df = get_historical_data(symbol, CONFIG["TIMEFRAME"])
                if df is None or len(df) < CONFIG["ATR_PERIOD_STRENGTH"]: continue
                
                df.ta.atr(length=CONFIG["ATR_PERIOD_STRENGTH"], append=True, col_names=(f'ATRr_{CONFIG["ATR_PERIOD_STRENGTH"]}',))
                df.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True, col_names=(f'ATRr_{CONFIG["ATR_PERIOD_SL"]}',))
                atr_strength = df[f'ATRr_{CONFIG["ATR_PERIOD_STRENGTH"]}'].iloc[-1]
                atr_sl = df[f'ATRr_{CONFIG["ATR_PERIOD_SL"]}'].iloc[-1]
                
                if state['daily_bias'] is None and ny_time >= ny_time.replace(hour=CONFIG["OR_END_HOUR"], minute=CONFIG["OR_END_MIN"]):
                    or_start_dt = ny_time.replace(hour=CONFIG["OR_START_HOUR"], minute=CONFIG["OR_START_MIN"], second=0)
                    or_end_dt = ny_time.replace(hour=CONFIG["OR_END_HOUR"], minute=CONFIG["OR_END_MIN"], second=0)
                    or_candles_utc = mt5.copy_rates_range(symbol, CONFIG["TIMEFRAME"], or_start_dt, or_end_dt)
                    if or_candles_utc is None or len(or_candles_utc) == 0: continue
                    state['or_high'] = pd.DataFrame(or_candles_utc)['high'].max()
                    state['or_low'] = pd.DataFrame(or_candles_utc)['low'].min()
                    
                    last_candle = df.iloc[-2]
                    if last_candle['close'] > state['or_high']: state['daily_bias'] = 'BULLISH'
                    elif last_candle['close'] < state['or_low']: state['daily_bias'] = 'BEARISH'

                if state['daily_bias'] and not state['zone_defined']:
                    df_historical = df.tz_localize('UTC').tz_convert(CONFIG['NY_TIMEZONE'])
                    or_start_time = ny_time.replace(hour=CONFIG["OR_START_HOUR"], minute=CONFIG["OR_START_MIN"], second=0)
                    
                    breakout_idx = -1
                    for i in range(len(df_historical) - 2, 0, -1):
                        candle_time = df_historical.index[i]
                        if candle_time < or_start_time: break
                        is_break = (state['daily_bias'] == 'BULLISH' and df_historical['close'][i] > state['or_high']) or \
                                   (state['daily_bias'] == 'BEARISH' and df_historical['close'][i] < state['or_low'])
                        if is_break: breakout_idx = i; break

                    if breakout_idx > 0:
                        impulse_start_candle_idx = -1
                        for i in range(breakout_idx, 0, -1):
                            is_down = df_historical['close'][i] < df_historical['open'][i]
                            if (state['daily_bias'] == 'BULLISH' and is_down) or (state['daily_bias'] == 'BEARISH' and not is_down):
                                impulse_start_candle_idx = i; break
                        
                        if impulse_start_candle_idx != -1:
                            zone_candle = df_historical.iloc[impulse_start_candle_idx]
                            impulse_run = df_historical.iloc[impulse_start_candle_idx+1 : breakout_idx+2]
                            zone_high, zone_low = zone_candle['high'], zone_candle['low']
                            impulse_peak = impulse_run['high'].max() if state['daily_bias'] == 'BULLISH' else impulse_run['low'].min()
                            impulse_distance = abs(impulse_peak - (zone_high if state['daily_bias'] == 'BULLISH' else zone_low))

                            if impulse_distance >= (CONFIG['IMPULSE_STRENGTH_FACTOR'] * atr_strength):
                                state['zone_defined'] = True
                                sl_buffer = CONFIG['SL_BUFFER_ATR_FACTOR'] * atr_sl
                                entry_price, sl, tp = 0, 0, 0
                                if state['daily_bias'] == 'BULLISH':
                                    entry_price, sl = zone_high, zone_low - sl_buffer
                                    risk = entry_price - sl
                                    potential_tp = impulse_peak
                                    tp = potential_tp if (potential_tp - entry_price) >= risk * CONFIG['MIN_RR_RATIO'] else entry_price + risk * CONFIG['MIN_RR_RATIO']
                                else: # BEARISH
                                    entry_price, sl = zone_low, zone_high + sl_buffer
                                    risk = sl - entry_price
                                    potential_tp = impulse_peak
                                    tp = potential_tp if (entry_price - potential_tp) >= risk * CONFIG['MIN_RR_RATIO'] else entry_price - risk * CONFIG['MIN_RR_RATIO']
                                
                                if len(get_open_positions()) + len(get_pending_orders()) < CONFIG["MAX_CONCURRENT_TRADES"]:
                                    volume = calculate_position_size(symbol)
                                    if volume > 0:
                                        order_type = mt5.ORDER_TYPE_BUY_LIMIT if state['daily_bias'] == 'BULLISH' else mt5.ORDER_TYPE_SELL_LIMIT
                                        ticket = place_limit_order(symbol, order_type, entry_price, sl, tp, volume, atr_sl)
                                        if ticket: state['order_placed_ticket'] = ticket
                                state['trade_taken_today'] = True # Mark as attempted

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
        time.sleep(CONFIG["LOOP_SLEEP_SECONDS"])

# --- 7. SCRIPT EXECUTION ---
if __name__ == "__main__":
    if initialize_mt5_interface():
        try:
            run_bot()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user (Ctrl+C).")
        finally:
            logger.info("Cancelling outstanding pending orders on exit...")
            for order in get_pending_orders():
                cancel_pending_order(order.ticket)
            shutdown_mt5_interface()