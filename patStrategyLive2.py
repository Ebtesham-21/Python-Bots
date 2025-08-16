import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, time as dt_time
import pytz
import logging

# --- 1. BOT CONFIGURATION ---
CONFIG = {
    # Trading symbols to monitor
    "SYMBOLS": ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                           "AUDJPY",   "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF", "CADCHF", "CADJPY", "EURCAD",
                           "GBPCAD", "NZDCAD", "GBPAUD", "GBPNZD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD", "EURAUD","NZDJPY",  "CHFJPY", "EURGBP",  "USDCNH","USDHKD", "USDMXN", 
                       "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD","AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500",
                       "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C"],
    # MT5 chart timeframe
    "TIMEFRAME": mt5.TIMEFRAME_M5,
    # Session timezone
    "NY_TIMEZONE": pytz.timezone("America/New_York"),
    # Opening Range (OR) times (NY Time)
    "OR_START_HOUR": 9, "OR_START_MIN": 30,
    "OR_END_HOUR": 9, "OR_END_MIN": 45,
    # Trading window (NY Time)
    "TRADE_START_HOUR": 9, "TRADE_START_MIN": 45,
    "TRADE_END_HOUR": 12, "TRADE_END_MIN": 0,
    # Risk management
    "RISK_PER_TRADE_PERCENT": 1.0,
    # Strategy parameters (IDENTICAL TO BACKTEST)
    "ATR_PERIOD_STRENGTH": 14,
    "ATR_PERIOD_SL": 5,
    "IMPULSE_STRENGTH_FACTOR": 0.5,
    "SL_BUFFER_ATR_FACTOR": 1.5,
    "MIN_RR_RATIO": 2.0,
    "TRAIL_ACTIVATION_PROFIT_FACTOR": 0.8,
    # Bot operational parameters
    "MAGIC_NUMBER": 202401,  # Unique ID for this bot's orders
    "LOOP_SLEEP_SECONDS": 30,
    "PENDING_ORDER_TIMEOUT_MINUTES": 10,
    # How often the bot checks for signals/manages trades
    "BARS_TO_FETCH": 100,      # Number of recent bars to fetch for indicator calculation
}

# --- 2. LOGGING & GLOBALS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

ALL_SYMBOL_PROPERTIES = {}
# This dictionary will hold the daily state for each symbol
DAILY_STATE = {}
# This will hold the entry ATR for the currently managed trade
MANAGED_TRADE_STATE = {}

# --- 3. CORE LOGIC (Identical to Backtester) ---

def get_or_range(df_today):
    or_start = dt_time(CONFIG['OR_START_HOUR'], CONFIG['OR_START_MIN'])
    or_end = dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN'])
    or_df = df_today.between_time(or_start, or_end)
    if or_df.empty: return None, None
    return or_df['high'].max(), or_df['low'].min()

def find_breakout_and_zone(df, or_high, or_low, bias, breakout_row, breakout_ts):
    impulse_peak = breakout_row['high'] if bias == 'BULLISH' else breakout_row['low']
    breakout_idx = df.index.get_loc(breakout_ts)
    if breakout_idx == 0: return None
    
    zone_candle = df.iloc[breakout_idx - 1]
    zone_high, zone_low = zone_candle['high'], zone_candle['low']

    strength_atr = zone_candle[f'ATR_{CONFIG["ATR_PERIOD_STRENGTH"]}']
    impulse_distance = (breakout_row['high'] - zone_high) if bias == 'BULLISH' else (zone_low - breakout_row['low'])

    if impulse_distance < (strength_atr * CONFIG["IMPULSE_STRENGTH_FACTOR"]):
        return None
        
    return zone_high, zone_low, impulse_peak

# --- 4. LIVE TRADING & MT5 INTERACTION ---

def initialize_bot():
    """Connects to MT5 and gathers symbol properties."""
    global ALL_SYMBOL_PROPERTIES
    if not mt5.initialize():
        logger.error("MT5 initialize() failed. Exiting."); return False
    
    account_info = mt5.account_info()
    if not account_info:
        logger.error("Could not get account info. Exiting."); return False
        
    logger.info(f"MT5 Initialized on account {account_info.login}. Balance: {account_info.balance:.2f} {account_info.currency}")

    for symbol in CONFIG["SYMBOLS"]:
        info = mt5.symbol_info(symbol)
        if info is None: continue
        if not info.visible: mt5.symbol_select(symbol, True); time.sleep(0.5)

        ALL_SYMBOL_PROPERTIES[symbol] = {
            'point': info.point, 'digits': info.digits,
            'trade_tick_size': info.trade_tick_size, 'trade_tick_value': info.trade_tick_value,
            'volume_min': info.volume_min,
        }
    logger.info("Bot initialized successfully.")
    return True

def get_latest_data(symbol):
    """Fetches the latest bars and calculates indicators."""
    rates = mt5.copy_rates_from_pos(symbol, CONFIG["TIMEFRAME"], 0, CONFIG["BARS_TO_FETCH"])
    if rates is None or len(rates) == 0: return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(CONFIG['NY_TIMEZONE'])
    df.set_index('time', inplace=True)
    
    df.ta.atr(length=CONFIG["ATR_PERIOD_STRENGTH"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_STRENGTH"]}',))
    df.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_SL"]}',))
    return df

def check_risk_and_get_volume(symbol, sl_price_distance):
    """Calculates trade volume based on risk percentage of REAL account equity."""
    props = ALL_SYMBOL_PROPERTIES.get(symbol)
    if not props: return 0.0

    account_info = mt5.account_info()
    if not account_info: return 0.0

    volume = props['volume_min']
    max_risk_dollars = account_info.equity * (CONFIG["RISK_PER_TRADE_PERCENT"] / 100.0)

    tick_value = props['trade_tick_value']
    tick_size = props['trade_tick_size']

    if tick_value == 0 or tick_size == 0 or sl_price_distance <= 0:
        logger.warning(f"Cannot calculate risk for {symbol}, using min volume without check.")
        return volume

    value_per_point = tick_value / tick_size
    potential_loss = sl_price_distance * value_per_point * volume
    
    if potential_loss > max_risk_dollars:
        logger.info(f"SKIP {symbol}: Risk ${potential_loss:.2f} > Max Risk ${max_risk_dollars:.2f}")
        return 0.0
    
    return volume

def place_limit_order(signal):
    """
    Attempts to place a real limit order.
    Returns True on success, False on failure.
    """
    global MANAGED_TRADE_STATE
    
    volume = check_risk_and_get_volume(signal['symbol'], abs(signal['entry_price'] - signal['sl']))
    if volume <= 0:
        # This is not an error, it's the risk gate working. No need to log here as check_risk logs it.
        return False # Signal failure to the main loop

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal['bias'] == 'BULLISH' else mt5.ORDER_TYPE_SELL_LIMIT

    request = {
        "action": mt5.TRADE_ACTION_PENDING, "symbol": signal['symbol'], "volume": volume,
        "type": order_type, "price": signal['entry_price'], "sl": signal['sl'], "tp": signal['tp'],
        "magic": CONFIG["MAGIC_NUMBER"], "comment": "S/D Pullback Bot",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"SUCCESS: Placed {signal['bias']} LIMIT order for {signal['symbol']} @ {signal['entry_price']:.5f}, Order Ticket: {result.order}")
        MANAGED_TRADE_STATE[result.order] = {'entry_atr': signal['sl_atr'], 'status': 'PENDING'}
        return True # Signal success
    else:
        logger.error(f"FAILED to place LIMIT order for {signal['symbol']}. Code: {result.retcode}, Comment: {result.comment}")
        return False # Signal failure

def modify_position_sl(position, new_sl):
    """Modifies the stop loss of an open position."""
    request = {
        "action": mt5.TRADE_ACTION_SLTP, "position": position.ticket,
        "sl": round(new_sl, ALL_SYMBOL_PROPERTIES[position.symbol]['digits']),
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"SUCCESS: Trailed SL for ticket {position.ticket} to {new_sl:.5f}")
    else:
        logger.error(f"FAILED to trail SL for {position.ticket}. Code: {result.retcode}, Comment: {result.comment}")

def manage_daily_state(symbol):
    """Resets the state for a symbol if it's a new day."""
    today = datetime.now(CONFIG['NY_TIMEZONE']).date()
    if symbol not in DAILY_STATE or DAILY_STATE[symbol].get('date') != today:
        DAILY_STATE[symbol] = {
            'date': today, 'or_defined': False, 'bias': None, 'zone_fresh': False,
        }
        logger.info(f"State for {symbol} reset for new day: {today}")

def sync_trade_state():
    """
    Finds any open positions and pending orders, syncs their state, and cleans up any closed/cancelled ones.
    This version correctly handles the state of pending orders.
    """
    global MANAGED_TRADE_STATE
    
    positions = mt5.positions_get(magic=CONFIG["MAGIC_NUMBER"])
    orders = mt5.orders_get(magic=CONFIG["MAGIC_NUMBER"])
    
    # --- Part 1: Reconstruct state for any active positions found without it ---
    for pos in positions:
        if pos.ticket not in MANAGED_TRADE_STATE:
            logger.warning(f"Position {pos.ticket} found without a state. Reconstructing...")
            
            open_timestamp = pd.to_datetime(pos.time, unit='s', utc=True)
            rates = mt5.copy_rates_from(pos.symbol, CONFIG["TIMEFRAME"], open_timestamp, 20)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Could not fetch history to reconstruct state for position {pos.ticket}. Cannot trail.")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
            df.set_index('time', inplace=True)
            df.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_SL"]}',))

            last_candle_before_open = df[df.index < open_timestamp].iloc[-1]
            
            if pd.notna(last_candle_before_open[f'ATR_{CONFIG["ATR_PERIOD_SL"]}']):
                reconstructed_atr = last_candle_before_open[f'ATR_{CONFIG["ATR_PERIOD_SL"]}']
                MANAGED_TRADE_STATE[pos.ticket] = {'entry_atr': reconstructed_atr, 'status': 'ACTIVE_RECONSTRUCTED'}
                logger.info(f"State for position {pos.ticket} reconstructed with entry_atr: {reconstructed_atr:.5f}")
            else:
                logger.error(f"Failed to find valid ATR for position {pos.ticket}. Cannot trail.")

    # --- Part 2: Clean up the state dictionary ---
    # Create a set of all known active tickets (both from positions and pending orders)
    active_tickets = {pos.ticket for pos in positions} | {order.ticket for order in orders}

    # Find any state entries for tickets that are no longer active
    stale_tickets = [ticket for ticket in MANAGED_TRADE_STATE if ticket not in active_tickets]
    
    for ticket in stale_tickets:
        logger.info(f"Position/Order {ticket} is closed or cancelled. Removing from state.")
        del MANAGED_TRADE_STATE[ticket]

def cancel_stale_pending_orders():
    """
    Checks all pending orders and cancels any that are older than the configured timeout.
    """
    pending_orders = mt5.orders_get(magic=CONFIG["MAGIC_NUMBER"])
    if not pending_orders:
        return

    # Get the current time once, being mindful of timezones
    now_utc = datetime.now(pytz.utc)
    
    for order in pending_orders:
        order_time_utc = pd.to_datetime(order.time_setup, unit='s', utc=True)
        time_since_placement = now_utc - order_time_utc
        
        if time_since_placement.total_seconds() > CONFIG["PENDING_ORDER_TIMEOUT_MINUTES"] * 60:
            logger.info(f"Pending order {order.ticket} for {order.symbol} has expired. Cancelling...")
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "magic": CONFIG["MAGIC_NUMBER"]
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Successfully cancelled stale order {order.ticket}.")
                # The sync_trade_state function will clean up the state dictionary later
            else:
                logger.error(f"Failed to cancel stale order {order.ticket}. Code: {result.retcode}")

# --- 5. MAIN TRADING LOOP ---
def run_live_trading_loop():
    while True:
        try:
            sync_trade_state()
            # --- THIS IS THE NEW LINE ---
            # Clean up any old pending orders before looking for new ones
            cancel_stale_pending_orders()
            # --- END OF NEW LINE ---
            current_time = datetime.now(CONFIG['NY_TIMEZONE']).time()
            or_end_time = dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN'])
            trade_start_time = dt_time(CONFIG['TRADE_START_HOUR'], CONFIG['TRADE_START_MIN'])
            trade_end_time = dt_time(CONFIG['TRADE_END_HOUR'], CONFIG['TRADE_END_MIN'])

            # A. Manage any currently open position
            open_positions = mt5.positions_get(magic=CONFIG["MAGIC_NUMBER"])
            if open_positions:
                position = open_positions[0] # We only manage one trade at a time
                if position.ticket not in MANAGED_TRADE_STATE:
                    logger.warning(f"Position {position.ticket} is missing state. Cannot apply trailing SL.")
                else:
                    df = get_latest_data(position.symbol)
                    if df is not None:
                        current_atr = df.iloc[-2][f'ATR_{CONFIG["ATR_PERIOD_SL"]}']
                        entry_atr = MANAGED_TRADE_STATE[position.ticket]['entry_atr']
                        
                        # Trailing SL Logic
                        if current_atr < entry_atr * CONFIG["TRAIL_ACTIVATION_PROFIT_FACTOR"]:
                            in_profit = (position.price_current > position.price_open) if position.type == mt5.ORDER_TYPE_BUY else (position.price_current < position.price_open)
                            if in_profit:
                                logger.info(f"Trailing SL activated for {position.ticket}")
                                new_sl = 0
                                if position.type == mt5.ORDER_TYPE_BUY:
                                    new_sl = position.price_current - (current_atr * CONFIG['SL_BUFFER_ATR_FACTOR'])
                                    if new_sl > position.sl: modify_position_sl(position, new_sl)
                                else: # SELL
                                    new_sl = position.price_current + (current_atr * CONFIG['SL_BUFFER_ATR_FACTOR'])
                                    if new_sl < position.sl: modify_position_sl(position, new_sl)
            else:
                 # Clear any lingering state if position was closed
                if MANAGED_TRADE_STATE:
                    logger.info("Position closed, clearing trade state.")
                    MANAGED_TRADE_STATE.clear()


            # B. Look for new trade signals (only if no position and no pending orders)
            pending_orders = mt5.orders_get(magic=CONFIG["MAGIC_NUMBER"])
            if not open_positions and not pending_orders:
                candidates = []
                for symbol in CONFIG["SYMBOLS"]:
                    manage_daily_state(symbol)
                    state = DAILY_STATE[symbol]

                    # Only proceed if we are in the trading window
                    if not (trade_start_time <= current_time < trade_end_time): continue
                        
                    df = get_latest_data(symbol)
                    if df is None: continue
                    
                    # 1. Define OR if not already defined for the day
                    if not state['or_defined'] and current_time > or_end_time:
                        or_high, or_low = get_or_range(df)
                        if or_high:
                            state['or_defined'] = True
                            state['or_high'] = or_high
                            state['or_low'] = or_low
                            logger.info(f"OR for {symbol} defined: H={or_high:.5f} L={or_low:.5f}")

                    # 2. Set bias and find zone if OR is defined but bias is not
                    if state.get('or_defined') and state['bias'] is None:
                        row = df.iloc[-2] # Last closed candle
                        bias = None
                        if row['close'] > state['or_high']: bias = 'BULLISH'
                        elif row['close'] < state['or_low']: bias = 'BEARISH'
                        
                        if bias:
                            state['bias'] = bias
                            logger.info(f"Session bias for {symbol} set to: {bias}")
                            zone_info = find_breakout_and_zone(df, state['or_high'], state['or_low'], bias, row, row.name)
                            if zone_info:
                                state['zone_high'], state['zone_low'], state['impulse_peak'] = zone_info
                                state['zone_fresh'] = True
                                logger.info(f"S/D Zone for {symbol} defined: {state['zone_low']:.5f} - {state['zone_high']:.5f}")

                    # 3. Look for pullback entry if bias and fresh zone exist
                    if state.get('zone_fresh'):
                        row = df.iloc[-2]
                        entry_price = None
                        if state['bias'] == 'BULLISH' and row['low'] <= state['zone_high']:
                            entry_price = state['zone_high']
                        elif state['bias'] == 'BEARISH' and row['high'] >= state['zone_low']:
                            entry_price = state['zone_low']
                        
                        if entry_price:
                            state['zone_fresh'] = False
                            
                            sl_atr = row[f'ATR_{CONFIG["ATR_PERIOD_SL"]}']
                            sl_buffer = sl_atr * CONFIG['SL_BUFFER_ATR_FACTOR']
                            
                            if state['bias'] == 'BULLISH':
                                sl = state['zone_low'] - sl_buffer
                                tp = state['impulse_peak']
                                if (tp - entry_price) < CONFIG['MIN_RR_RATIO'] * (entry_price - sl):
                                    tp = entry_price + (CONFIG['MIN_RR_RATIO'] * (entry_price - sl))
                            else: # BEARISH
                                sl = state['zone_high'] + sl_buffer
                                tp = state['impulse_peak']
                                if (entry_price - tp) < CONFIG['MIN_RR_RATIO'] * (sl - entry_price):
                                    tp = entry_price - (CONFIG['MIN_RR_RATIO'] * (sl - entry_price))
                            
                            candidates.append({
                                'symbol': symbol, 'bias': state['bias'], 'sl': sl, 'tp': tp,
                                'entry_price': entry_price, 'sl_atr': sl_atr,
                                'score': abs(state['impulse_peak'] - (state['zone_high'] if state['bias'] == 'BULLISH' else state['zone_low']))
                            })
                
                # c. If we have candidates, try to place an order for the best available one
                if candidates:
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    logger.info(f"Found {len(candidates)} trade candidate(s). Attempting to place order...")
                    
                    # Loop through sorted candidates and try to place an order
                    for signal in candidates:
                        logger.info(f"Attempting to place order for best candidate: {signal['symbol']} ({signal['bias']})")
                        
                        # place_limit_order now returns True/False
                        if place_limit_order(signal):
                            # If the order was successfully placed, stop trying.
                            logger.info("Order placed successfully. Halting search for new trades.")
                            break # Exit the 'for signal in candidates' loop
                    
                    # If the loop finishes without placing an order, it means all were rejected by the risk gate.
                    # No special action is needed; the bot will just try again on the next cycle.
            time.sleep(CONFIG["LOOP_SLEEP_SECONDS"])

        except KeyboardInterrupt:
            logger.info("Bot shutting down by user request.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            time.sleep(60)

# --- 6. SCRIPT EXECUTION ---
if __name__ == "__main__":
    if initialize_bot():
        try:
            run_live_trading_loop()
        finally:
            # Clean up any pending orders on exit
            pending_orders = mt5.orders_get(magic=CONFIG["MAGIC_NUMBER"])
            if pending_orders:
                logger.info("Cancelling all pending orders before shutdown...")
                for order in pending_orders:
                    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket, "magic": CONFIG["MAGIC_NUMBER"]}
                    mt5.order_send(req)
            
            mt5.shutdown()
            logger.info("MT5 connection shut down.")