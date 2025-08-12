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
    "SYMBOLS": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "XAUUSD"],
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
    # Risk management
    "RISK_PER_TRADE_PERCENT": 1.0,
    # Strategy parameters
    "ATR_PERIOD_SL": 5, # Using the shorter ATR for SL calculation
    "SL_BUFFER_ATR_FACTOR": 1.5,
    "MIN_RR_RATIO": 2.0,
    # Bot operational parameters
    "MAGIC_NUMBER": 123457,  # Unique ID for this bot's orders
    "LOOP_SLEEP_SECONDS": 30,
    "BARS_TO_FETCH": 100, # More bars to ensure OR period is covered
}

# --- 2. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- 3. GLOBAL STATE & HELPERS ---
ALL_SYMBOL_PROPERTIES = {}
# This dictionary will hold the daily state for each symbol
# e.g., {'EURUSD': {'date': ..., 'or_high': ..., 'trade_taken': ...}}
DAILY_STATE = {}

def get_current_time_ny():
    return datetime.now(CONFIG['NY_TIMEZONE'])

# --- 4. CORE STRATEGY LOGIC (OPENING RANGE BREAKOUT) ---

def manage_daily_state(symbol):
    """Resets the state for a symbol if it's a new day."""
    global DAILY_STATE
    today = get_current_time_ny().date()
    
    if symbol not in DAILY_STATE or DAILY_STATE[symbol].get('date') != today:
        DAILY_STATE[symbol] = {
            'date': today,
            'or_defined': False,
            'or_high': None,
            'or_low': None,
            'trade_taken': False,
        }
        logger.info(f"State for {symbol} has been reset for new day: {today}")

def define_opening_range(symbol, df):
    """Calculates and sets the OR high/low for the day."""
    global DAILY_STATE
    
    or_start = dt_time(CONFIG['OR_START_HOUR'], CONFIG['OR_START_MIN'])
    or_end = dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN'])

    # Filter the dataframe for the OR time period of the current day
    or_df = df.between_time(or_start, or_end)
    
    if not or_df.empty:
        or_high = or_df['high'].max()
        or_low = or_df['low'].min()
        DAILY_STATE[symbol]['or_high'] = or_high
        DAILY_STATE[symbol]['or_low'] = or_low
        DAILY_STATE[symbol]['or_defined'] = True
        logger.info(f"Opening Range for {symbol} defined: High={or_high:.5f}, Low={or_low:.5f}")
    else:
        logger.warning(f"Could not define OR for {symbol}: No data found in the {or_start}-{or_end} period.")

def check_or_breakout_signal(symbol, df):
    """Checks for a breakout of the defined Opening Range."""
    state = DAILY_STATE.get(symbol, {})
    if not state.get('or_defined') or state.get('trade_taken'):
        return None

    # Time-based filter for active trading
    current_time = get_current_time_ny().time()
    trade_start = dt_time(CONFIG['TRADE_START_HOUR'], CONFIG['TRADE_START_MIN'])
    trade_end = dt_time(CONFIG['TRADE_END_HOUR'], CONFIG['TRADE_END_MIN'])
    if not (trade_start <= current_time < trade_end):
        return None

    row = df.iloc[-2]  # Last closed candle
    or_high = state['or_high']
    or_low = state['or_low']

    trade_type = None
    if row['close'] > or_high and row['low'] > or_low: # Clean break above
        trade_type = 'BUY'
    elif row['close'] < or_low and row['high'] < or_high: # Clean break below
        trade_type = 'SELL'
        
    if trade_type:
        sl_atr = row.get(f'ATR_{CONFIG["ATR_PERIOD_SL"]}')
        if pd.isna(sl_atr): return None
        
        sl_distance = sl_atr * CONFIG["SL_BUFFER_ATR_FACTOR"]
        tp_distance = sl_distance * CONFIG["MIN_RR_RATIO"]

        entry_price = row['close']
        if trade_type == 'BUY':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else: # SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        logger.info(f"OR Breakout Signal: {trade_type} {symbol} breaking range {or_low:.5f}-{or_high:.5f}")
        DAILY_STATE[symbol]['trade_taken'] = True # Mark that we've taken a trade for this symbol today

        return {
            'symbol': symbol,
            'type': trade_type,
            'signal_price': entry_price,
            'sl': sl,
            'tp': tp,
        }
    return None

# --- 5. LIVE TRADING & MT5 INTERACTION FUNCTIONS (Mostly Unchanged) ---

def initialize_bot():
    """Connects to MT5 and gathers symbol properties."""
    global ALL_SYMBOL_PROPERTIES
    if not mt5.initialize():
        logger.error("MT5 initialize() failed. Exiting.")
        return False
    
    account_info = mt5.account_info()
    if not account_info:
        logger.error("Could not get account info. Exiting.")
        return False
        
    logger.info(f"MT5 Initialized. Account: {account_info.name}, Balance: {account_info.balance:.2f} {account_info.currency}")

    for symbol in CONFIG["SYMBOLS"]:
        symbol_info_obj = mt5.symbol_info(symbol)
        if symbol_info_obj is None:
            logger.warning(f"Symbol {symbol} not found. Skipping.")
            continue
        if not symbol_info_obj.visible:
            mt5.symbol_select(symbol, True)
            time.sleep(0.5)

        ALL_SYMBOL_PROPERTIES[symbol] = {
            'point': symbol_info_obj.point,
            'digits': symbol_info_obj.digits,
            'volume_min': symbol_info_obj.volume_min,
            'trade_tick_value': symbol_info_obj.trade_tick_value,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
        }
        manage_daily_state(symbol) # Initialize state for each symbol

    logger.info("Bot initialized successfully.")
    return True

def get_latest_data(symbol):
    """Fetches the latest bars and calculates indicators."""
    rates = mt5.copy_rates_from_pos(symbol, CONFIG["TIMEFRAME"], 0, CONFIG["BARS_TO_FETCH"])
    if rates is None or len(rates) == 0:
        logger.warning(f"Could not fetch data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(CONFIG['NY_TIMEZONE'])
    df.set_index('time', inplace=True)
    
    df.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_SL"]}',))
    return df

def check_risk_and_get_volume(symbol, sl_price_distance):
    """Calculates trade volume based on risk percentage of REAL account equity."""
    symbol_props = ALL_SYMBOL_PROPERTIES.get(symbol)
    if not symbol_props: return 0.0

    account_info = mt5.account_info()
    if not account_info: return 0.0

    equity = account_info.equity
    max_risk_dollars = equity * (CONFIG["RISK_PER_TRADE_PERCENT"] / 100.0)

    volume = symbol_props['volume_min']
    tick_value = symbol_props['trade_tick_value']
    tick_size = symbol_props['trade_tick_size']

    if tick_value == 0 or tick_size == 0 or sl_price_distance <= 0:
        logger.warning(f"Cannot calculate risk for {symbol} due to missing data. Allowing min volume.")
        return volume

    value_per_point = tick_value / tick_size
    potential_loss_dollars = sl_price_distance * value_per_point * volume
    
    if potential_loss_dollars > max_risk_dollars:
        logger.info(f"SKIPPING TRADE [{symbol}]: Risk ${potential_loss_dollars:.2f} > Max Risk ${max_risk_dollars:.2f}")
        return 0.0
    
    return volume

def place_market_order(signal):
    """Places a real market order on MT5."""
    symbol = signal['symbol']
    trade_type = signal['type']
    
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"Could not get live tick for {symbol}. Cancelling order.")
        return

    entry_price = tick.ask if trade_type == 'BUY' else tick.bid
    
    sl_dist = abs(signal['signal_price'] - signal['sl'])
    tp_dist = abs(signal['tp'] - signal['signal_price'])
    
    if trade_type == 'BUY':
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist
        
    volume = check_risk_and_get_volume(symbol, abs(entry_price - sl))
    if volume <= 0: return

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if trade_type == 'BUY' else mt5.ORDER_TYPE_SELL,
        "price": entry_price, "sl": round(sl, ALL_SYMBOL_PROPERTIES[symbol]['digits']),
        "tp": round(tp, ALL_SYMBOL_PROPERTIES[symbol]['digits']),
        "magic": CONFIG["MAGIC_NUMBER"], "comment": "OR Breakout Bot",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"SUCCESS: Opened {trade_type} {symbol} @ {result.price}, Ticket: {result.order}")
    else:
        logger.error(f"FAILED to open trade for {symbol}. Code: {result.retcode}, Comment: {result.comment}")

# --- 6. MAIN TRADING LOOP ---
def run_live_trading_loop():
    """The main continuous loop for the live bot."""
    while True:
        try:
            current_time_ny = get_current_time_ny()
            or_end_time = dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN'])

            # Only run checks during the broader trading day
            if dt_time(8, 0) <= current_time_ny.time() <= dt_time(17, 0):
                
                # We can only trade if there are no open positions from this bot
                if not mt5.positions_get(magic=CONFIG["MAGIC_NUMBER"]):
                    
                    for symbol in CONFIG["SYMBOLS"]:
                        manage_daily_state(symbol) # Ensure state is up-to-date
                        state = DAILY_STATE[symbol]

                        df = get_latest_data(symbol)
                        if df is None: continue
                        
                        # Define OR if period has passed and it's not yet defined
                        if not state['or_defined'] and current_time_ny.time() > or_end_time:
                            define_opening_range(symbol, df)

                        # Check for breakout signal if OR is defined
                        if state['or_defined']:
                            signal = check_or_breakout_signal(symbol, df)
                            if signal:
                                place_market_order(signal)
                                # Since we only want one trade at a time, we break the symbol loop
                                break 
            
            # Wait for the next loop iteration
            time.sleep(CONFIG["LOOP_SLEEP_SECONDS"])

        except KeyboardInterrupt:
            logger.info("Bot shutting down by user request.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            time.sleep(60)

# --- 7. SCRIPT EXECUTION ---
if __name__ == "__main__":
    if initialize_bot():
        try:
            run_live_trading_loop()
        finally:
            mt5.shutdown()
            logger.info("MT5 connection shut down.")