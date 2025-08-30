import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime, time as dt_time, timedelta
import pytz
import logging

# --- 1. BOT CONFIGURATION (with Backtest settings) ---
CONFIG = {
    # --- Backtest-Specific Settings ---
    "BACKTEST_START_DATE": "2010-01-01",
    "BACKTEST_END_DATE": "2025-08-20",
    "INITIAL_EQUITY": 200.0,

    # --- Trading Symbols (can be a smaller subset for faster backtesting) ---
    "SYMBOLS": ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                           "AUDJPY",   "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF", "CADCHF", "CADJPY", "EURCAD",
                           "GBPCAD", "NZDCAD", "GBPAUD", "GBPNZD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD", "EURAUD","NZDJPY",  "CHFJPY", "EURGBP",  "USDCNH","USDHKD", "USDMXN", 
                       "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD","AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500",
                       "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C"],
    
    # --- Core Strategy Settings (IDENTICAL TO LIVE BOT) ---
    "TIMEFRAME": mt5.TIMEFRAME_M5,
    "NY_TIMEZONE": pytz.timezone("America/New_York"),
    "OR_START_HOUR": 9, "OR_START_MIN": 30,
    "OR_END_HOUR": 9, "OR_END_MIN": 45,
    "TRADE_START_HOUR": 9, "TRADE_START_MIN": 45,
    "TRADE_END_HOUR": 12, "TRADE_END_MIN": 0,
    "RISK_PER_TRADE_PERCENT": 1.0,
    "ATR_PERIOD_STRENGTH": 14,
    "ATR_PERIOD_SL": 5,
    "IMPULSE_STRENGTH_FACTOR": 0.5,
    "SL_BUFFER_ATR_FACTOR": 1.5,
    "MIN_RR_RATIO": 2.0,
    "TRAIL_ACTIVATION_PROFIT_FACTOR": 0.8,
    "MAGIC_NUMBER": 202401,
    "PENDING_ORDER_TIMEOUT_MINUTES": 10,
}

# --- 2. LOGGING & GLOBALS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Backtester state variables
DAILY_STATE = {}
SYMBOL_PROPERTIES = {}

# --- 3. CORE LOGIC (Identical to Live Bot) ---
# NOTE: These functions are copied directly to ensure the logic is unchanged.

def get_or_range(df_today):
    """Calculates the high and low of the opening range."""
    or_start = dt_time(CONFIG['OR_START_HOUR'], CONFIG['OR_START_MIN'])
    or_end = dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN'])
    or_df = df_today.between_time(or_start, or_end)
    if or_df.empty: return None, None
    return or_df['high'].max(), or_df['low'].min()

def find_breakout_and_zone(df, or_high, or_low, bias, breakout_row, breakout_ts):
    """Validates the impulse move and defines the S/D zone."""
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

# --- 4. BACKTESTING ENGINE & DATA HANDLING ---

def initialize_mt5():
    """Connects to MT5 and gathers symbol properties for the backtest."""
    global SYMBOL_PROPERTIES
    if not mt5.initialize():
        logger.error("MT5 initialize() failed. Exiting.")
        return False
    logger.info("MT5 Initialized for data fetching.")
    
    for symbol in CONFIG["SYMBOLS"]:
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Could not get info for {symbol}. Skipping.")
            continue
        SYMBOL_PROPERTIES[symbol] = {'point': info.point, 'digits': info.digits}
        
    return True

def fetch_backtest_data():
    """Fetches historical data for all symbols for the entire backtest period."""
    all_data = {}
    start_dt = pytz.utc.localize(datetime.strptime(CONFIG["BACKTEST_START_DATE"], "%Y-%m-%d"))
    end_dt = pytz.utc.localize(datetime.strptime(CONFIG["BACKTEST_END_DATE"], "%Y-%m-%d") + timedelta(days=1))
    
    logger.info(f"Fetching data from {start_dt} to {end_dt}...")

    for symbol in CONFIG["SYMBOLS"]:
        rates = mt5.copy_rates_range(symbol, CONFIG["TIMEFRAME"], start_dt, end_dt)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data fetched for {symbol} in the given range.")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(CONFIG['NY_TIMEZONE'])
        df.set_index('time', inplace=True)
        
        # Pre-calculate indicators for the entire dataset
        df.ta.atr(length=CONFIG["ATR_PERIOD_STRENGTH"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_STRENGTH"]}',))
        df.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_SL"]}',))
        
        df.dropna(inplace=True)
        df['symbol'] = symbol
        all_data[symbol] = df
        logger.info(f"Fetched and processed {len(df)} bars for {symbol}.")
        
    return all_data

def analyze_results(trade_history, initial_equity):
    """Calculates and prints performance metrics from the trade history."""
    if not trade_history:
        logger.info("No trades were executed. No results to analyze.")
        return

    df = pd.DataFrame(trade_history)
    df['pnl_points'] = (df['exit_price'] - df['entry_price']) * df['direction']
    df['pnl_percent'] = (df['pnl_points'] * df['volume'] * SYMBOL_PROPERTIES[df['symbol'].iloc[0]]['point'] * 100000) / initial_equity # Approximation for Forex pairs

    wins = df[df['pnl_points'] > 0]
    losses = df[df['pnl_points'] <= 0]

    total_trades = len(df)
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl_points = df['pnl_points'].sum()
    
    avg_win_points = wins['pnl_points'].mean() if len(wins) > 0 else 0
    avg_loss_points = losses['pnl_points'].mean() if len(losses) > 0 else 0
    
    risk_reward_ratio = abs(avg_win_points / avg_loss_points) if avg_loss_points != 0 else float('inf')
    
    gross_profit = wins['pnl_points'].sum()
    gross_loss = abs(losses['pnl_points'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print("\n--- Backtest Results ---")
    print(f"Period: {CONFIG['BACKTEST_START_DATE']} to {CONFIG['BACKTEST_END_DATE']}")
    print(f"Symbols Tested: {', '.join(CONFIG['SYMBOLS'])}")
    print("-" * 25)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Win (Points): {avg_win_points:.5f}")
    print(f"Average Loss (Points): {avg_loss_points:.5f}")
    print(f"Avg Risk/Reward Ratio: {risk_reward_ratio:.2f}:1")
    print(f"Total Net PnL (Points): {total_pnl_points:.5f}")
    print("-" * 25)


# --- 5. MAIN BACKTESTING LOOP ---

def run_backtest(all_data):
    global DAILY_STATE
    
    # Simulation state
    open_position = None
    pending_orders = []
    trade_history = []
    
    # Combine all dataframes and sort by time to process chronologically
    if not all_data:
        logger.error("No data available for backtest. Exiting.")
        return []
        
    combined_df = pd.concat(all_data.values()).sort_index()
    
    logger.info(f"Starting backtest simulation loop with {len(combined_df)} total bars to process...")
    
    # Main loop iterates through each timestamp in the combined dataframe
    for timestamp, group in combined_df.groupby(level=0):
        current_time = timestamp.time()
        
        # --- A. Manage State & Check Triggers for existing orders/positions ---
        
        # Cancel stale pending orders
        pending_orders = [o for o in pending_orders if timestamp < o['expiry_time']]

        # Check for open position SL/TP hits
        if open_position:
            symbol = open_position['symbol']
            if symbol in group['symbol'].values:
                row = group[group['symbol'] == symbol].iloc[0]
                pos_type = open_position['type']
                
                # Check SL
                if (pos_type == 'BUY' and row['low'] <= open_position['sl']) or \
                   (pos_type == 'SELL' and row['high'] >= open_position['sl']):
                    exit_price = open_position['sl']
                    logger.info(f"{timestamp} | SL HIT for {symbol} {pos_type} at {exit_price}")
                    trade_history.append({**open_position, 'exit_time': timestamp, 'exit_price': exit_price, 'reason': 'SL'})
                    open_position = None
                
                # Check TP
                elif (pos_type == 'BUY' and row['high'] >= open_position['tp']) or \
                     (pos_type == 'SELL' and row['low'] <= open_position['tp']):
                    exit_price = open_position['tp']
                    logger.info(f"{timestamp} | TP HIT for {symbol} {pos_type} at {exit_price}")
                    trade_history.append({**open_position, 'exit_time': timestamp, 'exit_price': exit_price, 'reason': 'TP'})
                    open_position = None
                
                # Trail SL Logic
                else:
                    current_atr = row[f'ATR_{CONFIG["ATR_PERIOD_SL"]}']
                    entry_atr = open_position['entry_atr']
                    if current_atr < entry_atr * CONFIG["TRAIL_ACTIVATION_PROFIT_FACTOR"]:
                        in_profit = (pos_type == 'BUY' and row['close'] > open_position['entry_price']) or \
                                    (pos_type == 'SELL' and row['close'] < open_position['entry_price'])
                        if in_profit:
                            new_sl = 0
                            if pos_type == 'BUY':
                                new_sl = row['close'] - (current_atr * CONFIG['SL_BUFFER_ATR_FACTOR'])
                                if new_sl > open_position['sl']:
                                    open_position['sl'] = new_sl
                                    logger.info(f"{timestamp} | TRAIL SL for {symbol} BUY to {new_sl:.5f}")
                            else: # SELL
                                new_sl = row['close'] + (current_atr * CONFIG['SL_BUFFER_ATR_FACTOR'])
                                if new_sl < open_position['sl']:
                                    open_position['sl'] = new_sl
                                    logger.info(f"{timestamp} | TRAIL SL for {symbol} SELL to {new_sl:.5f}")

        # Check for pending order triggers (only if no position is open)
        if not open_position and pending_orders:
            for order in pending_orders[:]: # Iterate on a copy
                symbol = order['symbol']
                if symbol in group['symbol'].values:
                    row = group[group['symbol'] == symbol].iloc[0]
                    order_type = order['type']
                    
                    if (order_type == 'BUY_LIMIT' and row['low'] <= order['entry_price']) or \
                       (order_type == 'SELL_LIMIT' and row['high'] >= order['entry_price']):
                        
                        logger.info(f"{timestamp} | TRIGGERED {order_type} for {symbol} at {order['entry_price']:.5f}")
                        open_position = {
                            'symbol': symbol, 'entry_time': timestamp, 'entry_price': order['entry_price'],
                            'type': 'BUY' if order_type == 'BUY_LIMIT' else 'SELL', 'volume': 1.0, # Volume is fixed at 1 for point calculation
                            'sl': order['sl'], 'tp': order['tp'], 'entry_atr': order['sl_atr'],
                            'direction': 1 if order_type == 'BUY_LIMIT' else -1
                        }
                        pending_orders = [] # Cancel all other pending orders
                        break # Stop checking other orders

        # --- B. Look for New Signals (only if no position and no pending orders) ---
        if not open_position and not pending_orders:
            # Loop through each symbol present at this specific timestamp
            for _, row in group.iterrows():
                symbol = row['symbol']
                
                # Reset daily state if it's a new day
                today = timestamp.date()
                if symbol not in DAILY_STATE or DAILY_STATE[symbol].get('date') != today:
                    DAILY_STATE[symbol] = {'date': today, 'or_defined': False, 'bias': None, 'zone_fresh': False}

                state = DAILY_STATE[symbol]
                
                # We need the history for the current day up to the current bar
                df_hist = all_data[symbol].loc[:timestamp]
                df_today = df_hist[df_hist.index.date == today]
                
                if not (dt_time(CONFIG['TRADE_START_HOUR'], CONFIG['TRADE_START_MIN']) <= current_time < dt_time(CONFIG['TRADE_END_HOUR'], CONFIG['TRADE_END_MIN'])):
                    continue

                # 1. Define OR
                if not state['or_defined'] and current_time >= dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN']):
                    or_high, or_low = get_or_range(df_today)
                    if or_high:
                        state.update({'or_defined': True, 'or_high': or_high, 'or_low': or_low})

                # 2. Set bias and find zone
                if state.get('or_defined') and state['bias'] is None:
                    bias = None
                    if row['close'] > state['or_high']: bias = 'BULLISH'
                    elif row['close'] < state['or_low']: bias = 'BEARISH'
                    
                    if bias:
                        state['bias'] = bias
                        zone_info = find_breakout_and_zone(df_hist, state['or_high'], state['or_low'], bias, row, timestamp)
                        if zone_info:
                            state['zone_high'], state['zone_low'], state['impulse_peak'] = zone_info
                            state['zone_fresh'] = True
                
                # 3. Look for pullback entry
                if state.get('zone_fresh'):
                    entry_price = None
                    if state['bias'] == 'BULLISH' and row['low'] <= state['zone_high']:
                        entry_price = state['zone_high']
                    elif state['bias'] == 'BEARISH' and row['high'] >= state['zone_low']:
                        entry_price = state['zone_low']

                    if entry_price:
                        sl_atr = row[f'ATR_{CONFIG["ATR_PERIOD_SL"]}']
                        sl_buffer = sl_atr * CONFIG['SL_BUFFER_ATR_FACTOR']
                        
                        if state['bias'] == 'BULLISH':
                            sl = state['zone_low'] - sl_buffer
                            tp_candidate = state['impulse_peak']
                            # Ensure min RR
                            if (tp_candidate - entry_price) < CONFIG['MIN_RR_RATIO'] * (entry_price - sl):
                                tp = entry_price + (CONFIG['MIN_RR_RATIO'] * (entry_price - sl))
                            else:
                                tp = tp_candidate
                        else: # BEARISH
                            sl = state['zone_high'] + sl_buffer
                            tp_candidate = state['impulse_peak']
                            # Ensure min RR
                            if (entry_price - tp_candidate) < CONFIG['MIN_RR_RATIO'] * (sl - entry_price):
                                tp = entry_price - (CONFIG['MIN_RR_RATIO'] * (sl - entry_price))
                            else:
                                tp = tp_candidate

                        # Create a pending order
                        order_type = 'BUY_LIMIT' if state['bias'] == 'BULLISH' else 'SELL_LIMIT'
                        expiry = timestamp + timedelta(minutes=CONFIG['PENDING_ORDER_TIMEOUT_MINUTES'])
                        
                        pending_orders.append({
                            'symbol': symbol, 'type': order_type, 'entry_price': entry_price,
                            'sl': sl, 'tp': tp, 'sl_atr': sl_atr, 'place_time': timestamp,
                            'expiry_time': expiry
                        })
                        
                        logger.info(f"{timestamp} | PLACED PENDING {order_type} for {symbol} at {entry_price:.5f}")
                        
                        # Mark zone as used
                        state['zone_fresh'] = False
                        # Since we now place a pending order, we don't break, we allow orders for all symbols.
                        # The logic to only have one open trade is handled by checking 'if not open_position'.
    
    logger.info("Backtest simulation finished.")
    return trade_history

# --- 6. SCRIPT EXECUTION ---
if __name__ == "__main__":
    if initialize_mt5():
        try:
            # 1. Fetch all necessary historical data
            historical_data = fetch_backtest_data()
            
            # 2. Run the simulation
            results = run_backtest(historical_data)
            
            # 3. Analyze and print the results
            analyze_results(results, CONFIG["INITIAL_EQUITY"])
            
        finally:
            mt5.shutdown()
            logger.info("MT5 connection shut down.")