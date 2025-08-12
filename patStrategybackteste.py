import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, time as dt_time
import pytz
import logging
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
CONFIG = {
    # Backtest period
    "BACKTEST_START_DATE": "2010-01-01",
    "BACKTEST_END_DATE": "2025-07-30",
    # Trading symbols
    "SYMBOLS": ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                           "AUDJPY",   "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF", "CADCHF", "CADJPY", "EURCAD",
                           "GBPCAD", "NZDCAD", "GBPAUD", "GBPNZD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD", "EURAUD","NZDJPY",  "CHFJPY", "EURGBP",  "USDCNH","USDHKD", "USDMXN", 
                       "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD","AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500",
                       "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C"], # Keep it small for faster testing
    # Chart timeframe
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
    "INITIAL_BALANCE": 200.0,
    "RISK_PER_TRADE_PERCENT": 1.0,
    # Strategy parameters
    "ATR_PERIOD_STRENGTH": 14,
    "ATR_PERIOD_SL": 5,
    "IMPULSE_STRENGTH_FACTOR": 0.5, # Impulse distance >= 0.5 * ATR(14)
    "SL_BUFFER_ATR_FACTOR": 1.5,
    "MIN_RR_RATIO": 2.0,
    "TRAIL_ACTIVATION_PROFIT_FACTOR": 0.8, # Activate trailing if current_atr < entry_atr * 0.8
}

# --- 2. LOGGING & GLOBALS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

ALL_SYMBOL_PROPERTIES = {}
EPS = 1e-12

# --- 3. MT5 & DATA SETUP ---
def setup_and_fetch_data(symbols, start_date, end_date):
    """Initializes MT5 and fetches all historical data needed."""
    global ALL_SYMBOL_PROPERTIES
    if not mt5.initialize():
        logger.error("MT5 initialize() failed."); return None
    logger.info("MT5 Initialized for data fetching.")
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    dataframes = {}
    for symbol in symbols:
        info = mt5.symbol_info(symbol)
        if info is None: continue
        if not info.visible: mt5.symbol_select(symbol, True); time.sleep(0.5)

        ALL_SYMBOL_PROPERTIES[symbol] = {
            'point': info.point, 'digits': info.digits,
            'trade_tick_size': info.trade_tick_size, 'trade_tick_value': info.trade_tick_value,
            'volume_min': info.volume_min, 'currency_profit': info.currency_profit,
        }
        
        rates = mt5.copy_rates_range(symbol, CONFIG["TIMEFRAME"], start_dt, end_dt)
        if rates is None or len(rates) == 0: continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert(CONFIG['NY_TIMEZONE'])
        
        # Pre-calculate all necessary indicators
        df.ta.atr(length=CONFIG["ATR_PERIOD_STRENGTH"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_STRENGTH"]}',))
        df.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True, col_names=(f'ATR_{CONFIG["ATR_PERIOD_SL"]}',))
        df.dropna(inplace=True)
        
        dataframes[symbol] = df
        logger.info(f"Fetched {len(df)} bars for {symbol}.")

    mt5.shutdown()
    logger.info("MT5 Shutdown.")
    return dataframes

# --- 4. CORE ALGORITHM HELPERS ---

def get_or_range(df_today):
    """Calculates the Opening Range high and low."""
    or_start = dt_time(CONFIG['OR_START_HOUR'], CONFIG['OR_START_MIN'])
    or_end = dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN'])
    or_df = df_today.between_time(or_start, or_end)
    if or_df.empty:
        return None, None
    return or_df['high'].max(), or_df['low'].min()

def find_breakout_and_zone(df, or_high, or_low, bias, breakout_row, breakout_ts):
    """
    Finds the impulse and defines the S/D zone using only information available up to the breakout candle.
    THIS IS THE CAUSAL, NON-LOOKAHEAD VERSION.
    Returns: zone_high, zone_low, impulse_peak
    """
    # The "impulse candle" is the breakout_row itself.
    impulse_peak = breakout_row['high'] if bias == 'BULLISH' else breakout_row['low']

    # The "candle before the impulse" is the one just before our breakout candle.
    # We find its index location and get the previous row.
    breakout_idx = df.index.get_loc(breakout_ts)
    if breakout_idx == 0:
        return None # Cannot get previous candle if it's the first in the dataset
    
    # In this new logic, the zone candle is simply the one before the breakout.
    zone_candle = df.iloc[breakout_idx - 1]
    zone_high, zone_low = zone_candle['high'], zone_candle['low']

    # Qualify the zone based on impulse strength of the breakout candle
    strength_atr = zone_candle[f'ATR_{CONFIG["ATR_PERIOD_STRENGTH"]}']
    if bias == 'BULLISH':
        # Impulse is from the top of the zone to the high of the breakout candle
        impulse_distance = breakout_row['high'] - zone_high
    else: # BEARISH
        # Impulse is from the bottom of the zone to the low of the breakout candle
        impulse_distance = zone_low - breakout_row['low']

    if impulse_distance < (strength_atr * CONFIG["IMPULSE_STRENGTH_FACTOR"]):
        # logger.debug(f"[{breakout_ts}] Impulse for {df.iloc[0]['symbol']} not strong enough.")
        return None # Impulse was not strong enough

    return zone_high, zone_low, impulse_peak

# --- 5. SIMULATION & TRADING FUNCTIONS ---

def check_risk_and_get_volume(symbol, equity, sl_price_distance):
    """
    Ensures the trade respects the 1% risk rule with the minimum lot size.
    """
    props = ALL_SYMBOL_PROPERTIES.get(symbol)
    if not props: return 0.0

    # This strategy ALWAYS uses the minimum lot size.
    volume = props['volume_min']
    
    max_risk_dollars = equity * (CONFIG["RISK_PER_TRADE_PERCENT"] / 100.0)

    # Simplified risk check for backtesting (assumes USD account and direct pairs)
    # A full implementation would need the currency conversion logic.
    if props['currency_profit'] != 'USD':
         # In a real scenario, you'd convert. For this backtest, we assume it's close enough.
         pass
    
    tick_value = props['trade_tick_value']
    tick_size = props['trade_tick_size']

    if tick_value == 0 or tick_size == 0 or sl_price_distance <= 0:
        return 0.0

    value_per_point = tick_value / tick_size
    potential_loss = sl_price_distance * value_per_point * volume

    if potential_loss > max_risk_dollars:
        # logger.info(f"SKIP {symbol}: Risk ${potential_loss:.2f} > Max Risk ${max_risk_dollars:.2f}")
        return 0.0
    
    return volume

def open_trade(sim_account, signal, entry_price, entry_ts):
    """Simulates opening a trade."""
    sl_distance = abs(entry_price - signal['sl'])
    volume = check_risk_and_get_volume(signal['symbol'], sim_account['equity'], sl_distance)
    if volume <= 0:
        return False

    trade = {
        'symbol': signal['symbol'], 'type': signal['bias'],
        'volume': volume, 'entry_price': entry_price,
        'sl': signal['sl'], 'tp': signal['tp'], 'entry_time': entry_ts,
        'entry_atr': signal['sl_atr'], # ATR at the time of entry signal
        'trailing_active': False
    }
    sim_account['open_trades'].append(trade)
    logger.info(f"[{entry_ts}] OPEN {trade['type']} {trade['symbol']} @ {entry_price:.5f} | SL={trade['sl']:.5f} TP={trade['tp']:.5f}")
    return True

def close_trade(sim_account, trade, exit_price, exit_ts, reason):
    """Simulates closing a trade and records the result."""
    props = ALL_SYMBOL_PROPERTIES[trade['symbol']]
    pnl = 0
    price_diff = (exit_price - trade['entry_price']) if trade['type'] == 'BULLISH' else (trade['entry_price'] - exit_price)
    
    if props['trade_tick_size'] > 0:
        pnl = (price_diff / props['trade_tick_size']) * props['trade_tick_value'] * trade['volume']

    record = {
        'symbol': trade['symbol'], 'pnl': pnl, 'reason': reason,
        'entry_time': trade['entry_time'], 'exit_time': exit_ts,
    }
    sim_account['closed_trades'].append(record)
    sim_account['equity'] += pnl
    sim_account['equity_curve'].append({'time': exit_ts, 'equity': sim_account['equity']})
    sim_account['open_trades'].remove(trade)
    logger.info(f"[{exit_ts}] CLOSE {trade['symbol']} @ {exit_price:.5f} | PnL={pnl:.2f} ({reason})")

# --- 6. MAIN BACKTEST LOOP ---
def run_backtest(symbol_dfs):
    sim_account = {
        'equity': CONFIG['INITIAL_BALANCE'],
        'open_trades': [], 'closed_trades': [],
        'equity_curve': [{'time': list(symbol_dfs.values())[0].index[0], 'equity': CONFIG['INITIAL_BALANCE']}]
    }

    # Group data by day to process daily logic
    all_data = pd.concat(symbol_dfs.values(), keys=symbol_dfs.keys(), names=['symbol', 'time']).sort_index()
    daily_groups = all_data.groupby(all_data.index.get_level_values('time').date)

    daily_states = {s: {} for s in symbol_dfs.keys()}

    for day, df_day_all_symbols in daily_groups:
        if sim_account.get('open_trades'): continue # Skip to next day if trade is held overnight

        # --- 1. Preprocessing for the day ---
        for symbol in symbol_dfs.keys():
            daily_states[symbol] = {'bias': None, 'zone_high': None, 'zone_low': None, 'zone_fresh': False, 'impulse_peak': None}
            df_sym_day = df_day_all_symbols.loc[df_day_all_symbols.index.get_level_values('symbol') == symbol]
            df_sym_day = df_sym_day.reset_index(level='symbol')
            if df_sym_day.empty: continue
            
            or_high, or_low = get_or_range(df_sym_day)
            if or_high is None: continue
            daily_states[symbol]['or_high'] = or_high
            daily_states[symbol]['or_low'] = or_low

        # --- 2. Iterate through the timestamps of the day ---
        timestamps = sorted(df_day_all_symbols.index.get_level_values('time').unique())
        
        trade_start = dt_time(CONFIG['TRADE_START_HOUR'], CONFIG['TRADE_START_MIN'])
        trade_end = dt_time(CONFIG['TRADE_END_HOUR'], CONFIG['TRADE_END_MIN'])
        
        for ts in timestamps:
            current_time = ts.time()

            # --- A. Manage Open Trade ---
            if sim_account.get('open_trades'):
                trade = sim_account['open_trades'][0]
                try:
                    row = all_data.loc[(trade['symbol'], ts)]
                except KeyError: continue # No data for this symbol at this timestamp
                
                # Trailing Stop Logic (from previous bot)
                current_atr = row[f'ATR_{CONFIG["ATR_PERIOD_SL"]}']
                if not trade['trailing_active'] and current_atr < trade['entry_atr'] * CONFIG["TRAIL_ACTIVATION_PROFIT_FACTOR"]:
                    in_profit = (trade['type'] == 'BULLISH' and row['close'] > trade['entry_price']) or \
                                (trade['type'] == 'BEARISH' and row['close'] < trade['entry_price'])
                    if in_profit:
                        trade['trailing_active'] = True
                        logger.info(f"[{ts}] Trailing SL activated for {trade['symbol']}")
                
                if trade['trailing_active']:
                    if trade['type'] == 'BULLISH':
                        new_sl = row['close'] - (current_atr * CONFIG['SL_BUFFER_ATR_FACTOR'])
                        trade['sl'] = max(trade['sl'], new_sl)
                    else:
                        new_sl = row['close'] + (current_atr * CONFIG['SL_BUFFER_ATR_FACTOR'])
                        trade['sl'] = min(trade['sl'], new_sl)
                
                # Check for SL/TP hit
                if trade['type'] == 'BULLISH':
                    if row['low'] <= trade['sl']: close_trade(sim_account, trade, trade['sl'], ts, 'SL'); break
                    if row['high'] >= trade['tp']: close_trade(sim_account, trade, trade['tp'], ts, 'TP'); break
                else: # BEARISH
                    if row['high'] >= trade['sl']: close_trade(sim_account, trade, trade['sl'], ts, 'SL'); break
                    if row['low'] <= trade['tp']: close_trade(sim_account, trade, trade['tp'], ts, 'TP'); break
                continue # If trade is open, do nothing else for this timestamp

            # --- B. Look for Entry Signals (if no trade is open) ---
            if not sim_account.get('open_trades'):
                candidates = []
                for symbol in symbol_dfs.keys():
                    state = daily_states[symbol]
                    try:
                        row = all_data.loc[(symbol, ts)]
                    except KeyError: continue

                    # a. Determine daily bias if not set yet
                    or_end_time = dt_time(CONFIG['OR_END_HOUR'], CONFIG['OR_END_MIN'])
                    if state['bias'] is None and state.get('or_high') and current_time > or_end_time:
                        if row['close'] > state['or_high']: state['bias'] = 'BULLISH'
                        elif row['close'] < state['or_low']: state['bias'] = 'BEARISH'
                        
                        if state['bias']:
                            logger.info(f"[{ts}] Session bias for {symbol} set to: {state['bias']}")
                            # Find and define the zone using the causal function
                            # We pass the current 'row' which represents the breakout candle
                            zone_info = find_breakout_and_zone(symbol_dfs[symbol], state['or_high'], state['or_low'], state['bias'], row, ts)
                            if zone_info:
                                # The new function returns 3 items, not 4
                                state['zone_high'], state['zone_low'], state['impulse_peak'] = zone_info
                                state['zone_fresh'] = True
                                logger.info(f"[{ts}] {symbol} S/D Zone defined: {state['zone_low']:.5f} - {state['zone_high']:.5f}")
                    # b. If bias and a fresh zone exist, look for pullback entry
                    if state.get('bias') and state.get('zone_fresh') and trade_start <= current_time < trade_end:
                        entry_price = None
                        if state['bias'] == 'BULLISH' and row['low'] <= state['zone_high']:
                            entry_price = state['zone_high'] # Enter at top of demand zone
                        elif state['bias'] == 'BEARISH' and row['high'] >= state['zone_low']:
                            entry_price = state['zone_low'] # Enter at bottom of supply zone
                        
                        if entry_price:
                            state['zone_fresh'] = False # Zone is no longer fresh
                            
                            # Calculate SL and TP based on the algorithm
                            sl_atr = row[f'ATR_{CONFIG["ATR_PERIOD_SL"]}']
                            sl_buffer = sl_atr * CONFIG['SL_BUFFER_ATR_FACTOR']
                            
                            if state['bias'] == 'BULLISH':
                                sl = state['zone_low'] - sl_buffer
                                tp = state['impulse_peak']
                                # Enforce min R:R
                                if (tp - entry_price) < CONFIG['MIN_RR_RATIO'] * (entry_price - sl):
                                    tp = entry_price + (CONFIG['MIN_RR_RATIO'] * (entry_price - sl))
                            else: # BEARISH
                                sl = state['zone_high'] + sl_buffer
                                tp = state['impulse_peak']
                                # Enforce min R:R
                                if (entry_price - tp) < CONFIG['MIN_RR_RATIO'] * (sl - entry_price):
                                    tp = entry_price - (CONFIG['MIN_RR_RATIO'] * (sl - entry_price))
                            
                            candidates.append({
                                'symbol': symbol, 'bias': state['bias'], 'sl': sl, 'tp': tp,
                                'entry_price': entry_price, 'entry_ts': ts, 'sl_atr': sl_atr,
                                'score': abs(state['impulse_peak'] - (state['zone_high'] if state['bias'] == 'BULLISH' else state['zone_low'])) # Score by impulse size
                            })
                
                # c. If we have candidates, pick the best one and open the trade
                if candidates:
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    best_signal = candidates[0]
                    logger.info(f"[{ts}] Best signal found: {best_signal['symbol']}. Attempting to open trade.")
                    if open_trade(sim_account, best_signal, best_signal['entry_price'], best_signal['entry_ts']):
                        # Once trade is open, stop looking for more signals today
                        pass

    return sim_account

# --- 7. REPORTING ---
def generate_report(sim_account):
    """Generates and prints a performance report."""
    history_df = pd.DataFrame(sim_account['closed_trades'])
    if history_df.empty:
        logger.info("No trades were executed during the backtest."); return

    logger.info("\n--- Backtest Performance Report ---")
    
    initial_balance = CONFIG['INITIAL_BALANCE']
    final_equity = sim_account['equity']
    
    total_trades = len(history_df)
    wins = history_df[history_df['pnl'] > 0]
    win_rate = len(wins) / total_trades * 100
    
    profit_factor = abs(wins['pnl'].sum() / history_df[history_df['pnl'] < 0]['pnl'].sum()) if history_df[history_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
    
    print(f"Period: {CONFIG['BACKTEST_START_DATE']} to {CONFIG['BACKTEST_END_DATE']}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Equity:    ${final_equity:,.2f}")
    print(f"Net PnL:         ${(final_equity - initial_balance):,.2f} ({(final_equity/initial_balance - 1)*100:.2f}%)")
    print("-" * 35)
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Profit Factor:   {profit_factor:.2f}")

    equity_curve_df = pd.DataFrame(sim_account['equity_curve']).set_index('time')
    equity_curve_df['high_water_mark'] = equity_curve_df['equity'].cummax()
    equity_curve_df['drawdown'] = (equity_curve_df['high_water_mark'] - equity_curve_df['equity']) / equity_curve_df['high_water_mark'] * 100
    max_dd = equity_curve_df['drawdown'].max()
    print(f"Max Drawdown:    {max_dd:.2f}%")
    print("-" * 35)

    plt.figure(figsize=(12, 6))
    equity_curve_df['equity'].plot(title='Equity Curve', grid=True)
    plt.show()

# --- 8. SCRIPT EXECUTION ---
if __name__ == "__main__":
    backtest_data = setup_and_fetch_data(
        CONFIG["SYMBOLS"], 
        CONFIG["BACKTEST_START_DATE"], 
        CONFIG["BACKTEST_END_DATE"]
    )
    
    if backtest_data:
        final_account = run_backtest(backtest_data)
        generate_report(final_account)
    else:
        logger.error("Could not run backtest due to data fetching issues.")