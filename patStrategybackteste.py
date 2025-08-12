import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
import pytz
import math
import logging
import matplotlib.pyplot as plt



EPS = 1e-12
# --- 1. BACKTEST CONFIGURATION ---
CONFIG = {
    # Backtest period
    "BACKTEST_START_DATE": "2025-02-01",
    "BACKTEST_END_DATE": "2025-02-28",
    # Trading symbols
    "SYMBOLS": ["EURUSD", "USDCHF",   "GBPJPY", "GBPUSD",
                           "AUDJPY",   "EURNZD", "NZDUSD", "AUDUSD", "USDCAD","USDJPY", "EURJPY","EURCHF", "CADCHF", "CADJPY", "EURCAD",
                           "GBPCAD", "NZDCAD", "GBPAUD", "GBPNZD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD", "EURAUD","NZDJPY",  "CHFJPY", "EURGBP",  "USDCNH","USDHKD", "USDMXN", 
                       "USOIL", "UKOIL", "XAUUSD", "XAGUSD",
                       "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD","AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "US500",
                       "USTEC", "INTC", "MO", "BABA", "ABT", "LI", "TME", "ADBE", "MMM", "WMT", "PFE", "EQIX", "F", "ORCL", "BA", "NKE", "C"], # Keep it to a few symbols for faster backtesting
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
    "INITIAL_BALANCE": 200.0,
    "RISK_PER_TRADE_PERCENT": 1.0,
    # Strategy parameters (same as live bot)
    "ATR_PERIOD_STRENGTH": 14,
    "ATR_PERIOD_SL": 5,
    "IMPULSE_STRENGTH_FACTOR": 0.5,
    "SL_BUFFER_ATR_FACTOR": 1.5,
    "MIN_RR_RATIO": 2.0,
    "VOLATILITY_EXIT_FACTOR": 0.6,
}

# --- 2. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- 3. GLOBAL VARIABLES ---
ALL_SYMBOL_PROPERTIES = {}

# --- 4. MT5 CONNECTION & DATA FETCHING ---
def setup_and_fetch_data(symbols, start_date, end_date):
    """Initializes MT5, gets symbol properties, and fetches all historical data needed for the backtest."""
    global ALL_SYMBOL_PROPERTIES
    if not mt5.initialize():
        logger.error("MT5 initialize() failed.")
        return None
    logger.info("MT5 Initialized for data fetching.")
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    dataframes = {}
    for symbol in symbols:
        # Get Symbol Properties (reusing part of the live bot's init)
        symbol_info_obj = mt5.symbol_info(symbol)
        if symbol_info_obj is None:
            logger.warning(f"Symbol {symbol} not found. Skipping.")
            continue
        if not symbol_info_obj.visible: mt5.symbol_select(symbol, True)
        time.sleep(0.5)

        ALL_SYMBOL_PROPERTIES[symbol] = {
            'point': symbol_info_obj.point,
            'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
            'trade_tick_value': symbol_info_obj.trade_tick_value,
            'volume_min': symbol_info_obj.volume_min,
            'volume_step': symbol_info_obj.volume_step if symbol_info_obj.volume_step > 0 else 0.01,
        }
        
        # Fetch Data
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        rates = mt5.copy_rates_range(symbol, CONFIG["TIMEFRAME"], start_dt, end_dt)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data found for {symbol} in the specified range. Skipping.")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # IMPORTANT: MT5 data is in UTC. Convert to NY time for logic processing.
        df.set_index('time', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert(CONFIG['NY_TIMEZONE'])
        
        # Pre-calculate indicators to avoid recalculating in the loop
        df.ta.atr(length=CONFIG["ATR_PERIOD_STRENGTH"], append=True)
        df.ta.atr(length=CONFIG["ATR_PERIOD_SL"], append=True)
        df.dropna(inplace=True) # Drop rows with NaN indicators at the start
        
        dataframes[symbol] = df
        logger.info(f"Successfully fetched and prepared {len(df)} bars for {symbol}.")

    mt5.shutdown()
    logger.info("MT5 Shutdown.")
    return dataframes

# --- 5. SIMULATION & HELPER FUNCTIONS ---

# -------------------- Helper: compute signal strength --------------------
def compute_signal_strength(entry_price, sl, tp, entry_atr):
    """
    Scores a signal so we can compare them when multiple appear at once.
    - impulse: distance from entry to SL in ATR units
    - rr: reward/risk ratio
    Final score is weighted: impulse (65%) + rr (35%)
    """
    impulse = abs(entry_price - sl) / (entry_atr + EPS)
    rr = abs(tp - entry_price) / (max(abs(entry_price - sl), EPS))
    score = 0.65 * impulse + 0.35 * rr
    return score

# -------------------- Helper: open a trade --------------------
def open_trade(sim_account, symbol, trade_type, entry_price, sl, tp, entry_atr, timestamp):
    sl_distance = abs(entry_price - sl)
    volume = check_risk_and_get_volume(symbol, sim_account['equity'], sl_distance)
    if not volume or volume <= 0:
        return False

    trade = {
        'symbol': symbol,
        'type': trade_type,
        'volume': volume,
        'entry_price': entry_price,
        'sl': sl,
        'tp': tp,
        'entry_time': timestamp,
        'entry_atr': entry_atr,
        'trailing_active': False,
        'trail_distance': None
    }
    sim_account['open_trades'].append(trade)
    print(f"[{timestamp}] OPEN {trade_type} {symbol} @ {entry_price:.5f} vol={volume} (SL={sl:.5f}, TP={tp:.5f})")
    return True

# -------------------- Helper: close a trade --------------------
def close_trade(sim_account, trade, exit_price, timestamp, reason="close"):
    symbol_props = ALL_SYMBOL_PROPERTIES[trade['symbol']]
    
    # --- THIS IS THE CORRECTED PNL CALCULATION ---
    tick_size = symbol_props['trade_tick_size']
    tick_value = symbol_props['trade_tick_value']

    # Safety check to prevent division by zero
    if tick_size <= 0:
        logger.warning(f"Symbol {trade['symbol']} has invalid tick_size {tick_size}. PNL will be 0.")
        pnl = 0
    else:
        if trade['type'] == 'BUY':
            price_difference = exit_price - trade['entry_price']
        else: # SELL
            price_difference = trade['entry_price'] - exit_price
            
        # 1. Calculate how many ticks the price moved
        ticks_difference = price_difference / tick_size
        
        # 2. Calculate PnL using the correct formula
        pnl = ticks_difference * tick_value * trade['volume']
    # --- END OF CORRECTION ---

    record = {
        'symbol': trade['symbol'],
        'type': trade['type'],
        'volume': trade['volume'],
        'entry_price': trade['entry_price'],
        'exit_price': exit_price,
        'sl': trade['sl'],
        'tp': trade['tp'],
        'entry_time': trade['entry_time'],
        'exit_time': timestamp,
        'entry_atr': trade.get('entry_atr'),
        'pnl': pnl,
        'reason': reason
    }
    
    # Use 'closed_trades' consistently
    sim_account.setdefault('closed_trades', []).append(record)
    sim_account['balance'] += pnl
    sim_account['equity'] += pnl
    
    # Record equity change for the chart
    sim_account['equity_curve'].append({'time': timestamp, 'equity': sim_account['equity']})
    
    sim_account['open_trades'].remove(trade)
    print(f"[{timestamp}] CLOSE {trade['symbol']} {trade['type']} exit={exit_price:.5f} pnl={pnl:.2f} reason={reason}")


def reset_daily_state():
    # ... (this function remains the same)
    return {
        "daily_bias": None, "or_high": None, "or_low": None, "zone_defined": False,
        "zone_high": None, "zone_low": None, "zone_touched": False, "entry_price": None,
        "stop_loss": None, "take_profit": None, "order_placed": False,
        "trade_taken_today": False,
    }

# --- NEW REPLACEMENT FUNCTION ---
def check_risk_and_get_volume(symbol, equity, sl_price_distance):
    """
    This function acts as a final risk gatekeeper.
    1. It uses the FIXED minimum lot size for the symbol.
    2. It calculates the potential loss if this trade hits the stop loss.
    3. It compares this potential loss to the max allowed risk (1% of equity).
    4. If risk is acceptable, it returns the minimum lot size.
    5. If risk is too high, it logs a warning and returns 0 (skipping the trade).
    """
    symbol_props = ALL_SYMBOL_PROPERTIES.get(symbol)
    if not symbol_props:
        logger.warning(f"No properties for {symbol}, cannot size position.")
        return 0

    # Rule 1: Always use the minimum lot size
    volume = symbol_props['volume_min']

    # --- The Risk Check ---
    # Calculate the maximum allowed loss in dollars based on equity
    max_allowed_loss_dollars = equity * (CONFIG["RISK_PER_TRADE_PERCENT"] / 100.0)

    # Calculate the potential loss in dollars for this specific trade
    tick_value = symbol_props['trade_tick_value']
    tick_size = symbol_props['trade_tick_size']
    if tick_value == 0 or tick_size == 0 or sl_price_distance <= 0:
        return 0 # Cannot calculate risk

    value_per_point = tick_value / tick_size
    potential_loss_dollars = sl_price_distance * value_per_point * volume
    
    # Rule 2 & 3: The Decision Gate
    if potential_loss_dollars <= max_allowed_loss_dollars:
        # Risk is acceptable, return the lot size to proceed with the trade
        return volume
    else:
        # Risk is too high, skip the trade by returning 0
        logger.info(
            f"SKIPPING TRADE [{symbol}]: Potential risk ${potential_loss_dollars:.2f} "
            f"(with {volume} lots) exceeds max allowed risk of ${max_allowed_loss_dollars:.2f}"
        )
        return 0


# --- STRATEGY LOGIC & SIGNAL DETECTION ---
def check_entry_signal(symbol, row):
    """
    This is the core strategy function.
    It analyzes a single row (candle) of data and determines if a trade signal exists.

    Args:
        symbol (str): The symbol being checked.
        row (pd.Series): A row from the symbol's DataFrame, containing o,h,l,c and indicators.

    Returns:
        dict: A dictionary with trade details if a signal is found, otherwise None.
    """
    # --- Time-based filter: Only trade within the allowed window ---
    current_time = row.name.time()
    trade_start_time = datetime.strptime(f"{CONFIG['TRADE_START_HOUR']}:{CONFIG['TRADE_START_MIN']}", "%H:%M").time()
    trade_end_time = datetime.strptime(f"{CONFIG['TRADE_END_HOUR']}:{CONFIG['TRADE_END_MIN']}", "%H:%M").time()

    if not (trade_start_time <= current_time < trade_end_time):
        return None # Outside of trading hours

    # --- Strategy Logic: Impulse Candle ---
    entry_atr = row.get(f'ATRr_{CONFIG["ATR_PERIOD_STRENGTH"]}')
    sl_atr = row.get(f'ATRr_{CONFIG["ATR_PERIOD_SL"]}')

    if pd.isna(entry_atr) or pd.isna(sl_atr):
        return None # Not enough data for indicators

    candle_body = abs(row['close'] - row['open'])

    # Check for a strong impulse move
    if candle_body > entry_atr * CONFIG["IMPULSE_STRENGTH_FACTOR"]:
        trade_type = None
        if row['close'] > row['open']:
            trade_type = 'BUY'
        elif row['close'] < row['open']:
            trade_type = 'SELL'

        if trade_type:
            entry_price = row['close']
            sl_distance = sl_atr * CONFIG["SL_BUFFER_ATR_FACTOR"]
            tp_distance = sl_distance * CONFIG["MIN_RR_RATIO"]

            if trade_type == 'BUY':
                sl = entry_price - sl_distance
                tp = entry_price + tp_distance
            else: # SELL
                sl = entry_price + sl_distance
                tp = entry_price - tp_distance

            # Return a dictionary with all the required trade information
            return {
                'type': trade_type,
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'entry_atr': entry_atr,
            }

    # If no signal was found, return None
    return None




# -------------------- Main Backtest Loop --------------------
def run_backtest_single_trade_at_time(symbol_dfs, sim_account, CONFIG):
    all_timestamps = sorted(set().union(*[set(df.index) for df in symbol_dfs.values()]))

    for ts in all_timestamps:
        # ====== A) Manage the single open trade (if any) ======
        if sim_account.get('open_trades'):
            trade = sim_account['open_trades'][0]
            df = symbol_dfs.get(trade['symbol'])
            if df is not None and ts in df.index:
                row = df.loc[ts]
                current_atr = row.get(f'ATRr_{CONFIG["ATR_PERIOD_SL"]}', None)

                # --- Dynamic trailing SL activation ---
                if current_atr is not None:
                    if (not trade['trailing_active']) and (current_atr < trade['entry_atr'] * 0.8):
                        # Optional: require trade to be in profit before activating
                        in_profit = (trade['type'] == 'BUY' and row['close'] > trade['entry_price']) or \
                                    (trade['type'] == 'SELL' and row['close'] < trade['entry_price'])
                        if in_profit:
                            trade['trailing_active'] = True

                    if trade['trailing_active']:
                        atr_ratio = current_atr / (trade['entry_atr'] + EPS)
                        unrealized_atr = abs(row['close'] - trade['entry_price']) / (current_atr + EPS)

                        if atr_ratio >= 0.6:
                            multiplier = 1.5
                        elif atr_ratio >= 0.5:
                            multiplier = 1.0
                        else:
                            multiplier = 0.5

                        if unrealized_atr > 3:
                            multiplier = min(multiplier, 0.8)

                        trade['trail_distance'] = current_atr * multiplier
                        if trade['type'] == 'BUY':
                            trade['sl'] = max(trade['sl'], row['close'] - trade['trail_distance'])
                        else:
                            trade['sl'] = min(trade['sl'], row['close'] + trade['trail_distance'])

                # --- SL/TP checks (with intrabar heuristic) ---
                if trade['type'] == 'BUY':
                    hit_sl = row['low'] <= trade['sl']
                    hit_tp = row['high'] >= trade['tp']
                else:
                    hit_sl = row['high'] >= trade['sl']
                    hit_tp = row['low'] <= trade['tp']

                if hit_sl and hit_tp:
                    if row['close'] >= row['open']:
                        if trade['type'] == 'BUY':
                            close_trade(sim_account, trade, trade['sl'], ts, reason='SL (intrabar)')
                        else:
                            close_trade(sim_account, trade, trade['tp'], ts, reason='TP (intrabar)')
                    else:
                        if trade['type'] == 'BUY':
                            close_trade(sim_account, trade, trade['tp'], ts, reason='TP (intrabar)')
                        else:
                            close_trade(sim_account, trade, trade['sl'], ts, reason='SL (intrabar)')
                elif hit_sl:
                    close_trade(sim_account, trade, trade['sl'], ts, reason='SL')
                elif hit_tp:
                    close_trade(sim_account, trade, trade['tp'], ts, reason='TP')

        # ====== B) If no trade open, gather & rank signals ======
        if not sim_account.get('open_trades'):
            candidates = []
            for symbol, df in symbol_dfs.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]

                # Your existing signal detection here:
                sig = check_entry_signal(symbol, row)
                if not sig:
                    continue

                trade_type = sig['type']  # should be 'BUY' or 'SELL'
                entry_price = sig['entry_price']
                sl = sig['sl']
                tp = sig['tp']
                entry_atr = sig['entry_atr']

                score = compute_signal_strength(entry_price, sl, tp, entry_atr)
                candidates.append({
                    'symbol': symbol,
                    'type': trade_type,
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'entry_atr': entry_atr,
                    'score': score
                })

            if candidates:
                candidates.sort(key=lambda x: (x['score'], x['entry_atr']), reverse=True)
                best = candidates[0]
                print(f"[{ts}] Signals: {[ (c['symbol'], round(c['score'],3)) for c in candidates ]}")
                open_trade(sim_account, best['symbol'], best['type'], best['entry_price'],
                           best['sl'], best['tp'], best['entry_atr'], ts)



# --- 6. BACKTEST ORCHESTRATION ---
def run_backtest(symbol_dfs):
    """
    Initializes the simulation account, runs the main backtest loop,
    and returns the final account state.
    """
    # Initialize the simulation account state
    sim_account = {
        'balance': CONFIG['INITIAL_BALANCE'],
        'equity': CONFIG['INITIAL_BALANCE'],
        'open_trades': [],
        'closed_trades': [],
        'equity_curve': [] # Will be populated by close_trade
    }

    # Find the very first timestamp across all dataframes to start the curve
    try:
        first_ts = min(df.index[0] for df in symbol_dfs.values() if not df.empty)
        sim_account['equity_curve'].append({'time': first_ts, 'equity': sim_account['equity']})
    except ValueError: # handles case where all dataframes are empty
        sim_account['equity_curve'].append({'time': None, 'equity': sim_account['equity']})


    # Call the main loop function which modifies sim_account in place
    run_backtest_single_trade_at_time(symbol_dfs, sim_account, CONFIG)

    return sim_account

# --- 7. REPORTING ---
def generate_report(sim_account):
    """Calculates and prints performance metrics from the backtest."""
    history_df = pd.DataFrame(sim_account['closed_trades'])
    if history_df.empty:
        logger.info("No trades were executed during the backtest.")
        return

    logger.info("\n--- Backtest Performance Report ---")
    
    initial_balance = CONFIG['INITIAL_BALANCE']
    # --- BEST PRACTICE: Use the final equity for the report ---
    final_balance = sim_account['equity'] 
    total_pnl = final_balance - initial_balance
    
    total_trades = len(history_df)
    wins = history_df[history_df['pnl'] > 0]
    losses = history_df[history_df['pnl'] <= 0]
    
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    
    avg_win = wins['pnl'].mean()
    avg_loss = losses['pnl'].mean()
    
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
    
    # Max Drawdown
    equity_curve = pd.DataFrame(sim_account['equity_curve']).set_index('time')
    equity_curve['high_water_mark'] = equity_curve['equity'].cummax()
    equity_curve['drawdown'] = equity_curve['high_water_mark'] - equity_curve['equity']
    equity_curve['drawdown_pct'] = equity_curve['drawdown'] / equity_curve['high_water_mark'] * 100
    max_drawdown_pct = equity_curve['drawdown_pct'].max()
    
    print(f"Period: {CONFIG['BACKTEST_START_DATE']} to {CONFIG['BACKTEST_END_DATE']}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:   ${final_balance:,.2f}")
    print(f"Total PnL:       ${total_pnl:,.2f} ({total_pnl/initial_balance*100:.2f}%)")
    print("-" * 35)
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"Avg Win:         ${avg_win:,.2f}")
    print(f"Avg Loss:        ${avg_loss:,.2f}")
    print(f"Max Drawdown:    {max_drawdown_pct:.2f}%")
    print("-" * 35)

    # Plot Equity Curve
    plt.figure(figsize=(12, 6))
    equity_curve['equity'].plot(title='Equity Curve', grid=True)
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.show()


# --- 8. SCRIPT EXECUTION ---
if __name__ == "__main__":
    backtest_data = setup_and_fetch_data(
        CONFIG["SYMBOLS"], 
        CONFIG["BACKTEST_START_DATE"], 
        CONFIG["BACKTEST_END_DATE"]
    )
    
    if backtest_data:
        final_account_state = run_backtest(backtest_data)
        generate_report(final_account_state)
    else:
        logger.error("Could not run backtest due to data fetching issues.")