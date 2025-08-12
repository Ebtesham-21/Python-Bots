import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
import pytz
import logging
import matplotlib.pyplot as plt

# --- 1. BACKTEST CONFIGURATION ---
CONFIG = {
    # Backtest period
    "BACKTEST_START_DATE": "2020-01-01",
    "BACKTEST_END_DATE": "2025-7-30",
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

# --- 5. SIMULATION & HELPER FUNCTIONS ---

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

def run_backtest(dataframes):
    """Main backtesting loop that simulates the strategy bar-by-bar."""
    
    # Initialize simulated account and state tracking
    sim_account = {
        "balance": CONFIG["INITIAL_BALANCE"],
        "equity": CONFIG["INITIAL_BALANCE"],
        "open_trades": [],
        "trade_history": [],
        "equity_curve": []
    }
    symbol_states = {symbol: reset_daily_state() for symbol in dataframes.keys()}
    
    # Combine all dataframes and sort by time to process chronologically
    all_data_combined = pd.concat(dataframes.values(), keys=dataframes.keys(), names=['symbol', 'time']).sort_index(level='time')

    last_processed_day = None
    
    logger.info("Starting bar-by-bar simulation...")
    # --- The Main Simulation Loop ---
    for (symbol, timestamp), row in all_data_combined.iterrows():
        
        state = symbol_states[symbol]
        df_symbol = dataframes[symbol]
        
        # --- A. Daily Reset ---
        current_day = timestamp.day
        if last_processed_day != current_day:
            for s in symbol_states:
                symbol_states[s] = reset_daily_state()
            last_processed_day = current_day

        # --- B. Update Equity and Manage Open Trades ---
        # Update equity at the start of each bar based on open positions
        current_equity = sim_account['balance']
        pnl = 0
        for trade in sim_account['open_trades']:
             if trade['symbol'] == symbol:
                price_diff = row['close'] - trade['entry_price']
                if trade['type'] == 'SELL':
                    price_diff = -price_diff
                pnl += price_diff * (trade['volume'] / ALL_SYMBOL_PROPERTIES[symbol]['trade_tick_size']) * ALL_SYMBOL_PROPERTIES[symbol]['trade_tick_value']
        sim_account['equity'] = sim_account['balance'] + pnl
        sim_account['equity_curve'].append({'time': timestamp, 'equity': sim_account['equity']})

        # Check for SL/TP/Time exit for open trades
              # --- NEW REPLACEMENT CODE ---
        trades_to_close = []
        for trade in sim_account['open_trades']:
            if trade['symbol'] != symbol:
                continue

            # Check for Primary Exits: SL/TP
            hit_sl = False
            hit_tp = False
            
            if trade['type'] == 'BUY':
                if row['low'] <= trade['sl']: hit_sl = True
                if row['high'] >= trade['tp']: hit_tp = True
            elif trade['type'] == 'SELL':
                if row['high'] >= trade['sl']: hit_sl = True
                if row['low'] <= trade['tp']: hit_tp = True

            # In case both are hit in one bar, assume SL is hit first (conservative)
            if hit_sl:
                trades_to_close.append({'trade': trade, 'exit_price': trade['sl'], 'reason': 'SL'})
            elif hit_tp:
                trades_to_close.append({'trade': trade, 'exit_price': trade['tp'], 'reason': 'TP'})
            
            # --- NEW: Check for Volatility Drop Exit ---
            # This check only runs if SL or TP was not already hit in this bar.
            else: 
                current_atr = row[f'ATRr_{CONFIG["ATR_PERIOD_SL"]}']
                entry_atr = trade['entry_atr']
                volatility_threshold = entry_atr * CONFIG['VOLATILITY_EXIT_FACTOR']
                
                if current_atr < volatility_threshold:
                    trades_to_close.append({
                        'trade': trade, 
                        'exit_price': row['close'], # Exit at market (bar close)
                        'reason': 'Volatility Exit'
                    })
        # --- END OF NEW CODE ---

        # Process closed trades
        for closure in trades_to_close:
            trade = closure['trade']
            pnl = (closure['exit_price'] - trade['entry_price']) * (trade['volume'] / ALL_SYMBOL_PROPERTIES[trade['symbol']]['trade_tick_size']) * ALL_SYMBOL_PROPERTIES[trade['symbol']]['trade_tick_value']
            if trade['type'] == 'SELL':
                pnl = -pnl
            
            sim_account['balance'] += pnl
            trade['pnl'] = pnl
            trade['exit_price'] = closure['exit_price']
            trade['exit_time'] = timestamp
            trade['exit_reason'] = closure['reason']
            sim_account['trade_history'].append(trade)
            sim_account['open_trades'].remove(trade)
            logger.info(f"[{timestamp}] CLOSED {trade['type']} {trade['symbol']}: Entry={trade['entry_price']:.5f}, Exit={closure['exit_price']:.5f}, PnL={pnl:.2f}, Reason={closure['reason']}")

        # --- C. Strategy Logic (identical to live bot) ---
        # Define time windows for the current bar's timestamp
        or_start_time = timestamp.replace(hour=CONFIG["OR_START_HOUR"], minute=CONFIG["OR_START_MIN"], second=0, microsecond=0)
        or_end_time = timestamp.replace(hour=CONFIG["OR_END_HOUR"], minute=CONFIG["OR_END_MIN"], second=0, microsecond=0)
        trade_start_time = timestamp.replace(hour=CONFIG["TRADE_START_HOUR"], minute=CONFIG["TRADE_START_MIN"], second=0, microsecond=0)
        trade_end_time = timestamp.replace(hour=CONFIG["TRADE_END_HOUR"], minute=CONFIG["TRADE_END_MIN"], second=0, microsecond=0)
        
        if state['trade_taken_today']:
            continue
        
        # Get historical slice up to current point to avoid lookahead
        # The `loc` accessor is inclusive, so this is data up to and including the current bar
        df_historical = df_symbol.loc[:timestamp]

        # 1. Awaiting Bias
        if state['daily_bias'] is None and timestamp >= or_end_time:
            or_candles = df_historical.between_time(or_start_time.strftime('%H:%M'), or_end_time.strftime('%H:%M'))
            if not or_candles.empty:
                state['or_high'] = or_candles['high'].max()
                state['or_low'] = or_candles['low'].min()
                
                # Use the previous bar's close for breakout confirmation
                prev_bar = df_historical.iloc[-2]
                if prev_bar['close'] > state['or_high']: state['daily_bias'] = 'BULLISH'
                elif prev_bar['close'] < state['or_low']: state['daily_bias'] = 'BEARISH'

        # 2. Bias Set, Find Zone
        if state['daily_bias'] and not state['zone_defined'] and timestamp >= trade_start_time:
            # Replicating the live logic to find the zone without lookahead
            # We look backwards from the current bar
            breakout_idx = -1
            for i in range(len(df_historical) - 2, 0, -1):
                candle_time = df_historical.index[i]
                if candle_time < or_start_time: break
                
                is_bullish_break = state['daily_bias'] == 'BULLISH' and df_historical['close'][i] > state['or_high']
                is_bearish_break = state['daily_bias'] == 'BEARISH' and df_historical['close'][i] < state['or_low']
                if is_bullish_break or is_bearish_break:
                    breakout_idx = i
                    break

            if breakout_idx > 0:
                impulse_start_candle_idx = -1
                for i in range(breakout_idx, 0, -1):
                    is_down_candle = df_historical['close'][i] < df_historical['open'][i]
                    if state['daily_bias'] == 'BULLISH' and is_down_candle: impulse_start_candle_idx = i; break
                    if state['daily_bias'] == 'BEARISH' and not is_down_candle: impulse_start_candle_idx = i; break
                
                if impulse_start_candle_idx != -1:
                    zone_candle = df_historical.iloc[impulse_start_candle_idx]
                    impulse_run = df_historical.iloc[impulse_start_candle_idx+1 : breakout_idx+2]
                    
                    zone_high, zone_low = zone_candle['high'], zone_candle['low']
                    impulse_peak = impulse_run['high'].max() if state['daily_bias'] == 'BULLISH' else impulse_run['low'].min()
                    
                    atr_strength = df_historical[f'ATRr_{CONFIG["ATR_PERIOD_STRENGTH"]}'].iloc[-1]
                    impulse_distance = abs(impulse_peak - (zone_high if state['daily_bias'] == 'BULLISH' else zone_low))

                    if impulse_distance >= (CONFIG['IMPULSE_STRENGTH_FACTOR'] * atr_strength):
                        state['zone_high'], state['zone_low'], state['zone_defined'] = zone_high, zone_low, True
                        
                        atr_sl = df_historical[f'ATRr_{CONFIG["ATR_PERIOD_SL"]}'].iloc[-1]
                        sl_buffer = CONFIG['SL_BUFFER_ATR_FACTOR'] * atr_sl
                        if state['daily_bias'] == 'BULLISH':
                            state['entry_price'] = zone_high
                            state['stop_loss'] = zone_low - sl_buffer
                            risk_dist = state['entry_price'] - state['stop_loss']
                            potential_tp = impulse_peak
                            rr_ratio = (potential_tp - state['entry_price']) / risk_dist if risk_dist > 0 else 0
                            state['take_profit'] = potential_tp if rr_ratio >= CONFIG['MIN_RR_RATIO'] else state['entry_price'] + (risk_dist * CONFIG['MIN_RR_RATIO'])
                        else: # BEARISH
                            state['entry_price'] = zone_low
                            state['stop_loss'] = zone_high + sl_buffer
                            risk_dist = state['stop_loss'] - state['entry_price']
                            potential_tp = impulse_peak
                            rr_ratio = (state['entry_price'] - potential_tp) / risk_dist if risk_dist > 0 else 0
                            state['take_profit'] = potential_tp if rr_ratio >= CONFIG['MIN_RR_RATIO'] else state['entry_price'] - (risk_dist * CONFIG['MIN_RR_RATIO'])
                        
                        state['order_placed'] = True # In backtest, this means the trade is now armed
                        state['trade_taken_today'] = True

        # 3. Check for Limit Order Fill
        # We check if an order was defined in a *previous* step and if the *current* bar hits it
        # In the run_backtest function...

        # 3. Check for Limit Order Fill
        # We check if an order was defined in a *previous* step and if the *current* bar hits it
        if state['order_placed'] and not state['zone_touched']:
            entry_price = state['entry_price']
            
            is_buy_fill = state['daily_bias'] == 'BULLISH' and row['low'] <= entry_price
            is_sell_fill = state['daily_bias'] == 'BEARISH' and row['high'] >= entry_price
            
            if is_buy_fill or is_sell_fill:
                # --- START OF MODIFICATION ---
                
                # First, calculate the stop loss distance in price points
                sl_dist = abs(state['entry_price'] - state['stop_loss'])
                
                # Call the new gatekeeper function to check risk and get volume
                volume = check_risk_and_get_volume(symbol, sim_account['equity'], sl_dist)
                
                # ONLY proceed if the gatekeeper returned a valid volume (i.e., risk was acceptable)
                if volume > 0:
                    entry_atr_value = row[f'ATRr_{CONFIG["ATR_PERIOD_SL"]}']

                    trade = {
                        'symbol': symbol, 'type': state['daily_bias'], 'volume': volume,
                        'entry_price': entry_price, 'sl': state['stop_loss'], 'tp': state['take_profit'],
                        'entry_time': timestamp,
                        'entry_atr': entry_atr_value,
                    }
                    sim_account['open_trades'].append(trade)
                    state['zone_touched'] = True 
                    logger.info(f"[{timestamp}] FILLED {trade['type']} {trade['symbol']} @ {entry_price:.5f} with {volume} lots (Entry ATR={entry_atr_value:.5f})")

                # --- END OF MODIFICATION ---

    return sim_account


# --- 7. REPORTING ---
def generate_report(sim_account):
    """Calculates and prints performance metrics from the backtest."""
    history_df = pd.DataFrame(sim_account['trade_history'])
    if history_df.empty:
        logger.info("No trades were executed during the backtest.")
        return

    logger.info("\n--- Backtest Performance Report ---")
    
    initial_balance = CONFIG['INITIAL_BALANCE']
    final_balance = sim_account['balance']
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