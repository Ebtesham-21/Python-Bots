import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# --- Global MT5 Connection Variables ---
mt5_connection_active = False
ACCOUNT_CURRENCY = "USD" # Default, will be updated

# --- Your MT5 Connection Functions (Provided by User) ---
def initialize_mt5_connection_bt():
    global mt5_connection_active, ACCOUNT_CURRENCY
    if mt5_connection_active: return True
    if not mt5.initialize():
        print(f"Initialize failed, error code = {mt5.last_error()}")
        return False
    print("MetaTrader 5 Initialized Successfully")
    mt5_connection_active = True
    terminal_info = mt5.terminal_info()
    if terminal_info:
        if hasattr(terminal_info, 'currency'):
             ACCOUNT_CURRENCY = terminal_info.currency
             print(f"Account Currency detected: {ACCOUNT_CURRENCY}")
        else:
             print(f"Could not detect account currency from terminal_info. Assuming: {ACCOUNT_CURRENCY}")
    else:
        print(f"Could not get terminal info. Assuming Account Currency: {ACCOUNT_CURRENCY}")
    return True

def shutdown_mt5_bt():
    global mt5_connection_active
    if mt5_connection_active:
        mt5.shutdown()
        mt5_connection_active = False
        print("MetaTrader 5 Shutdown")

# --- Configuration ---
SYMBOL = "XAUUSD" # Gold
TIMEFRAME_DAILY = mt5.TIMEFRAME_D1
TIMEFRAME_H4 = mt5.TIMEFRAME_H4
TIMEFRAME_ENTRY = mt5.TIMEFRAME_H1 # Entry timeframe (e.g., H1)

# Strategy Parameters
SMA_FAST_DAILY = 50
SMA_SLOW_DAILY = 100 # Using 100 instead of 200 for more responsiveness on daily gold
SMA_FAST_H4 = 20
SMA_SLOW_H4 = 50
SMA_ENTRY_PULLBACK = 20 # SMA on entry timeframe for pullback reference

FIB_DISCOUNT_LEVEL = 0.50 # Enter only if pullback is below 50% Fib (for BUYs) / above 50% (for SELLs)
IMPULSE_LOOKBACK = 30 # Bars to look back for identifying impulse move for Fib
MIN_IMPULSE_SIZE_PIPS = 150 # Minimum size of an impulse move in pips for it to be considered

STOP_LOSS_PIPS_OFFSET = 50 # Pips below the pullback low (BUY) or above pullback high (SELL)
TAKE_PROFIT_RR = 2.0 # Risk:Reward Ratio for Take Profit (e.g., 2 means TP is 2x SL distance)

# Backtest Period
START_DATE = datetime(2025, 5, 1)
END_DATE = datetime(2025, 5, 21) # Adjusted to give some recent data too

# Session Filter Configuration
TRADING_SESSIONS = {
    'london':  {'start': 7,  'end': 15},  # 07:00–15:59 UTC (inclusive of hour 15)
    'newyork': {'start': 12, 'end': 21},  # 12:00–21:59 UTC (inclusive of hour 21)
}

# --- Helper Functions ---
def get_historical_data(symbol, timeframe, start_date, end_date):
    if not mt5_connection_active:
        if not initialize_mt5_connection_bt():
            return None
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol} on {timeframe_to_string(timeframe)} from {start_date} to {end_date}. Error: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s') # MT5 time is UTC
    df.set_index('time', inplace=True)
    return df

def timeframe_to_string(tf_int):
    map_tf = {
        mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_H4: "H4", mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M5: "M5"
    }
    return map_tf.get(tf_int, str(tf_int))

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

def get_symbol_info(symbol):
    if not mt5_connection_active: 
        if not initialize_mt5_connection_bt():
            return None
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Symbol {symbol} not found, can not call symbol_info: {mt5.last_error()}")
        return None
    return info

def is_within_trading_session(current_dt, sessions):
    """
    Checks if the current datetime (UTC) falls within any of the defined trading sessions.
    Args:
        current_dt (pd.Timestamp): The current timestamp (assumed to be UTC).
        sessions (dict): A dictionary defining session start and end hours (UTC).
                         Example: {'london': {'start': 7, 'end': 15}, ...}
    Returns:
        bool: True if within a trading session, False otherwise.
    """
    current_hour_utc = current_dt.hour # .hour from pd.Timestamp gives the hour
    for session_name, times in sessions.items():
        # Check if the current hour is within the session's start and end (inclusive)
        if times['start'] <= current_hour_utc <= times['end']:
            return True
    return False

# --- Main Backtesting Logic ---
def run_backtest():
    print(f"Starting backtest for {SYMBOL} from {START_DATE} to {END_DATE}")
    if not initialize_mt5_connection_bt():
        return

    symbol_info = get_symbol_info(SYMBOL)
    if not symbol_info:
        return
    point = symbol_info.point
    digits = symbol_info.digits
    contract_size = symbol_info.trade_contract_size 

    daily_data_start = START_DATE - timedelta(days=SMA_SLOW_DAILY + IMPULSE_LOOKBACK + 10) 
    h4_data_start = START_DATE - timedelta(days=(SMA_SLOW_H4//6) + (IMPULSE_LOOKBACK//24) + 10) 
    entry_data_start = START_DATE - timedelta(days=(IMPULSE_LOOKBACK//24) + 10) 

    daily_data = get_historical_data(SYMBOL, TIMEFRAME_DAILY, daily_data_start, END_DATE)
    h4_data = get_historical_data(SYMBOL, TIMEFRAME_H4, h4_data_start, END_DATE)
    entry_data = get_historical_data(SYMBOL, TIMEFRAME_ENTRY, entry_data_start, END_DATE)

    if daily_data is None or h4_data is None or entry_data is None:
        print("Failed to fetch all required historical data.")
        return
    if len(daily_data) < SMA_SLOW_DAILY or len(h4_data) < SMA_SLOW_H4 or len(entry_data) < IMPULSE_LOOKBACK:
        print("Insufficient historical data for indicator calculation or lookback.")
        return

    daily_data['sma_fast'] = calculate_sma(daily_data['close'], SMA_FAST_DAILY)
    daily_data['sma_slow'] = calculate_sma(daily_data['close'], SMA_SLOW_DAILY)
    h4_data['sma_fast'] = calculate_sma(h4_data['close'], SMA_FAST_H4)
    h4_data['sma_slow'] = calculate_sma(h4_data['close'], SMA_SLOW_H4)
    entry_data['sma_pullback'] = calculate_sma(entry_data['close'], SMA_ENTRY_PULLBACK)

    trades = []
    open_trade = None
    initial_balance = 200.0 
    balance = initial_balance
    peak_equity = balance
    max_drawdown_value = 0
    max_drawdown_percentage = 0
    equity_at_trade_points = [balance] 

    start_index_entry = max(IMPULSE_LOOKBACK, SMA_ENTRY_PULLBACK) 
    entry_data_for_iteration = entry_data[entry_data.index >= START_DATE]
    if entry_data_for_iteration.empty:
        print(f"No entry data available from {START_DATE} onwards.")
        return

    for i in range(len(entry_data_for_iteration)):
        current_time = entry_data_for_iteration.index[i]
        original_idx = entry_data.index.get_loc(current_time)
        if original_idx < start_index_entry: 
            continue

        current_bar_entry = entry_data.iloc[original_idx]
        prev_bar_entry = entry_data.iloc[original_idx-1]

        # --- Manage Open Trade (SL/TP) ---
        if open_trade:
            if open_trade['direction'] == 'BUY':
                if current_bar_entry['low'] <= open_trade['sl']:
                    profit = (open_trade['sl'] - open_trade['entry_price']) * open_trade['lots'] * contract_size
                    balance += profit
                    trades.append({**open_trade, 'exit_price': open_trade['sl'], 'exit_time': current_time, 'profit': profit, 'status': 'SL', 'balance_after': balance})
                    open_trade = None
                    equity_at_trade_points.append(balance)
                elif current_bar_entry['high'] >= open_trade['tp']:
                    profit = (open_trade['tp'] - open_trade['entry_price']) * open_trade['lots'] * contract_size
                    balance += profit
                    trades.append({**open_trade, 'exit_price': open_trade['tp'], 'exit_time': current_time, 'profit': profit, 'status': 'TP', 'balance_after': balance})
                    open_trade = None
                    equity_at_trade_points.append(balance)
            elif open_trade['direction'] == 'SELL':
                if current_bar_entry['high'] >= open_trade['sl']: # SL for short
                    profit = (open_trade['entry_price'] - open_trade['sl']) * open_trade['lots'] * contract_size
                    balance += profit
                    trades.append({**open_trade, 'exit_price': open_trade['sl'], 'exit_time': current_time, 'profit': profit, 'status': 'SL', 'balance_after': balance})
                    open_trade = None
                    equity_at_trade_points.append(balance)
                elif current_bar_entry['low'] <= open_trade['tp']: # TP for short
                    profit = (open_trade['entry_price'] - open_trade['tp']) * open_trade['lots'] * contract_size
                    balance += profit
                    trades.append({**open_trade, 'exit_price': open_trade['tp'], 'exit_time': current_time, 'profit': profit, 'status': 'TP', 'balance_after': balance})
                    open_trade = None
                    equity_at_trade_points.append(balance)
        
        # If a trade is still open after management, skip to next bar for new entry signals
        if open_trade: 
            continue

        # --- SESSION FILTER ---
        # Check if current bar's time is within defined trading sessions before considering new trades
        if not is_within_trading_session(current_time, TRADING_SESSIONS):
            continue # Skip to the next bar if outside trading hours

        # --- Trade Entry Logic (Only if no trade is open AND within session) ---
        current_daily_data_slice = daily_data[daily_data.index < current_time]
        current_h4_data_slice = h4_data[h4_data.index < current_time]

        if current_daily_data_slice.empty or current_h4_data_slice.empty:
            continue 
        
        current_daily_bar = current_daily_data_slice.iloc[-1:]
        current_h4_bar = current_h4_data_slice.iloc[-1:]

        daily_bullish = (not current_daily_bar.empty and 
                         current_daily_bar['sma_fast'].iloc[-1] > current_daily_bar['sma_slow'].iloc[-1])
        h4_bullish = (not current_h4_bar.empty and
                      current_h4_bar['sma_fast'].iloc[-1] > current_h4_bar['sma_slow'].iloc[-1])
        
        daily_bearish = (not current_daily_bar.empty and 
                         current_daily_bar['sma_fast'].iloc[-1] < current_daily_bar['sma_slow'].iloc[-1])
        h4_bearish = (not current_h4_bar.empty and
                      current_h4_bar['sma_fast'].iloc[-1] < current_h4_bar['sma_slow'].iloc[-1])

        lookback_data_entry = entry_data.iloc[original_idx - IMPULSE_LOOKBACK : original_idx] 
        if len(lookback_data_entry) < 2: continue 

        # --- BUY Logic ---
        if daily_bullish and h4_bullish:
            impulse_swing_low_price_buy = lookback_data_entry['low'].min()
            idx_swing_low_buy = lookback_data_entry['low'].idxmin()
            data_after_swing_low_buy = lookback_data_entry[lookback_data_entry.index > idx_swing_low_buy]
            if not data_after_swing_low_buy.empty:
                impulse_swing_high_price_buy = data_after_swing_low_buy['high'].max()
                idx_swing_high_buy = data_after_swing_low_buy['high'].idxmax()

                if idx_swing_high_buy > idx_swing_low_buy and (entry_data.index[original_idx-1] - idx_swing_high_buy).total_seconds() <= 5 * 3600:
                    impulse_range_buy = impulse_swing_high_price_buy - impulse_swing_low_price_buy
                    if impulse_range_buy >= MIN_IMPULSE_SIZE_PIPS * point:
                        fib_50_buy = impulse_swing_high_price_buy - impulse_range_buy * FIB_DISCOUNT_LEVEL
                        is_pullback_to_discount = prev_bar_entry['low'] <= fib_50_buy
                        
                        current_bar_range_buy = current_bar_entry['high'] - current_bar_entry['low']
                        if current_bar_range_buy == 0: current_bar_range_buy = point
                        
                        is_bullish_candle = (current_bar_entry['close'] > current_bar_entry['open'] and
                                             (current_bar_entry['close'] - current_bar_entry['open']) > 0.3 * current_bar_range_buy and 
                                             current_bar_entry['close'] > (current_bar_entry['low'] + 0.66 * current_bar_range_buy))

                        if is_pullback_to_discount and is_bullish_candle and current_bar_entry['low'] > impulse_swing_low_price_buy:
                            entry_price = current_bar_entry['close']
                            stop_loss_level = min(current_bar_entry['low'], prev_bar_entry['low']) - STOP_LOSS_PIPS_OFFSET * point
                            risk_pips_value = entry_price - stop_loss_level
                            if risk_pips_value > 0:
                                risk_amount_per_trade = balance * 0.02
                                lot_size = (risk_amount_per_trade / (risk_pips_value * contract_size)) if (risk_pips_value * contract_size) > 0 else 0.01
                                lot_size = round(max(0.01, lot_size), 2)
                                take_profit_level = entry_price + (risk_pips_value * TAKE_PROFIT_RR)
                                if lot_size * contract_size * risk_pips_value <= balance * 0.5 : # Ensure trade risk isn't too large fraction of balance
                                    open_trade = {'entry_time': current_time, 'entry_price': entry_price, 'sl': stop_loss_level,
                                                  'tp': take_profit_level, 'lots': lot_size, 'direction': 'BUY', 'symbol': SYMBOL}
                                    if balance > peak_equity: peak_equity = balance


        # --- SELL Logic ---
        elif daily_bearish and h4_bearish:
            impulse_swing_high_price_sell = lookback_data_entry['high'].max()
            idx_swing_high_sell = lookback_data_entry['high'].idxmax()
            data_after_swing_high_sell = lookback_data_entry[lookback_data_entry.index > idx_swing_high_sell]
            if not data_after_swing_high_sell.empty:
                impulse_swing_low_price_sell = data_after_swing_high_sell['low'].min()
                idx_swing_low_sell = data_after_swing_high_sell['low'].idxmin()

                if idx_swing_low_sell > idx_swing_high_sell and (entry_data.index[original_idx-1] - idx_swing_low_sell).total_seconds() <= 5 * 3600:
                    impulse_range_sell = impulse_swing_high_price_sell - impulse_swing_low_price_sell
                    if impulse_range_sell >= MIN_IMPULSE_SIZE_PIPS * point:
                        fib_50_sell = impulse_swing_low_price_sell + impulse_range_sell * FIB_DISCOUNT_LEVEL # Premium for sell
                        is_pullback_to_premium = prev_bar_entry['high'] >= fib_50_sell
                        
                        current_bar_range_sell = current_bar_entry['high'] - current_bar_entry['low']
                        if current_bar_range_sell == 0: current_bar_range_sell = point
                        
                        is_bearish_candle = (current_bar_entry['close'] < current_bar_entry['open'] and
                                             (current_bar_entry['open'] - current_bar_entry['close']) > 0.3 * current_bar_range_sell and
                                             current_bar_entry['close'] < (current_bar_entry['high'] - 0.66 * current_bar_range_sell))

                        if is_pullback_to_premium and is_bearish_candle and current_bar_entry['high'] < impulse_swing_high_price_sell:
                            entry_price = current_bar_entry['close']
                            stop_loss_level = max(current_bar_entry['high'], prev_bar_entry['high']) + STOP_LOSS_PIPS_OFFSET * point
                            risk_pips_value = stop_loss_level - entry_price
                            if risk_pips_value > 0:
                                risk_amount_per_trade = balance * 0.02
                                lot_size = (risk_amount_per_trade / (risk_pips_value * contract_size)) if (risk_pips_value * contract_size) > 0 else 0.01
                                lot_size = round(max(0.01, lot_size), 2)
                                take_profit_level = entry_price - (risk_pips_value * TAKE_PROFIT_RR)
                                if lot_size * contract_size * risk_pips_value <= balance * 0.5 : # Ensure trade risk isn't too large fraction of balance
                                    open_trade = {'entry_time': current_time, 'entry_price': entry_price, 'sl': stop_loss_level,
                                                  'tp': take_profit_level, 'lots': lot_size, 'direction': 'SELL', 'symbol': SYMBOL}
                                    if balance > peak_equity: peak_equity = balance
            
    current_peak = equity_at_trade_points[0] if equity_at_trade_points else initial_balance
    for equity_value in equity_at_trade_points:
        if equity_value > current_peak:
            current_peak = equity_value
        drawdown = current_peak - equity_value
        if drawdown > max_drawdown_value:
            max_drawdown_value = drawdown
        if current_peak > 0 : 
            dd_percent = (drawdown / current_peak) * 100
            if dd_percent > max_drawdown_percentage:
                max_drawdown_percentage = dd_percent
            
    print("\n--- Backtest Finished ---")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Symbol: {SYMBOL}, Entry Timeframe: {timeframe_to_string(TIMEFRAME_ENTRY)}")
    print(f"Trading Sessions Filtered: London (07-15 UTC), New York (12-21 UTC)")
    print(f"Initial Balance: {initial_balance:.2f} {ACCOUNT_CURRENCY}")
    print(f"Final Balance: {balance:.2f} {ACCOUNT_CURRENCY}")
    total_profit_abs = balance - initial_balance
    total_profit_perc = (total_profit_abs / initial_balance) * 100 if initial_balance > 0 else 0
    print(f"Total Net Profit: {total_profit_abs:.2f} {ACCOUNT_CURRENCY} ({total_profit_perc:.2f}%)")
    print(f"Total Trades: {len(trades)}")

    wins = [t for t in trades if t['profit'] > 0]
    losses = [t for t in trades if t['profit'] <= 0] 
    
    print(f"Winning Trades: {len(wins)}")
    print(f"Losing Trades: {len(losses)}")

    if len(trades) > 0:
        win_rate = (len(wins) / len(trades)) * 100
        print(f"Win Rate: {win_rate:.2f}%")
        
        total_gross_profit = sum(t['profit'] for t in wins)
        total_gross_loss = abs(sum(t['profit'] for t in losses)) 

        print(f"Gross Profit: {total_gross_profit:.2f}")
        print(f"Gross Loss: {total_gross_loss:.2f}")

        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')
        print(f"Profit Factor: {profit_factor:.2f}")
    else:
        print("No trades executed.")

    print(f"Max Drawdown: {max_drawdown_value:.2f} {ACCOUNT_CURRENCY} ({max_drawdown_percentage:.2f}%)")

    if equity_at_trade_points and len(equity_at_trade_points) > 1:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12,6))
            plt.plot(equity_at_trade_points) 
            plt.title(f"{SYMBOL} Equity Curve ({timeframe_to_string(TIMEFRAME_ENTRY)}) - Session Filtered")
            plt.xlabel("Trade Number")
            plt.ylabel(f"Equity ({ACCOUNT_CURRENCY})")
            plt.grid(True)
            plt.savefig(f"{SYMBOL}_{timeframe_to_string(TIMEFRAME_ENTRY)}_equity_curve_session_filtered.png") # Save the plot
            print(f"Equity curve saved to {SYMBOL}_{timeframe_to_string(TIMEFRAME_ENTRY)}_equity_curve_session_filtered.png")
            # plt.show() # Comment out if running in a non-GUI environment
        except ImportError:
            print("Matplotlib not installed. Skipping equity curve plot.")
        except Exception as e:
            print(f"Error plotting equity curve: {e}")
    else:
        print("Not enough equity points to plot the curve.")

    shutdown_mt5_bt()

if __name__ == "__main__":
    run_backtest()