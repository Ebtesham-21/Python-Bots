import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone
import pytz
import time
import traceback
import uuid # For unique trade IDs

# --- Configuration (Same as your original, with additions for backtesting) ---
SYMBOLS_TO_TRADE = ["BTCUSD", "ETHUSD",  "BTCXAU" ]
# SYMBOLS_TO_TRADE = ["XAUUSD", "EURUSD"] # Exness typically uses XAUUSDm for micro, XAUUSD for standard
# SYMBOLS_TO_TRADE = ["BTCUSD"]

TIMEFRAME_STR = "M30"
MAGIC_NUMBER = 123456

RISK_PERCENT_PER_TRADE = 1.0
DAILY_CAPITAL_ALLOCATION_PERCENT =  2.0 # Set to 100% for simpler backtesting if not testing this feature specifically

# Strategy Parameters
EMA_SHORT_LEN = 5
EMA_MEDIUM_LEN = 8
EMA_LONG_LEN = 13
RISK_REWARD_RATIO = 1.5
SL_BUFFER_PIPS = 5
USE_ALTERNATIVE_EXIT = True

# --- Backtesting Specific Configuration ---
BACKTEST_START_DATE = datetime(2024, 5, 1, tzinfo=timezone.utc)
BACKTEST_END_DATE = datetime(2025, 5, 18, tzinfo=timezone.utc) # Ensure this covers enough data
INITIAL_BALANCE = 200.0
SPREAD_PIPS = 0 # Average spread in pips for simulation (e.g., 2 pips for EURUSD)

# --- MT5 Timeframe Mapping ---
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
}
MT5_TIMEFRAME = TIMEFRAME_MAP.get(TIMEFRAME_STR)

# --- Global Variables for Backtesting ---
sim_balance = INITIAL_BALANCE
sim_equity = INITIAL_BALANCE
sim_open_positions = [] # List of dicts for open trades
sim_trade_history = [] # List of dicts for closed trades
SYMBOLS_INFO = {} # Store symbol_info

# Daily capital allocation (simulated)
daily_allocated_capital_for_trading = 0.0
current_trading_day_date_sim = None


# --- MT5 Connection (Used for data download and symbol info) ---
def initialize_mt5_for_backtest():
    if not mt5.initialize():
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    print("MT5 Initialized for data download.")
    return True

def shutdown_mt5_after_backtest():
    mt5.shutdown()
    print("MetaTrader 5 Shutdown after data operations.")

# --- Data and Indicator Functions (Slightly Adapted) ---
def get_historical_data_for_backtest(symbol, timeframe_mt5, start_date, end_date):
    print(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
    rates = mt5.copy_rates_range(symbol, timeframe_mt5, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"No data returned for {symbol} in the given range.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC') # MT5 times are UTC
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    print(f"Fetched {len(df)} bars for {symbol}.")
    return df

def calculate_indicators(df): # Same as original
    if df.empty or len(df) < EMA_LONG_LEN:
        return pd.DataFrame()
    df.ta.ema(length=EMA_SHORT_LEN, append=True, col_names=(f'EMA_{EMA_SHORT_LEN}',))
    df.ta.ema(length=EMA_MEDIUM_LEN, append=True, col_names=(f'EMA_{EMA_MEDIUM_LEN}',))
    df.ta.ema(length=EMA_LONG_LEN, append=True, col_names=(f'EMA_{EMA_LONG_LEN}',))
    df.dropna(inplace=True)
    return df

# --- Position Sizing (Same as original, ensure symbol_info is available) ---
def calculate_lot_size(balance_for_risk_calc, risk_percent, sl_distance_price_units, symbol_info):
    if sl_distance_price_units <= 1e-9: return 0.0
    if balance_for_risk_calc <= 0 :
        # print(f"Warning: Balance for risk calculation ({balance_for_risk_calc:.2f}) is zero or negative. Cannot calculate lot size.")
        return 0.0

    risk_amount_account_currency = balance_for_risk_calc * (risk_percent / 100.0)

    if symbol_info.trade_tick_size == 0:
        # print(f"Warning: symbol_info.trade_tick_size for {symbol_info.name} is zero. Cannot calculate value per unit.")
        return symbol_info.volume_min if risk_amount_account_currency > 0 else 0.0

    value_per_full_price_unit_per_lot = symbol_info.trade_tick_value / symbol_info.trade_tick_size
    if abs(value_per_full_price_unit_per_lot) < 1e-9:
        # print(f"Warning: value_per_full_price_unit_per_lot for {symbol_info.name} is near zero. Using min volume.")
        return symbol_info.volume_min if risk_amount_account_currency > 0 else 0.0

    sl_value_per_lot = sl_distance_price_units * value_per_full_price_unit_per_lot
    if abs(sl_value_per_lot) < 1e-9:
        # print(f"Warning: sl_value_per_lot for {symbol_info.name} is near zero. Cannot calculate lot size effectively.")
        return 0.0

    raw_lot_size = risk_amount_account_currency / sl_value_per_lot

    if symbol_info.volume_step == 0: # Should not happen with proper symbol_info
        rounded_lot_size = round(raw_lot_size, 8) # Default to 8 decimal places if step is 0
    else:
        rounded_lot_size = (raw_lot_size // symbol_info.volume_step) * symbol_info.volume_step

    final_lot_size = max(symbol_info.volume_min, rounded_lot_size)
    final_lot_size = min(symbol_info.volume_max, final_lot_size)

    if final_lot_size < symbol_info.volume_min:
        return 0.0
    return final_lot_size

# --- Simulated Trade Execution ---
def sim_place_trade(symbol, trade_type, lot_size, entry_price, sl_price, tp_price, current_time, symbol_info):
    global sim_open_positions

    # Simulate spread
    spread_amount = SPREAD_PIPS * symbol_info.point
    actual_entry_price = entry_price
    if trade_type == mt5.ORDER_TYPE_BUY:
        actual_entry_price += spread_amount / 2 # Buy at Ask (higher)
    elif trade_type == mt5.ORDER_TYPE_SELL:
        actual_entry_price -= spread_amount / 2 # Sell at Bid (lower)

    position_id = str(uuid.uuid4()) # Unique ID for the trade
    position = {
        "ticket": position_id,
        "symbol": symbol,
        "type": trade_type, # mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
        "volume": lot_size,
        "price_open": actual_entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "time_open": current_time,
        "magic": MAGIC_NUMBER,
        "comment": "EMA_Strategy_Backtest"
    }
    sim_open_positions.append(position)
    trade_type_str = "BUY" if trade_type == mt5.ORDER_TYPE_BUY else "SELL"
    print(f"{current_time}: Placed {trade_type_str} {symbol} @ {actual_entry_price:.{symbol_info.digits}f}, SL={sl_price:.{symbol_info.digits}f}, TP={tp_price:.{symbol_info.digits}f}, Lots={lot_size}")

def sim_close_trade(position_ticket, close_price, close_time, reason="Manual Close"):
    global sim_balance, sim_equity, sim_open_positions, sim_trade_history

    position_to_close = None
    for i, pos in enumerate(sim_open_positions):
        if pos["ticket"] == position_ticket:
            position_to_close = sim_open_positions.pop(i)
            break

    if not position_to_close:
        print(f"Error: Could not find position {position_ticket} to close.")
        return

    symbol_info = SYMBOLS_INFO[position_to_close["symbol"]]

    # Simulate spread on close
    spread_amount = SPREAD_PIPS * symbol_info.point
    actual_close_price = close_price
    if position_to_close["type"] == mt5.ORDER_TYPE_BUY: # Closing a buy means selling
        actual_close_price -= spread_amount / 2
    elif position_to_close["type"] == mt5.ORDER_TYPE_SELL: # Closing a sell means buying
        actual_close_price += spread_amount / 2

    # Calculate P&L
    price_diff = 0
    if position_to_close["type"] == mt5.ORDER_TYPE_BUY:
        price_diff = actual_close_price - position_to_close["price_open"]
    elif position_to_close["type"] == mt5.ORDER_TYPE_SELL:
        price_diff = position_to_close["price_open"] - actual_close_price

    pnl_per_unit_price_per_lot = symbol_info.trade_tick_value / symbol_info.trade_tick_size
    pnl = price_diff * pnl_per_unit_price_per_lot * position_to_close["volume"]

    sim_balance += pnl
    sim_equity = sim_balance # In this simple model, equity = balance when no open trades, or updated per tick

    closed_trade_info = {
        **position_to_close,
        "price_close": actual_close_price,
        "time_close": close_time,
        "pnl": pnl,
        "reason": reason
    }
    sim_trade_history.append(closed_trade_info)
    print(f"{close_time}: Closed {position_to_close['symbol']} Pos {position_to_close['ticket']} @ {actual_close_price:.{symbol_info.digits}f}. P&L: {pnl:.2f}. Reason: {reason}. Balance: {sim_balance:.2f}")

# --- Check SL/TP for Open Positions ---
def check_sl_tp_hits(symbol, current_bar_high, current_bar_low, current_bar_time):
    global sim_open_positions
    positions_to_check = [p for p in sim_open_positions if p["symbol"] == symbol] # Check only relevant symbol

    for pos in list(positions_to_check): # Iterate over a copy in case of modification
        if pos["type"] == mt5.ORDER_TYPE_BUY:
            # Check SL
            if pos["sl"] is not None and current_bar_low <= pos["sl"]:
                sim_close_trade(pos["ticket"], pos["sl"], current_bar_time, reason="SL Hit")
                continue # Move to next position if closed
            # Check TP
            if pos["tp"] is not None and current_bar_high >= pos["tp"]:
                sim_close_trade(pos["ticket"], pos["tp"], current_bar_time, reason="TP Hit")
        elif pos["type"] == mt5.ORDER_TYPE_SELL:
            # Check SL
            if pos["sl"] is not None and current_bar_high >= pos["sl"]:
                sim_close_trade(pos["ticket"], pos["sl"], current_bar_time, reason="SL Hit")
                continue
            # Check TP
            if pos["tp"] is not None and current_bar_low <= pos["tp"]:
                sim_close_trade(pos["ticket"], pos["tp"], current_bar_time, reason="TP Hit")

# --- Main Trading Logic (Adapted for Backtesting) ---
def backtest_check_and_trade(symbol, historical_df_slice, current_bar_data, next_bar_open, symbol_info, capital_for_day_risk_calc):
    # current_bar_data is the bar ON WHICH THE SIGNAL IS GENERATED (e.g. df.iloc[-2] in live)
    # next_bar_open is the open of the bar AFTER the signal bar, used for entry.
    # historical_df_slice includes up to current_bar_data for indicator calculation.

    ema_s_col = f'EMA_{EMA_SHORT_LEN}'
    ema_m_col = f'EMA_{EMA_MEDIUM_LEN}'
    ema_l_col = f'EMA_{EMA_LONG_LEN}'
    sl_buffer_actual_price = SL_BUFFER_PIPS * symbol_info.point

    df_with_indicators = calculate_indicators(historical_df_slice.copy()) # Calculate on the slice
    if df_with_indicators.empty or len(df_with_indicators) < 3: # Need at least 3 bars for prev_bar and prev_bar_2 logic
        return

    if len(df_with_indicators) < 2: # Need at least current signal bar and one before it
        # print(f"{current_bar_data.name}: Not enough indicator data for {symbol}")
        return

    signal_bar = df_with_indicators.iloc[-1] # This is current_bar_data with indicators
    bar_before_signal = df_with_indicators.iloc[-2]


    # Check for existing positions for this symbol (managed by this bot)
    bot_position = None
    for pos in sim_open_positions:
        if pos["symbol"] == symbol and pos["magic"] == MAGIC_NUMBER:
            bot_position = pos
            break

    current_time = current_bar_data.name # Timestamp of the signal bar

    if bot_position:
        if USE_ALTERNATIVE_EXIT:
            exit_signal = False
            if bot_position["type"] == mt5.ORDER_TYPE_BUY: # Is a BUY position
                # If EMA short crosses BELOW medium on the signal_bar (compared to bar_before_signal)
                if signal_bar[ema_s_col] < signal_bar[ema_m_col] and bar_before_signal[ema_s_col] >= bar_before_signal[ema_m_col]:
                    # print(f"{current_time}: Alt exit for BUY on {symbol}: EMA Short crossed below Medium.")
                    exit_signal = True
            elif bot_position["type"] == mt5.ORDER_TYPE_SELL: # Is a SELL position
                # If EMA short crosses ABOVE medium on the signal_bar
                if signal_bar[ema_s_col] > signal_bar[ema_m_col] and bar_before_signal[ema_s_col] <= bar_before_signal[ema_m_col]:
                    # print(f"{current_time}: Alt exit for SELL on {symbol}: EMA Short crossed above Medium.")
                    exit_signal = True

            if exit_signal:
                # Exit at the open of the *next* bar
                sim_close_trade(bot_position["ticket"], next_bar_open, current_time + pd.Timedelta(minutes=MT5_TIMEFRAME), reason="Alt Exit EMA Cross") # Time of next bar open
                return # Don't check for new entries if we just exited
        return # If position exists and no alt exit, do nothing more on this bar

    # No open position for this symbol by this bot, check for entry
    if bot_position is None:
        trade_type_to_place = None
        stop_loss_price = None
        take_profit_price = None
        risk_price_diff_abs = 0

        entry_price_for_calc = signal_bar['Close'] # Use signal bar's close for SL/TP calculation base
                                                  # Actual entry will be next_bar_open

        # Long Entry: EMA short crosses ABOVE medium, AND price is above all 3 EMAs
        long_ema_cross = signal_bar[ema_s_col] > signal_bar[ema_m_col] and bar_before_signal[ema_s_col] <= bar_before_signal[ema_m_col]
        long_price_above_emas = signal_bar['Close'] > signal_bar[ema_s_col] and \
                                signal_bar['Close'] > signal_bar[ema_m_col] and \
                                signal_bar['Close'] > signal_bar[ema_l_col]

        if long_ema_cross and long_price_above_emas:
            trade_type_to_place = mt5.ORDER_TYPE_BUY
            sl_candidate = signal_bar['Low'] - sl_buffer_actual_price

            # Ensure SL is below the assumed entry price (next_bar_open)
            if sl_candidate >= next_bar_open:
                # print(f"{current_time}: Long SL candidate {sl_candidate:.{symbol_info.digits}f} for {symbol} too high vs next open {next_bar_open:.{symbol_info.digits}f}. Adjusting.")
                sl_candidate = next_bar_open - sl_buffer_actual_price - (symbol_info.point * 5)
                if sl_candidate >= next_bar_open : # Still bad
                    # print(f"{current_time}: Adjusted SL still too high for {symbol}. Skipping.")
                    return

            risk_price_diff_abs = next_bar_open - sl_candidate # Risk based on actual entry
            if risk_price_diff_abs > symbol_info.point * 2: # Min risk
                stop_loss_price = sl_candidate
                take_profit_price = next_bar_open + (risk_price_diff_abs * RISK_REWARD_RATIO)
            else:
                # print(f"{current_time}: Risk too small for BUY on {symbol}. Diff: {risk_price_diff_abs:.{symbol_info.digits}f}")
                trade_type_to_place = None # Invalidate trade

        # Short Entry: EMA short crosses BELOW medium, AND price is below all 3 EMAs
        short_ema_cross = signal_bar[ema_s_col] < signal_bar[ema_m_col] and bar_before_signal[ema_s_col] >= bar_before_signal[ema_m_col]
        short_price_below_emas = signal_bar['Close'] < signal_bar[ema_s_col] and \
                                 signal_bar['Close'] < signal_bar[ema_m_col] and \
                                 signal_bar['Close'] < signal_bar[ema_l_col]

        if trade_type_to_place is None and short_ema_cross and short_price_below_emas:
            trade_type_to_place = mt5.ORDER_TYPE_SELL
            sl_candidate = signal_bar['High'] + sl_buffer_actual_price

            if sl_candidate <= next_bar_open:
                # print(f"{current_time}: Short SL candidate {sl_candidate:.{symbol_info.digits}f} for {symbol} too low vs next open {next_bar_open:.{symbol_info.digits}f}. Adjusting.")
                sl_candidate = next_bar_open + sl_buffer_actual_price + (symbol_info.point * 5)
                if sl_candidate <= next_bar_open: # Still bad
                    # print(f"{current_time}: Adjusted SL still too low for {symbol}. Skipping.")
                    return

            risk_price_diff_abs = sl_candidate - next_bar_open
            if risk_price_diff_abs > symbol_info.point * 2: # Min risk
                stop_loss_price = sl_candidate
                take_profit_price = next_bar_open - (risk_price_diff_abs * RISK_REWARD_RATIO)
            else:
                # print(f"{current_time}: Risk too small for SELL on {symbol}. Diff: {risk_price_diff_abs:.{symbol_info.digits}f}")
                trade_type_to_place = None # Invalidate trade

        if trade_type_to_place is not None and stop_loss_price is not None and take_profit_price is not None:
            if capital_for_day_risk_calc <= 0:
                # print(f"{current_time}: Daily allocated capital ({capital_for_day_risk_calc:.2f}) is zero or less for {symbol}. Cannot place trade.")
                return

            calculated_lot = calculate_lot_size(capital_for_day_risk_calc, RISK_PERCENT_PER_TRADE, risk_price_diff_abs, symbol_info)

            if calculated_lot >= symbol_info.volume_min:
                sim_place_trade(symbol, trade_type_to_place, calculated_lot,
                                next_bar_open, # Entry price
                                stop_loss_price, take_profit_price,
                                current_time + pd.Timedelta(minutes=MT5_TIMEFRAME), # Trade opens at start of next bar
                                symbol_info)
            else:
                pass
                # print(f"{current_time}: Lot size {calculated_lot:.{symbol_info.digits+2}f} too small for {symbol}. Min: {symbol_info.volume_min}. Skipping.")


# --- Main Backtesting Loop ---
if __name__ == "__main__":
    if not MT5_TIMEFRAME:
        print(f"Error: Invalid timeframe string '{TIMEFRAME_STR}' in config.")
        exit()

    if not initialize_mt5_for_backtest():
        exit()

    all_historical_data = {}
    min_data_length = 0

    for sym in SYMBOLS_TO_TRADE:
        info = mt5.symbol_info(sym)
        if info is None:
            print(f"Symbol {sym} not found by MT5. Check spelling or broker availability. Exiting.")
            shutdown_mt5_after_backtest()
            exit()
        if not info.visible:
            print(f"Symbol {sym} not visible, enabling...")
            if not mt5.symbol_select(sym, True):
                print(f"Failed to enable {sym}. Error: {mt5.last_error()}. Exiting.")
                shutdown_mt5_after_backtest()
                exit()
            time.sleep(0.5) # Give MT5 a moment
        SYMBOLS_INFO[sym] = info

        df = get_historical_data_for_backtest(sym, MT5_TIMEFRAME, BACKTEST_START_DATE, BACKTEST_END_DATE)
        if df.empty:
            print(f"No historical data for {sym}. Exiting.")
            shutdown_mt5_after_backtest()
            exit()
        all_historical_data[sym] = df
        if min_data_length == 0 or len(df) < min_data_length:
            min_data_length = len(df)

    shutdown_mt5_after_backtest() # MT5 no longer needed after data download

    if not all_historical_data:
        print("No data loaded for any symbol. Exiting.")
        exit()

    common_index = None
    for sym_df in all_historical_data.values():
        if common_index is None:
            common_index = sym_df.index
        else:
            common_index = common_index.intersection(sym_df.index)

    if common_index is None or common_index.empty:
        print("No common time index found across symbols. Cannot proceed with synchronized backtest.")
        exit()

    print(f"Found {len(common_index)} common time steps for backtesting.")

    # Filter all dataframes to common index
    for sym in SYMBOLS_TO_TRADE:
        all_historical_data[sym] = all_historical_data[sym].loc[common_index].sort_index()

    print("\n--- Starting Backtest ---")
    print(f"Period: {BACKTEST_START_DATE.strftime('%Y-%m-%d')} to {BACKTEST_END_DATE.strftime('%Y-%m-%d')}")
    print(f"Initial Balance: {INITIAL_BALANCE:.2f}")
    print(f"Symbols: {', '.join(SYMBOLS_TO_TRADE)}")
    print(f"Timeframe: {TIMEFRAME_STR}")

    required_bars_for_signal = EMA_LONG_LEN + 5

    for i in range(required_bars_for_signal, len(common_index)):
        current_signal_bar_time = common_index[i-1] # Signal is based on close of i-1 bar
        next_bar_open_time = common_index[i]      # Trade opens at open of i bar

        today_utc_date_sim = current_signal_bar_time.date()
        if current_trading_day_date_sim is None or current_trading_day_date_sim != today_utc_date_sim:
            current_trading_day_date_sim = today_utc_date_sim
            daily_allocated_capital_for_trading = sim_balance * (DAILY_CAPITAL_ALLOCATION_PERCENT / 100.0)
            # print(f"\n--- NEW TRADING DAY (SIM): {current_trading_day_date_sim.strftime('%Y-%m-%d')} ---")
            # print(f"Simulated Account Balance: {sim_balance:.2f}")
            # print(f"Allocated Capital for Today's Trading ({DAILY_CAPITAL_ALLOCATION_PERCENT}%): {daily_allocated_capital_for_trading:.2f}\n")

        if daily_allocated_capital_for_trading <= 0 and DAILY_CAPITAL_ALLOCATION_PERCENT > 0 :
            # print(f"{current_signal_bar_time}: Daily capital depleted or negative. No new trades for today.")
            for sym in SYMBOLS_TO_TRADE:
                symbol_info = SYMBOLS_INFO[sym]
                current_bar_high_for_sltp = all_historical_data[sym].loc[next_bar_open_time]['High']
                current_bar_low_for_sltp = all_historical_data[sym].loc[next_bar_open_time]['Low']
                check_sl_tp_hits(sym, current_bar_high_for_sltp, current_bar_low_for_sltp, next_bar_open_time)
            continue


        for sym in SYMBOLS_TO_TRADE:
            symbol_info = SYMBOLS_INFO[sym]
            historical_df = all_historical_data[sym]

            current_bar_high_for_sltp = historical_df.loc[next_bar_open_time]['High']
            current_bar_low_for_sltp = historical_df.loc[next_bar_open_time]['Low']

            check_sl_tp_hits(sym, current_bar_high_for_sltp, current_bar_low_for_sltp, next_bar_open_time)

            signal_bar_iloc_pos = historical_df.index.get_loc(current_signal_bar_time)
            start_iloc_pos = max(0, signal_bar_iloc_pos - required_bars_for_signal + 1)

            df_slice_for_indicators = historical_df.iloc[start_iloc_pos : signal_bar_iloc_pos + 1]
            signal_generating_bar_data = historical_df.loc[current_signal_bar_time]
            entry_price_bar_open = historical_df.loc[next_bar_open_time]['Open']

            backtest_check_and_trade(sym,
                                     df_slice_for_indicators,
                                     signal_generating_bar_data,
                                     entry_price_bar_open,
                                     symbol_info,
                                     daily_allocated_capital_for_trading)

            current_equity_calc = sim_balance
            for pos in sim_open_positions:
                if pos["symbol"] == sym:
                    current_close_price = historical_df.loc[next_bar_open_time]['Close']
                    price_diff_ue = 0
                    if pos["type"] == mt5.ORDER_TYPE_BUY:
                        price_diff_ue = current_close_price - pos["price_open"]
                    elif pos["type"] == mt5.ORDER_TYPE_SELL:
                        price_diff_ue = pos["price_open"] - current_close_price

                    pnl_per_unit_price_per_lot_ue = symbol_info.trade_tick_value / symbol_info.trade_tick_size
                    unrealized_pnl = price_diff_ue * pnl_per_unit_price_per_lot_ue * pos["volume"]
                    current_equity_calc += unrealized_pnl
            sim_equity = current_equity_calc

    if sim_open_positions:
        print("\n--- Closing remaining open positions at end of backtest ---")
        last_bar_time = common_index[-1]
        for sym_open in list(sim_open_positions): # Iterate copy
            last_close_price = all_historical_data[sym_open["symbol"]].iloc[-1]['Close']
            sim_close_trade(sym_open["ticket"], last_close_price, last_bar_time, reason="End of Backtest")

    print("\n--- Backtest Finished ---")
    print(f"Final Balance: {sim_balance:.2f}")
    print(f"Total Trades: {len(sim_trade_history)}")

    if len(sim_trade_history) > 0:
        pnl_values = [trade['pnl'] for trade in sim_trade_history]
        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = sum(p for p in pnl_values if p < 0)
        net_profit = sum(pnl_values)

        winning_trades = sum(1 for p in pnl_values if p > 0)
        losing_trades = sum(1 for p in pnl_values if p < 0)
        win_rate = (winning_trades / len(sim_trade_history) * 100) if len(sim_trade_history) > 0 else 0

        print(f"Net Profit: {net_profit:.2f}")
        print(f"Gross Profit: {gross_profit:.2f}")
        print(f"Gross Loss: {gross_loss:.2f}")
        print(f"Profit Factor: {(gross_profit / abs(gross_loss)) if gross_loss != 0 else 'inf'}")
        print(f"Win Rate: {win_rate:.2f}% ({winning_trades}/{len(sim_trade_history)})")

        # Optional: Save trade history to CSV
        # trade_history_df = pd.DataFrame(sim_trade_history)
        # trade_history_df.to_csv("backtest_trade_history.csv", index=False)
        # print("Trade history saved to backtest_trade_history.csv")

    else:
        print("No trades were executed during the backtest.")