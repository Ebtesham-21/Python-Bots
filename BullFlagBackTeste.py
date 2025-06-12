import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone
import time
import logging

# --- 1. Logger Setup (As Requested) ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Corrected: Use __name__ for the logger and main block
logger = logging.getLogger(__name__)

# --- 2. Configuration ---

class Config:
    # --- Backtest Settings ---
    RUN_BACKTEST = True  # Set to True for backtesting
    INITIAL_ACCOUNT_BALANCE = 200.0
    BACKTEST_START_DATE = datetime(2024, 7, 1)
    BACKTEST_END_DATE = datetime(2025, 5, 31)

    # --- Trading Strategy Settings ---
    SYMBOLS_TO_TRADE =    ["EURUSD", "AUDUSD", "USDCHF", "USDCAD",
                       "GBPJPY",  "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                        "USOIL", "CADJPY", "XAGUSD", "XPTUSD", "UKOIL",
                        "BTCUSD", "BTCJPY",  "ETHUSD"   ]
    TIMEFRAME_EXECUTION = mt5.TIMEFRAME_M15
    TIMEFRAME_CONFIRMATION = mt5.TIMEFRAME_H1
    MAX_SPREAD_PIPS = 2.0
    MIN_RISK_PIPS = 8.0
    TAKE_PROFIT_RR = 1.5  # Use 1.5R or 2.0R as per strategy

    # --- Risk Management Settings ---
    RISK_PER_TRADE_PCT = 1.0  # Risk 1% of account balance per trade
    MAX_CONCURRENT_TRADES_TOTAL = 2
    MAX_CONCURRENT_TRADES_PER_SYMBOL = 1

    # --- Pattern Detection Settings ---
    IMPULSE_CANDLES_MIN = 2
    CORRECTION_CANDLES_MIN = 2
    PATTERN_LOOKBACK_CANDLES = 15 # How many candles to look back to find a pattern

# --- 3. MT5 Initialization & Symbol Properties (As Requested) ---

SYMBOLS_AVAILABLE_FOR_TRADE = []
ALL_SYMBOL_PROPERTIES = {}

def initialize_mt5_interface(symbols_to_check):
    """
    Initializes the MetaTrader 5 interface, checks account info, and
    gathers properties for the specified symbols.
    """
    global SYMBOLS_AVAILABLE_FOR_TRADE, ALL_SYMBOL_PROPERTIES
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    logger.info("MetaTrader 5 Initialized")

    account_info = mt5.account_info()
    if account_info is None and not Config.RUN_BACKTEST:
        logger.error(f"Failed to get account info, error code = {mt5.last_error()}")
        mt5.shutdown()
        return False
    elif account_info:
        logger.info(f"Account Login: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
    else:
        logger.warning("Could not get account info. Proceeding for backtest data access.")

    successfully_initialized_symbols = []
    temp_symbol_properties = {}
    for symbol_name in symbols_to_check:
        symbol_info_obj = mt5.symbol_info(symbol_name)
        if symbol_info_obj is None:
            logger.warning(f"Symbol {symbol_name} not found in MT5. Skipping.")
            continue
        
        # In a backtest, we don't need to select symbols, we just pull data.
        if not symbol_info_obj.visible and not Config.RUN_BACKTEST:
            logger.info(f"Symbol {symbol_name} not visible, attempting to select.")
            if not mt5.symbol_select(symbol_name, True):
                logger.warning(f"symbol_select({symbol_name}) failed. Skipping.")
                continue

        pip_value_std = 0.0001
        pip_value_jpy = 0.01
        current_pip_value = pip_value_jpy if 'JPY' in symbol_name.upper() else pip_value_std

        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point,
            'digits': symbol_info_obj.digits,
            'volume_step': symbol_info_obj.volume_step,
            'trade_contract_size': symbol_info_obj.trade_contract_size,
            'pip_value_calc': current_pip_value,
        }
        successfully_initialized_symbols.append(symbol_name)

    if not successfully_initialized_symbols:
        logger.error("No symbols were successfully initialized from the target list.")
        return False

    SYMBOLS_AVAILABLE_FOR_TRADE = successfully_initialized_symbols
    ALL_SYMBOL_PROPERTIES = temp_symbol_properties
    logger.info(f"Properties recorded for: {SYMBOLS_AVAILABLE_FOR_TRADE}")
    return True

# --- 4. Performance & Analysis Helper Functions (To support logging) ---

def calculate_performance_stats(trades_list, start_balance):
    """Calculates performance metrics from a list of closed trades."""
    if not trades_list:
        return {
            'start_balance': start_balance, 'end_balance': start_balance, 'total_trades': 0,
            'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0, 'net_profit': 0,
            'profit_factor': 0, 'max_drawdown_abs': 0, 'max_drawdown_pct': 0
        }

    net_profit = sum(t['pnl'] for t in trades_list)
    end_balance = start_balance + net_profit

    winning_trades = sum(1 for t in trades_list if t['pnl'] > 0)
    losing_trades = len(trades_list) - winning_trades
    win_rate = (winning_trades / len(trades_list)) * 100 if trades_list else 0

    gross_profit = sum(t['pnl'] for t in trades_list if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades_list if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calculate drawdown
    equity_curve = [start_balance]
    peak_equity = start_balance
    max_drawdown_abs = 0
    for trade in trades_list:
        current_equity = equity_curve[-1] + trade['pnl']
        equity_curve.append(current_equity)
        peak_equity = max(peak_equity, current_equity)
        drawdown = peak_equity - current_equity
        max_drawdown_abs = max(max_drawdown_abs, drawdown)

    max_drawdown_pct = (max_drawdown_abs / peak_equity) * 100 if peak_equity > 0 else 0

    return {
        'start_balance': start_balance, 'end_balance': end_balance, 'total_trades': len(trades_list),
        'winning_trades': winning_trades, 'losing_trades': losing_trades, 'win_rate': win_rate,
        'net_profit': net_profit, 'profit_factor': profit_factor, 'max_drawdown_abs': max_drawdown_abs,
        'max_drawdown_pct': max_drawdown_pct
    }

def analyze_rr_distribution(trades_list, symbol_properties):
    """Analyzes the PnL distribution in terms of 'R' (risk multiple)."""
    distribution = {
        'Loss > -1.1R': 0,
        'Loss (-0.9R to -1.1R)': 0,
        'Loss < -0.9R': 0,
        'Breakeven (~0R)': 0,
        'Win (0R to 1.4R)': 0,
        f'Win ({Config.TAKE_PROFIT_RR}R Target Hit)': 0,
        f'Win > {Config.TAKE_PROFIT_RR}R (Slippage)': 0,
    }

    for trade in trades_list:
        risk_pips = abs(trade['entry_price'] - trade['sl'])
        pnl_pips = (trade['close_price'] - trade['entry_price']) if trade['type'] == 'LONG' else (trade['entry_price'] - trade['close_price'])
        
        # Convert to pips
        props = symbol_properties[trade['symbol']]
        pnl_pips /= props['pip_value_calc']

        if risk_pips == 0: continue
        
        r_multiple = pnl_pips / risk_pips

        if r_multiple <= -1.1:
            distribution['Loss > -1.1R'] += 1
        elif -1.1 < r_multiple <= -0.9:
            distribution['Loss (-0.9R to -1.1R)'] += 1
        elif -0.9 < r_multiple < -0.1:
            distribution['Loss < -0.9R'] += 1
        elif -0.1 <= r_multiple <= 0.1:
            distribution['Breakeven (~0R)'] += 1
        elif 0.1 < r_multiple < Config.TAKE_PROFIT_RR - 0.1:
            distribution['Win (0R to 1.4R)'] += 1
        elif Config.TAKE_PROFIT_RR - 0.1 <= r_multiple <= Config.TAKE_PROFIT_RR + 0.1:
            distribution[f'Win ({Config.TAKE_PROFIT_RR}R Target Hit)'] += 1
        else:
            distribution[f'Win > {Config.TAKE_PROFIT_RR}R (Slippage)'] += 1
            
    return distribution

# --- 5. The Backtesting Bot ---

class BullFlagBacktester:
    # Corrected: Class constructor must be __init__
    def __init__(self):
        self.config = Config()
        self.account_balance = self.config.INITIAL_ACCOUNT_BALANCE
        self.open_trades = []
        self.closed_trades = []
        self.symbol_properties = ALL_SYMBOL_PROPERTIES

    def get_data(self, symbol, timeframe, start_date, end_date):
        """Fetches historical data from MT5."""
        # Add buffer for indicator calculation
        start_date_buffered = start_date - timedelta(days=50) 
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 99999) # Fetch a large chunk
        if rates is None or len(rates) == 0:
            logger.error(f"Could not fetch data for {symbol} on {timeframe}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[(df['time'] >= start_date_buffered) & (df['time'] <= end_date)]
        df.set_index('time', inplace=True)
        return df

    def calculate_indicators(self, df):
        """Calculates all required indicators."""
        df.ta.ema(length=6, append=True, col_names=('EMA_6'))
        df.ta.ema(length=18, append=True, col_names=('EMA_18'))
        df.ta.ema(length=50, append=True, col_names=('EMA_50'))
        df.ta.sma(length=200, append=True, col_names=('SMA_200'))
        df.ta.adx(length=14, append=True, col_names=('ADX_14', 'DMP_14', 'DMN_14'))
        # --- NEW: Added ATR for impulse strength check ---
        df.ta.atr(length=14, append=True, col_names=('ATR_14'))
        df.dropna(inplace=True)
        return df

    def detect_flag_pattern(self, df_slice, direction):
        """Identifies the impulse and correction waves for a flag pattern."""
        # df_slice should be the last N candles, e.g., 15
        # Note: df is indexed with most recent candle at the end
        
        # Iterate backwards to find a split point between correction and impulse
        for i in range(Config.CORRECTION_CANDLES_MIN, len(df_slice) - Config.IMPULSE_CANDLES_MIN):
            # The potential correction is the last `i` candles
            correction_df = df_slice.tail(i)
            # The potential impulse is the candles before that
            impulse_df = df_slice.iloc[-(i + Config.IMPULSE_CANDLES_MIN):-i]

            # --- Check Correction Phase ---
            correction_valid = True
            if direction == 'LONG':
                # Must make lower highs
                if not all(correction_df['high'].iloc[j] < correction_df['high'].iloc[j-1] for j in range(1, len(correction_df))):
                    correction_valid = False
                # Must stay above EMAs
                if not (correction_df['close'] > correction_df['EMA_6']).all() and not (correction_df['close'] > correction_df['EMA_18']).all():
                    correction_valid = False
            else: # SHORT
                # Must make higher lows
                if not all(correction_df['low'].iloc[j] > correction_df['low'].iloc[j-1] for j in range(1, len(correction_df))):
                    correction_valid = False
                # Must stay below EMAs
                if not (correction_df['close'] < correction_df['EMA_6']).all() and not (correction_df['close'] < correction_df['EMA_18']).all():
                    correction_valid = False

            if not correction_valid:
                continue

            # --- Check Impulse Phase ---
            impulse_valid = True
            if direction == 'LONG':
                # Must be bullish candles making higher highs
                if not (impulse_df['close'] > impulse_df['open']).all():
                    impulse_valid = False
                if not all(impulse_df['high'].iloc[j] > impulse_df['high'].iloc[j-1] for j in range(1, len(impulse_df))):
                    impulse_valid = False
            else: # SHORT
                 # Must be bearish candles making lower lows
                if not (impulse_df['close'] < impulse_df['open']).all():
                    impulse_valid = False
                if not all(impulse_df['low'].iloc[j] < impulse_df['low'].iloc[j-1] for j in range(1, len(impulse_df))):
                    impulse_valid = False

            if impulse_valid:
                # --- NEW: ATR Filter for Impulse Strength ---
                # Get the most recent ATR value from the slice
                full_atr_value = df_slice['ATR_14'].iloc[-1]
                # Calculate the total range of the impulse leg
                impulse_range = impulse_df['high'].max() - impulse_df['low'].min()
                
                # Compare the impulse range against 1.5 * ATR
                if impulse_range < 1.5 * full_atr_value:
                    continue  # Impulse is too weak, skip this pattern and continue searching
                
                # Pattern found, return the last corrective candle
                return correction_df.iloc[-1]

        return None

    def calculate_lot_size(self, symbol, sl_pips):
        """Calculates trade volume based on fixed percentage risk."""
        props = self.symbol_properties[symbol]
        risk_amount_usd = self.account_balance * (self.config.RISK_PER_TRADE_PCT / 100.0)

        # Pip value in account currency (assuming USD account)
        # This is a simplification; a robust version would handle currency conversion.
        pip_value_in_usd = props['trade_contract_size'] * props['pip_value_calc']
        
        # For pairs like USDJPY, we need to divide by current price
        if "USD" in symbol[:3]:
             # This is an approximation; real value depends on quote price
             pass # For USDXXX, value is fixed per lot
        
        if sl_pips == 0: return 0.0
        
        value_per_pip = pip_value_in_usd / 10 # for a standard lot, adjust based on pair
        
        # A simpler, more universal formula for backtesting:
        # risk_per_lot = sl_pips * (pip value for 1 lot)
        # lots = total_risk_amount / risk_per_lot
        # Note: A truly accurate calculation requires the quote price at the moment of trade
        # to convert pip value to account currency. We'll use a standard approximation.
        
        # A simplified formula for position size:
        risk_in_pips = sl_pips * props['pip_value_calc']
        if risk_in_pips == 0: return 0.0
        
        lot_size = risk_amount_usd / (risk_in_pips * props['trade_contract_size'])
        
        # Normalize to the symbol's volume step
        lot_size = max(props['volume_step'], round(lot_size / props['volume_step']) * props['volume_step'])
        return lot_size


    def run(self):
        """Main backtesting loop."""
        logger.info("Starting backtest...")
        
        # Fetch all data upfront
        all_data_m15 = {s: self.get_data(s, self.config.TIMEFRAME_EXECUTION, self.config.BACKTEST_START_DATE, self.config.BACKTEST_END_DATE) for s in SYMBOLS_AVAILABLE_FOR_TRADE}
        all_data_h1 = {s: self.get_data(s, self.config.TIMEFRAME_CONFIRMATION, self.config.BACKTEST_START_DATE, self.config.BACKTEST_END_DATE) for s in SYMBOLS_AVAILABLE_FOR_TRADE}

        # Calculate indicators on all data
        for symbol in SYMBOLS_AVAILABLE_FOR_TRADE:
            if all_data_m15[symbol] is not None:
                all_data_m15[symbol] = self.calculate_indicators(all_data_m15[symbol])
            if all_data_h1[symbol] is not None:
                all_data_h1[symbol] = self.calculate_indicators(all_data_h1[symbol])
        
        # Get the master timeline from a major pair like EURUSD
        timeline = all_data_m15['EURUSD'][all_data_m15['EURUSD'].index >= self.config.BACKTEST_START_DATE].index

        # --- Main Candle Loop ---
        for current_time in timeline:
            # 1. Manage Open Trades
            trades_to_close = []
            for trade in self.open_trades:
                symbol_data = all_data_m15[trade['symbol']]
                if current_time not in symbol_data.index: continue
                
                current_candle = symbol_data.loc[current_time]
                
                # Check SL/TP
                pnl = 0
                close_reason = None
                close_price = trade['entry_price']

                if trade['type'] == 'LONG':
                    if current_candle['low'] <= trade['sl']:
                        close_price = trade['sl']
                        close_reason = 'SL Hit'
                    elif current_candle['high'] >= trade['tp']:
                        close_price = trade['tp']
                        close_reason = 'TP Hit'
                elif trade['type'] == 'SHORT':
                    if current_candle['high'] >= trade['sl']:
                        close_price = trade['sl']
                        close_reason = 'SL Hit'
                    elif current_candle['low'] <= trade['tp']:
                        close_price = trade['tp']
                        close_reason = 'TP Hit'
                
                # Rollover close rule: Close trade if it's held into the next day
                if trade['entry_time'].date() < current_time.date():
                    close_price = current_candle['open'] # Close at next day's open
                    close_reason = 'End of Day Close'

                if close_reason:
                    pnl_points = (close_price - trade['entry_price']) if trade['type'] == 'LONG' else (trade['entry_price'] - close_price)
                    pnl = pnl_points * trade['volume'] * self.symbol_properties[trade['symbol']]['trade_contract_size']
                    
                    self.account_balance += pnl
                    trade['pnl'] = pnl
                    trade['close_price'] = close_price
                    trade['close_time'] = current_time
                    trade['close_reason'] = close_reason
                    self.closed_trades.append(trade)
                    trades_to_close.append(trade)
            
            self.open_trades = [t for t in self.open_trades if t not in trades_to_close]

            # 2. Check for New Trade Setups
            if len(self.open_trades) >= self.config.MAX_CONCURRENT_TRADES_TOTAL:
                continue

            for symbol in SYMBOLS_AVAILABLE_FOR_TRADE:
                # Max 1 trade per symbol
                if any(t['symbol'] == symbol for t in self.open_trades):
                    continue
                
                # Get data for current time
                df_m15 = all_data_m15[symbol].loc[:current_time]
                df_h1 = all_data_h1[symbol].loc[:current_time]
                
                if df_m15.empty or df_h1.empty: continue
                
                latest_m15 = df_m15.iloc[-1]
                latest_h1 = df_h1.iloc[-1]

                # --- START: NEW LOGIC INSERTED AS REQUESTED ---
                # Reject tight EMA cluster (weak trend)
                ema_distance = abs(latest_m15['EMA_6'] - latest_m15['SMA_200'])
                symbol_price = latest_m15['close']

                # Normalize distance to relative price (e.g., 0.001 = 0.1%)
                if symbol_price > 0:
                    relative_distance = ema_distance / symbol_price
                else:
                    relative_distance = 0

                # Threshold of 0.1% relative distance. You can adjust this value.
                if relative_distance < 0.001:  
                    # Using INFO level to be visible with current logger settings.
                    # This log helps understand why some periods have no trades.
                    logger.info(f"[{current_time}] SKIP {symbol}: Weak trend. EMA6-SMA200 spread is too tight ({relative_distance:.5f} < 0.001).")
                    continue
                # --- END: NEW LOGIC INSERTED ---
                
                # --- Step 1 & 2: Trend & ADX Filter ---
                is_long_trend = (latest_m15['EMA_6'] > latest_m15['EMA_18'] > latest_m15['EMA_50'] > latest_m15['SMA_200'] and
                                 latest_h1['EMA_6'] > latest_h1['EMA_18'] > latest_h1['EMA_50'] > latest_h1['SMA_200'])
                is_short_trend = (latest_m15['EMA_6'] < latest_m15['EMA_18'] < latest_m15['EMA_50'] < latest_m15['SMA_200'] and
                                  latest_h1['EMA_6'] < latest_h1['EMA_18'] < latest_h1['EMA_50'] < latest_h1['SMA_200'])
                
                is_adx_strong = latest_m15['ADX_14'] > 25

                if not (is_long_trend or is_short_trend) or not is_adx_strong:
                    continue

                # --- Step 3: Flag Pattern Detection ---
                direction = 'LONG' if is_long_trend else 'SHORT'
                pattern_lookback_df = df_m15.tail(self.config.PATTERN_LOOKBACK_CANDLES)
                
                last_corrective_candle = self.detect_flag_pattern(pattern_lookback_df, direction)

                if last_corrective_candle is None:
                    continue
                
                # --- Step 4: Entry, SL, TP Calculation ---
                props = self.symbol_properties[symbol]
                spread_points = (self.config.MAX_SPREAD_PIPS * props['pip_value_calc']) / props['point']
                
                entry_price, sl, tp = 0, 0, 0
                if direction == 'LONG':
                    entry_price_raw = last_corrective_candle['high']
                    entry_price = entry_price_raw + (spread_points * props['point'])
                    sl = pattern_lookback_df.tail(Config.CORRECTION_CANDLES_MIN)['low'].min()
                    
                    risk_pips = (entry_price - sl) / props['pip_value_calc']
                    if risk_pips < self.config.MIN_RISK_PIPS: continue

                    tp = entry_price + (risk_pips * props['pip_value_calc'] * self.config.TAKE_PROFIT_RR)
                
                else: # SHORT
                    entry_price_raw = last_corrective_candle['low']
                    entry_price = entry_price_raw - (spread_points * props['point'])
                    sl = pattern_lookback_df.tail(Config.CORRECTION_CANDLES_MIN)['high'].max()

                    risk_pips = (sl - entry_price) / props['pip_value_calc']
                    if risk_pips < self.config.MIN_RISK_PIPS: continue

                    tp = entry_price - (risk_pips * props['pip_value_calc'] * self.config.TAKE_PROFIT_RR)

                # --- Step 7: Risk Management & Trade Execution ---
                lot_size = self.calculate_lot_size(symbol, risk_pips)
                if lot_size <= 0:
                    continue
                
                # "Place" the trade
                trade_id = f"trade_{len(self.open_trades) + len(self.closed_trades) + 1}"
                self.open_trades.append({
                    'id': trade_id, 'symbol': symbol, 'type': direction,
                    'entry_time': current_time, 'entry_price': entry_price_raw, # Enter at raw price, SL/TP accounts for spread
                    'sl': sl, 'tp': tp, 'volume': lot_size
                })
                logger.info(f"[{current_time}] OPEN {direction} {symbol} @ {entry_price_raw:.5f} | SL: {sl:.5f} TP: {tp:.5f} | Risk Pips: {risk_pips:.2f}")

        # --- End of Backtest Loop ---
        self.log_results()

    def log_results(self):
        """Generates and prints the final performance summary."""
        # Sort all trades by entry time for accurate reporting
        self.closed_trades.sort(key=lambda x: x['entry_time'])

        # --- 2. Per-Symbol Summary Logging (As Requested) ---
        trades_per_symbol_map = {}
        for trade in self.closed_trades:
            if trade['symbol'] not in trades_per_symbol_map:
                trades_per_symbol_map[trade['symbol']] = []
            trades_per_symbol_map[trade['symbol']].append(trade)

        logger.info("\n\n===== All Symbol Simulations Complete. Generating Summaries. =====")

        for symbol_iter in SYMBOLS_AVAILABLE_FOR_TRADE:
            symbol_trades_list = trades_per_symbol_map.get(symbol_iter, [])
            logger.info(f"\n--- Performance Summary for Symbol: {symbol_iter} ---")
            logger.info(f"  Period: {self.config.BACKTEST_START_DATE.strftime('%Y-%m-%d')} to {self.config.BACKTEST_END_DATE.strftime('%Y-%m-%d')}")

            if not symbol_trades_list:
                logger.info(f"  No trades executed for {symbol_iter} during the backtest period.")
            else:
                # For symbol summary, the "start balance" is not meaningful. We report on PnL.
                symbol_pnl = sum(t['pnl'] for t in symbol_trades_list)
                symbol_stats = calculate_performance_stats(symbol_trades_list, 0) # Use 0 as base for stats not needing balance
                logger.info(f"  Total Trades: {symbol_stats['total_trades']}")
                logger.info(f"  Winning Trades: {symbol_stats['winning_trades']}")
                logger.info(f"  Losing Trades: {symbol_stats['losing_trades']}")
                logger.info(f"  Win Rate: {symbol_stats['win_rate']:.2f}%")
                logger.info(f"  Net Profit (for this symbol): {symbol_pnl:.2f} USD")
                logger.info(f"  Profit Factor (for this symbol): {symbol_stats['profit_factor']:.2f}")

        # --- 3. Overall Log Summary Code (As Requested) ---
        logger.info("\n\n===== Overall Backtest Performance Summary =====")
        if self.closed_trades:
            overall_stats = calculate_performance_stats(self.closed_trades, self.config.INITIAL_ACCOUNT_BALANCE)
            rr_distribution = analyze_rr_distribution(self.closed_trades, self.symbol_properties)
            
            logger.info(f"Tested Symbols: {SYMBOLS_AVAILABLE_FOR_TRADE}")
            logger.info(f"Overall Period: {self.config.BACKTEST_START_DATE.strftime('%Y-%m-%d')} to {self.config.BACKTEST_END_DATE.strftime('%Y-%m-%d')}")
            logger.info(f"Overall Starting Balance: {overall_stats['start_balance']:.2f} USD")
            logger.info(f"Overall Ending Balance: {overall_stats['end_balance']:.2f} USD")
            logger.info(f"Overall Total Trades: {overall_stats['total_trades']}")
            logger.info(f"Overall Winning Trades: {overall_stats['winning_trades']}")
            logger.info(f"Overall Losing Trades: {overall_stats['losing_trades']}")
            logger.info(f"Overall Win Rate: {overall_stats['win_rate']:.2f}%")
            logger.info(f"Overall Net Profit: {overall_stats['net_profit']:.2f} USD")
            logger.info(f"Overall Profit Factor: {overall_stats['profit_factor']:.2f}")
            logger.info(f"Overall Max Drawdown: {overall_stats['max_drawdown_abs']:.2f} USD ({overall_stats['max_drawdown_pct']:.2f}%)")

            logger.info("\n--- RR Distribution Summary ---")
            total_counted_trades = sum(rr_distribution.values())
            logger.info(f"  (Analysis based on {total_counted_trades} of {overall_stats['total_trades']} total trades)")
            for bucket, count in rr_distribution.items():
                if count > 0:
                    percentage = (count / total_counted_trades) * 100 if total_counted_trades > 0 else 0
                    logger.info(f"  {bucket:<35}: {count:<5} trades ({percentage:.2f}%)")
        else:
            logger.info("No trades were executed across any symbols during the backtest period.")
            logger.info(f"Overall Starting Balance: {self.config.INITIAL_ACCOUNT_BALANCE:.2f} USD")
            logger.info(f"Overall Ending Balance: {self.account_balance:.2f} USD")

# --- 6. Main Execution Block ---

# Corrected: Use __name__ to ensure this block runs only when the script is executed directly
if __name__ == "__main__":
    if initialize_mt5_interface(Config.SYMBOLS_TO_TRADE):
        bot = BullFlagBacktester()
        bot.run()
        mt5.shutdown()
        logger.info("MT5 connection shut down.")
    else:
        logger.error("Could not start backtester due to initialization failure.")