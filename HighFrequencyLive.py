import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time
import datetime
import numpy as np
import logging
import os
import math # Added for math.floor, math.log10

# --- General Configuration ---
SYMBOLS_TO_TRADE_INITIAL = ["EURUSD", "AUDUSD", "USDCHF", "USDCAD",
                             "GBPJPY", "XAUUSD", "NZDUSD", "EURCHF", "AUDJPY", "EURNZD", "GBPNZD",
                            "USOIL", "CADJPY",   "XAGUSD", "UKOIL",
                            "BTCUSD", "BTCJPY", "BTCXAU", "ETHUSD"]

ENTRY_TIMEFRAME_MT5 = mt5.TIMEFRAME_M5
FILTER_TIMEFRAME_MT5 = mt5.TIMEFRAME_H1

# --- Live Trading Specific Configuration ---
MAGIC_NUMBER = 123457
MAX_SLIPPAGE = 5

# --- Trading Hours Configuration (UTC) ---
TRADING_HOUR_START_UTC = 0
TRADING_HOUR_END_UTC = 20

# --- Symbol-Specific Session Filters (UTC) ---
SYMBOL_SESSIONS = {
    "EURUSD": [(7, 16)], "GBPUSD": [(7, 16)], "AUDUSD": [(0, 4), (7, 16)],
    "USDCHF": [(7, 16)], "USDCAD": [(12, 16)], "USDJPY": [(0, 4), (12, 16)],
    "EURJPY": [(0, 4), (7, 12)], "GBPJPY": [(7, 16)], "NZDUSD": [(0, 4), (7, 16)],
    "EURCHF": [(7, 12)], "AUDJPY": [(0, 4)], "CADJPY": [(12, 16)],
    "EURNZD": [(0, 4), (7, 12)], "GBPNZD": [(7, 12)], "XAUUSD": [(7, 16)],
    "XAGUSD": [(7, 16)], "XPTUSD": [(7, 16)], "USOIL":  [(12, 17)],
    "UKOIL":  [(7, 16)], "BTCUSD":[(7, 16)], "BTCJPY":[(0, 14)], "BTCXAU":[(7, 16)], "ETHUSD":[(7, 16)]
}

# --- Strategy Configuration ---
RISK_REWARD_RATIO = 2.0
M5_EMA_SHORT_PERIOD = 20
M5_EMA_MID_PERIOD = 50
M5_EMA_LONG_PERIOD = 100
H1_EMA_SHORT_PERIOD = 20
H1_EMA_MID_PERIOD = 50
H1_EMA_LONG_PERIOD = 100
H1_RSI_PERIOD = 14
H1_RSI_BULL_THRESHOLD = 55
H1_RSI_BEAR_THRESHOLD = 45
H1_MACD_FAST = 12
H1_MACD_SLOW = 26
H1_MACD_SIGNAL = 9

FRACTAL_LOOKBACK = 2
FRACTAL_LOOKFORWARD = 2
N_BARS_FOR_INDICATORS = 250
ADX_PERIOD = 14
ADX_THRESHOLD = 25

ATR_PERIOD = 14
ATR_ROLLING_MEAN_PERIOD = 50
ATR_MULTIPLIER_LOW = 0.8
ATR_MULTIPLIER_HIGH = 2.5

# --- Risk Management for Live Trading ---
MAX_TRADES_PER_SYMBOL_PER_DAY = 3
DAILY_MAX_ACCOUNT_RISK_PERCENT = 5.0
RISK_PER_TRADE_ACCOUNT_PERCENT = 1.0

# --- Trailing Stop Loss Configuration ---
TSL_ACTIVATION_RR_RATIO = 1.5

# --- Global State Variables ---
SYMBOLS_TO_TRADE = []
SYMBOL_PROPERTIES = {}
daily_trade_counts = {}
daily_start_balance_utc = 0.0
last_checked_day_utc_for_reset = None
daily_risk_budget_currency_global = 0.0
current_daily_risked_amount_global = 0.0
daily_risk_budget_exceeded_today_global = False

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()])

# --- Trade History Logging ---
TRADE_HISTORY_FILE = "trading_history.csv"
TRADE_HISTORY_COLUMNS = [
    "TicketID", "PositionID", "Symbol", "Type", "OpenTimeUTC", "EntryPrice",
    "LotSize", "SL_Price", "TP_Price", "CloseTimeUTC", "ExitPrice",
    "PNL_AccountCCY", "OpenComment", "CloseReason", "RiskedAmount"
]
trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)

def load_or_initialize_trade_history():
    global trade_history_df
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            temp_df_list = []
            with open(TRADE_HISTORY_FILE, 'r') as f:
                header = f.readline().strip()
                if not header:
                     trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
                     logging.info(f"Trade history file {TRADE_HISTORY_FILE} is empty. Initialized new history.")
                     return

                num_columns_expected = len(TRADE_HISTORY_COLUMNS)
                num_columns_file = len(header.split(','))

                if num_columns_file not in [num_columns_expected, num_columns_expected -1]: # Allow loading if file has one less column (RiskedAmount might be new)
                    logging.warning(f"Trade history CSV header mismatch. Expected {num_columns_expected} or {num_columns_expected-1} columns, got {num_columns_file}. Re-initializing.")
                    trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
                    return

                temp_df_list.append(header)
                for line in f:
                    if line.strip().startswith("--- Performance Summary ---"):
                        break
                    if len(line.strip().split(',')) == num_columns_file:
                         temp_df_list.append(line.strip())

            if len(temp_df_list) > 1:
                from io import StringIO
                csv_data_str = "\n".join(temp_df_list)
                trade_history_df = pd.read_csv(StringIO(csv_data_str))
                if 'RiskedAmount' not in trade_history_df.columns:
                    trade_history_df['RiskedAmount'] = np.nan
            else:
                trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)

            for col in ["OpenTimeUTC", "CloseTimeUTC"]:
                if col in trade_history_df.columns:
                    trade_history_df[col] = pd.to_datetime(trade_history_df[col], errors='coerce', utc=True)

            numeric_cols = ['EntryPrice', 'LotSize', 'SL_Price', 'TP_Price', 'ExitPrice', 'PNL_AccountCCY', 'TicketID', 'PositionID', 'RiskedAmount']
            for col in numeric_cols:
                if col in trade_history_df.columns:
                    trade_history_df[col] = pd.to_numeric(trade_history_df[col], errors='coerce')

            logging.info(f"Loaded {len(trade_history_df)} trade records from {TRADE_HISTORY_FILE}")
        except Exception as e:
            logging.error(f"Error loading trade history: {e}. Initializing new history.")
            trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
    else:
        trade_history_df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
        logging.info(f"No existing trade history file. Initialized new history.")

def save_trade_history():
    global trade_history_df
    try:
        df_to_save = trade_history_df.copy()
        for col in ['PNL_AccountCCY', 'RiskedAmount']:
             if col in df_to_save.columns:
                df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')

        df_to_save.to_csv(TRADE_HISTORY_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')

        summary_lines = []
        closed_trades = df_to_save[pd.notna(df_to_save['CloseTimeUTC']) & pd.notna(df_to_save['PNL_AccountCCY'])].copy()
        # IMPORTANT: Ensure chronological order for PNL sum and drawdown
        closed_trades.sort_values(by='CloseTimeUTC', inplace=True)


        summary_lines.append("\n--- Performance Summary ---")
        acc_info = mt5.account_info() # Get account info once for currency
        currency = acc_info.currency if acc_info else "N/A"

        if not closed_trades.empty:
            total_trades = len(closed_trades)
            winning_trades_df = closed_trades[closed_trades['PNL_AccountCCY'] > 0]
            losing_trades_df = closed_trades[closed_trades['PNL_AccountCCY'] < 0]
            num_wins = len(winning_trades_df)
            num_losses = len(losing_trades_df)
            win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
            total_pnl = closed_trades['PNL_AccountCCY'].sum()
            total_risked = closed_trades['RiskedAmount'].sum() if 'RiskedAmount' in closed_trades.columns and closed_trades['RiskedAmount'].notna().any() else np.nan
            sum_profits = winning_trades_df['PNL_AccountCCY'].sum()
            sum_losses = abs(losing_trades_df['PNL_AccountCCY'].sum())
            profit_factor = (sum_profits / sum_losses) if sum_losses > 0 else (float('inf') if sum_profits > 0 else 0.0)

            summary_lines.append(f"Total Closed Trades: {total_trades}")
            summary_lines.append(f"Winning Trades: {num_wins}")
            summary_lines.append(f"Losing Trades: {num_losses}")
            summary_lines.append(f"Win Rate: {win_rate:.2f}%")
            summary_lines.append(f"Total PNL ({currency}): {total_pnl:.2f}")
            if not pd.isna(total_risked):
                 summary_lines.append(f"Total Amount Risked ({currency}): {total_risked:.2f}")
            summary_lines.append(f"Profit Factor: {profit_factor:.2f}")

            # --- Max Drawdown Calculation ---
            max_drawdown_value_calc = 0.0
            max_drawdown_percentage_calc = 0.0
            
            pnls = closed_trades['PNL_AccountCCY'].values # Already sorted by 'CloseTimeUTC'

            # Use acc_info fetched earlier for balance, or try to get fresh one if needed
            account_info_for_dd = acc_info if acc_info else mt5.account_info()

            if account_info_for_dd and account_info_for_dd.balance is not None:
                current_balance = account_info_for_dd.balance
                # total_pnl_in_csv is already calculated as total_pnl
                estimated_start_balance = current_balance - total_pnl

                equity_curve_list = [estimated_start_balance]
                for pnl_item in pnls:
                    equity_curve_list.append(equity_curve_list[-1] + pnl_item)
                
                equity_series = pd.Series(equity_curve_list)

                if len(equity_series) > 1: # Need at least start balance + 1 PNL outcome
                    running_max_equity = equity_series.expanding(min_periods=1).max()
                    drawdown_abs_values = running_max_equity - equity_series
                    max_drawdown_value_calc = drawdown_abs_values.max()
                    
                    temp_running_max_equity = running_max_equity.copy()
                    temp_running_max_equity[temp_running_max_equity <= 1e-9] = np.nan # Avoid division by zero/small num
                    
                    drawdown_percentages = (drawdown_abs_values / temp_running_max_equity) * 100.0
                    valid_drawdown_percentages = drawdown_percentages.dropna()

                    if not valid_drawdown_percentages.empty:
                        max_drawdown_percentage_calc = valid_drawdown_percentages.max()
                    else: # All peaks non-positive or only one data point in series
                        max_drawdown_percentage_calc = 0.0 if max_drawdown_value_calc == 0 else np.nan
                else: # Only starting balance, no trades, or equity_series too short
                    max_drawdown_value_calc = 0.0
                    max_drawdown_percentage_calc = 0.0
            else: # Fallback if account info/balance is not available
                cumulative_pnl = pd.Series(pnls).cumsum()
                equity_from_zero_start = pd.concat([pd.Series([0.0]), cumulative_pnl], ignore_index=True)
                
                if len(equity_from_zero_start) > 1:
                    peak_equity_from_zero = equity_from_zero_start.expanding(min_periods=1).max()
                    drawdown_from_zero_peak = peak_equity_from_zero - equity_from_zero_start
                    max_drawdown_value_calc = drawdown_from_zero_peak.max()
                else:
                    max_drawdown_value_calc = 0.0
                max_drawdown_percentage_calc = np.nan

            summary_lines.append(f"Max Drawdown ({currency}): {max_drawdown_value_calc:.2f}")
            if pd.notna(max_drawdown_percentage_calc):
                summary_lines.append(f"Max Drawdown (%): {max_drawdown_percentage_calc:.2f}%")
            else:
                summary_lines.append(f"Max Drawdown (%): N/A")
        else: # No closed trades
            summary_lines.append("No closed trades to analyze for drawdown.")
            summary_lines.append(f"Max Drawdown ({currency}): 0.00")
            summary_lines.append(f"Max Drawdown (%): 0.00%")


        with open(TRADE_HISTORY_FILE, 'a') as f:
            for line in summary_lines:
                f.write(line + "\n")
        logging.info(f"Trade history and summary saved to {TRADE_HISTORY_FILE}")
    except Exception as e:
        logging.error(f"Error saving trade history or calculating summary: {e}", exc_info=True)


def log_opened_trade(order_result, symbol, trade_type_mt5, lot_size, sl_price, tp_price, comment, risked_amount_this_trade, position_id_override=0):
    global trade_history_df
    if order_result and order_result.retcode == mt5.TRADE_RETCODE_DONE:
        final_position_id = position_id_override if position_id_override > 0 else getattr(order_result, 'position', 0)

        if final_position_id == 0 and order_result.deal > 0 and position_id_override == 0:
            logging.warning(f"log_opened_trade: PositionID is 0. order_result.deal ({order_result.deal}) exists. Attempting to find PositionID again via DealID.")
            time.sleep(0.5)
            deals = mt5.history_deals_get(ticket=order_result.deal)
            if deals and len(deals) > 0:
                final_position_id = deals[0].position_id
                logging.info(f"log_opened_trade: Re-acquired PositionID {final_position_id} via DealID {order_result.deal} for order {order_result.order}")
            else:
                time.sleep(1)
                positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER)
                if positions:
                    for pos in positions:
                        if pos.ticket == order_result.order or pos.comment == comment:
                             logging.warning(f"log_opened_trade: PositionID still ambiguous after deal check. Found active positions for {symbol} by magic. Result PositionID was {final_position_id}.")
                             break
                if final_position_id == 0:
                     logging.error(f"log_opened_trade: Still could not determine PositionID for order {order_result.order} (DealID {order_result.deal}). Logging with PositionID 0.")
        elif final_position_id == 0 and position_id_override == 0:
             logging.error(f"log_opened_trade: Order {order_result.order} resulted in PositionID 0 even after checks. Trade history for this position might be incomplete.")

        new_trade_record = {
            "TicketID": order_result.order, "PositionID": final_position_id, "Symbol": symbol,
            "Type": "BUY" if trade_type_mt5 == mt5.ORDER_TYPE_BUY else "SELL",
            "OpenTimeUTC": datetime.datetime.now(datetime.timezone.utc),
            "EntryPrice": order_result.price, "LotSize": lot_size,
            "SL_Price": sl_price, "TP_Price": tp_price,
            "CloseTimeUTC": pd.NaT, "ExitPrice": np.nan, "PNL_AccountCCY": np.nan,
            "OpenComment": comment, "CloseReason": "", "RiskedAmount": risked_amount_this_trade
        }
        new_row_df = pd.DataFrame([new_trade_record])
        trade_history_df = pd.concat([trade_history_df, new_row_df], ignore_index=True)
        save_trade_history()
        logging.info(f"Logged opened trade. PositionID: {final_position_id}, OrderTicket: {order_result.order}, Risked: {risked_amount_this_trade:.2f}")

def check_and_log_closed_trades():
    global trade_history_df
    if trade_history_df.empty: return

    open_trades_in_log = trade_history_df[
        pd.isna(trade_history_df['CloseTimeUTC']) &
        pd.notna(trade_history_df['PositionID']) &
        (trade_history_df['PositionID'] != 0)
    ].copy()

    if open_trades_in_log.empty: return

    mt5_open_positions_by_magic = mt5.positions_get(magic=MAGIC_NUMBER)
    mt5_active_position_ids = {pos.ticket for pos in mt5_open_positions_by_magic} if mt5_open_positions_by_magic else set()

    updated_history = False
    for index, trade_row in open_trades_in_log.iterrows():
        position_id_from_log = int(trade_row['PositionID'])

        if position_id_from_log not in mt5_active_position_ids:
            logging.info(f"Position {position_id_from_log} (Order: {trade_row['TicketID']}) no longer active. Fetching history...")
            time.sleep(1)
            deals = mt5.history_deals_get(position=position_id_from_log)

            if deals is None or len(deals) == 0:
                logging.warning(f"Could not get history deals for closed position {position_id_from_log}. Error: {mt5.last_error()}. Will retry later.")
                continue

            closing_deal = None; position_total_pnl = 0.0
            deals = sorted(deals, key=lambda d: d.time_msc)

            for deal in deals:
                if deal.position_id == position_id_from_log:
                    position_total_pnl += deal.profit
                    if deal.entry == mt5.DEAL_ENTRY_OUT or deal.entry == mt5.DEAL_ENTRY_OUT_BY:
                        closing_deal = deal

            if closing_deal:
                trade_history_df.loc[index, 'CloseTimeUTC'] = pd.to_datetime(closing_deal.time_msc, unit='ms', utc=True)
                trade_history_df.loc[index, 'ExitPrice'] = closing_deal.price
                trade_history_df.loc[index, 'PNL_AccountCCY'] = position_total_pnl
                close_reason_map = {
                    mt5.DEAL_REASON_CLIENT: "Manual (Client/Desktop)", mt5.DEAL_REASON_MOBILE: "Manual (Mobile App)",
                    mt5.DEAL_REASON_WEB: "Manual (Web Platform)", mt5.DEAL_REASON_EXPERT: "Programmatic (EA/Script)",
                    mt5.DEAL_REASON_SL: "StopLoss", mt5.DEAL_REASON_TP: "TakeProfit", mt5.DEAL_REASON_SO: "StopOut",
                }
                close_reason = close_reason_map.get(closing_deal.reason, f"Other ({closing_deal.reason})")
                if closing_deal.comment:
                    comment_lower = closing_deal.comment.lower()
                    potentially_obscured_tsl = closing_deal.reason in [mt5.DEAL_REASON_EXPERT, mt5.DEAL_REASON_CLIENT]
                    if potentially_obscured_tsl:
                        if "sl" in comment_lower and "tp" not in comment_lower:
                            close_reason = f"{close_reason}/TSL(comment:sl)"
                trade_history_df.loc[index, 'CloseReason'] = close_reason
                logging.info(f"Logged closure for PositionID: {position_id_from_log}. Exit: {closing_deal.price}, PNL: {position_total_pnl:.2f}, Reason: {close_reason} (Raw Reason Code: {closing_deal.reason})")
                updated_history = True
            else:
                logging.warning(f"Position {position_id_from_log} seems closed (not in active positions), but no definitive closing deal (DEAL_ENTRY_OUT/OUT_BY) found. PNL from deals so far: {position_total_pnl:.2f}. Deals count: {len(deals) if deals else 0}. Raw deals: {deals}")
    if updated_history:
        save_trade_history()

def timeframe_to_string(tf_int):
    timeframes = {
        mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M2: "M2", mt5.TIMEFRAME_M3: "M3",
        mt5.TIMEFRAME_M4: "M4", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M6: "M6",
        mt5.TIMEFRAME_M10: "M10", mt5.TIMEFRAME_M12: "M12", mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M20: "M20", mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H2: "H2", mt5.TIMEFRAME_H3: "H3",
        mt5.TIMEFRAME_H4: "H4", mt5.TIMEFRAME_H6: "H6", mt5.TIMEFRAME_H8: "H8",
        mt5.TIMEFRAME_H12: "H12",
        mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1"
    }
    return timeframes.get(tf_int, f"UnknownTF({tf_int})")

def initialize_mt5_for_live():
    global SYMBOLS_TO_TRADE, SYMBOL_PROPERTIES
    if not mt5.initialize():
        logging.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    logging.info("MetaTrader 5 Initialized for Live Trading")
    account_info = mt5.account_info()
    if account_info is None:
        logging.error(f"Failed to get account info, error code = {mt5.last_error()}")
        mt5.shutdown()
        return False
    logging.info(f"Account Login: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
    symbols_to_check_locally = list(SYMBOLS_TO_TRADE_INITIAL)
    successfully_initialized_symbols = []
    temp_symbol_properties = {}
    for symbol_name in symbols_to_check_locally:
        symbol_info_obj = mt5.symbol_info(symbol_name)
        if symbol_info_obj is None:
            logging.warning(f"Symbol {symbol_name} not found by broker. Skipping.")
            continue
        if not symbol_info_obj.visible:
            logging.info(f"Symbol {symbol_name} not visible, trying to select.")
            if not mt5.symbol_select(symbol_name, True):
                logging.warning(f"symbol_select({symbol_name}) failed, error code = {mt5.last_error()}. Skipping.")
                continue
            time.sleep(0.1)
            symbol_info_obj = mt5.symbol_info(symbol_name)
            if symbol_info_obj is None or not symbol_info_obj.visible:
                 logging.warning(f"Symbol {symbol_name} still not available after selection. Skipping.")
                 continue
        temp_symbol_properties[symbol_name] = {
            'point': symbol_info_obj.point, 'digits': symbol_info_obj.digits,
            'trade_tick_size': symbol_info_obj.trade_tick_size,
            'trade_tick_value': symbol_info_obj.trade_tick_value,
            'volume_min': symbol_info_obj.volume_min,
            'volume_step': symbol_info_obj.volume_step,
            'volume_max': symbol_info_obj.volume_max,
            'trade_contract_size': symbol_info_obj.trade_contract_size,
            'spread': symbol_info_obj.spread,
            'currency_profit': symbol_info_obj.currency_profit,
            'currency_margin': symbol_info_obj.currency_margin,
        }
        successfully_initialized_symbols.append(symbol_name)
        logging.info(f"Symbol {symbol_name} available. Min Lot: {temp_symbol_properties[symbol_name]['volume_min']}, Max Lot: {temp_symbol_properties[symbol_name]['volume_max']}, Point: {temp_symbol_properties[symbol_name]['point']}")
    if not successfully_initialized_symbols:
        logging.error("No symbols were successfully initialized for trading.")
        mt5.shutdown()
        return False
    SYMBOLS_TO_TRADE = successfully_initialized_symbols
    SYMBOL_PROPERTIES = temp_symbol_properties
    logging.info(f"Successfully initialized symbols for trading: {SYMBOLS_TO_TRADE}")
    return True

def shutdown_mt5():
    mt5.shutdown()
    logging.info("MetaTrader 5 Shutdown")

def williams_fractals(df_high, df_low, n_left=2, n_right=2):
    up_fractals = pd.Series(index=df_high.index, dtype='bool').fillna(False)
    down_fractals = pd.Series(index=df_low.index, dtype='bool').fillna(False)
    if len(df_high) < n_left + n_right + 1: return pd.DataFrame({'fractal_up': up_fractals, 'fractal_down': down_fractals})
    for i in range(n_left, len(df_high) - n_right):
        is_up = all(df_high.iloc[i] >= df_high.iloc[i-j] for j in range(1, n_left + 1)) and \
                all(df_high.iloc[i] > df_high.iloc[i+j] for j in range(1, n_right + 1))
        if is_up: up_fractals.iloc[i] = True
        is_down = all(df_low.iloc[i] <= df_low.iloc[i-j] for j in range(1, n_left + 1)) and \
                  all(df_low.iloc[i] < df_low.iloc[i+j] for j in range(1, n_right + 1))
        if is_down: down_fractals.iloc[i] = True
    return pd.DataFrame({'fractal_up': up_fractals, 'fractal_down': down_fractals})

def calculate_indicators_for_df(df, ema_short_len, ema_mid_len, ema_long_len, adx_len=None, rsi_len=None, prefix="", **kwargs):
    if df.empty: return df
    df[f'{prefix}EMA_{ema_short_len}'] = ta.ema(df['close'], length=ema_short_len)
    df[f'{prefix}EMA_{ema_mid_len}'] = ta.ema(df['close'], length=ema_mid_len)
    df[f'{prefix}EMA_{ema_long_len}'] = ta.ema(df['close'], length=ema_long_len)

    if adx_len:
        adx_indicator = ta.adx(df['high'], df['low'], df['close'], length=adx_len)
        if adx_indicator is not None and not adx_indicator.empty and f'ADX_{adx_len}' in adx_indicator.columns:
            df[f'{prefix}ADX_{adx_len}'] = adx_indicator[f'ADX_{adx_len}']
        else: df[f'{prefix}ADX_{adx_len}'] = np.nan
    if rsi_len:
        df[f'{prefix}RSI_{rsi_len}'] = ta.rsi(df['close'], length=rsi_len)
    if not prefix and 'atr_len' in kwargs:
        atr_len_val = kwargs['atr_len']
        atr_col_name = f'ATR_{atr_len_val}'
        atr_sma_col_name = f'ATR_{atr_len_val}_SMA{ATR_ROLLING_MEAN_PERIOD}'
        df[atr_col_name] = ta.atr(df['high'], df['low'], df['close'], length=atr_len_val)
        df[atr_sma_col_name] = df[atr_col_name].rolling(window=ATR_ROLLING_MEAN_PERIOD, min_periods=max(1, ATR_ROLLING_MEAN_PERIOD // 2)).mean()
    if prefix == "H1_":
        if kwargs.get('add_macd', False):
            macd_df = ta.macd(df['close'], fast=H1_MACD_FAST, slow=H1_MACD_SLOW, signal=H1_MACD_SIGNAL)
            if macd_df is not None and not macd_df.empty:
                df[f'{prefix}MACD_LINE'] = macd_df.iloc[:,0]
                df[f'{prefix}MACD_HIST'] = macd_df.iloc[:,1]
                df[f'{prefix}MACD_SIGNAL'] = macd_df.iloc[:,2]
            else:
                df[f'{prefix}MACD_LINE'], df[f'{prefix}MACD_HIST'], df[f'{prefix}MACD_SIGNAL'] = np.nan, np.nan, np.nan
    if not prefix:
        fractal_df = williams_fractals(df['high'], df['low'], n_left=FRACTAL_LOOKBACK, n_right=FRACTAL_LOOKFORWARD)
        df = pd.concat([df, fractal_df], axis=1)
    return df

def get_latest_data_and_indicators(symbol, timeframe_mt5, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, num_bars)
    if rates is None or len(rates) == 0:
        logging.warning(f"Could not fetch {timeframe_to_string(timeframe_mt5)} data for {symbol}.")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time_dt', inplace=True)
    if timeframe_mt5 == ENTRY_TIMEFRAME_MT5:
        df_processed = calculate_indicators_for_df(df.copy(), M5_EMA_SHORT_PERIOD, M5_EMA_MID_PERIOD, M5_EMA_LONG_PERIOD,
                                                 adx_len=ADX_PERIOD, prefix="", atr_len=ATR_PERIOD)
    elif timeframe_mt5 == FILTER_TIMEFRAME_MT5:
        df_processed = calculate_indicators_for_df(df.copy(), H1_EMA_SHORT_PERIOD, H1_EMA_MID_PERIOD, H1_EMA_LONG_PERIOD,
                                                 rsi_len=H1_RSI_PERIOD, prefix="H1_", add_macd=True)
    else: df_processed = df
    return df_processed

def get_account_balance():
    acc_info = mt5.account_info()
    return acc_info.balance if acc_info else None

def get_value_of_one_point(symbol):
    props = SYMBOL_PROPERTIES.get(symbol)
    if not props:
        logging.error(f"Symbol {symbol} not found in SYMBOL_PROPERTIES for point value calculation.")
        return None
    point = props['point']
    tick_info = mt5.symbol_info_tick(symbol)
    if not tick_info:
        logging.error(f"Could not get tick info for {symbol} to calculate point value.")
        return None
    current_price_for_calc = tick_info.ask
    if current_price_for_calc == 0 and tick_info.bid != 0: current_price_for_calc = tick_info.bid
    elif current_price_for_calc == 0 and tick_info.last != 0: current_price_for_calc = tick_info.last
    if current_price_for_calc == 0 :
        logging.error(f"Current price for {symbol} is 0 (ask: {tick_info.ask}, bid: {tick_info.bid}, last: {tick_info.last}). Cannot calculate point value.")
        return None
    profit = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, current_price_for_calc, current_price_for_calc + point)
    if profit is None:
        logging.error(f"mt5.order_calc_profit returned None for {symbol} at price {current_price_for_calc}, point {point}. Error: {mt5.last_error()}")
        profit_10_points = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, current_price_for_calc, current_price_for_calc + (10 * point))
        if profit_10_points is not None:
            profit = profit_10_points / 10.0
            logging.warning(f"Used 10-point deviation for {symbol} pip value calculation. Result per point: {profit}")
        else:
            logging.error(f"mt5.order_calc_profit also failed for 10-point deviation for {symbol}. Error: {mt5.last_error()}")
            return None
    if profit < 0: profit = abs(profit)
    if profit == 0:
        logging.warning(f"Calculated point value is 0 for {symbol} at price {current_price_for_calc} with point size {point} using order_calc_profit.")
    return profit

def calculate_dynamic_lot_size(symbol, sl_pips, account_balance, risk_per_trade_percent):
    props = SYMBOL_PROPERTIES.get(symbol)
    if not props:
        logging.error(f"Symbol {symbol} not found in SYMBOL_PROPERTIES for lot calculation.")
        return 0.01
    if account_balance is None or account_balance <= 0:
        logging.error(f"Invalid account balance ({account_balance}) for lot calculation. Using min lot for {symbol}.")
        return props['volume_min']
    if sl_pips <= 0:
        logging.error(f"Stop loss in pips ({sl_pips}) is zero or negative for {symbol}. Cannot calculate dynamic lot. Using min lot.")
        return props['volume_min']
    amount_to_risk = account_balance * (risk_per_trade_percent / 100.0)
    value_of_one_point = get_value_of_one_point(symbol)
    if value_of_one_point is None or value_of_one_point <= 1e-9:
        logging.error(f"Invalid or zero point value ({value_of_one_point}) for {symbol}. Cannot calculate dynamic lot size. Using min lot.")
        return props['volume_min']
    value_of_sl_pips_one_lot = sl_pips * value_of_one_point
    if value_of_sl_pips_one_lot <= 1e-9:
        logging.error(f"SL pips monetary value ({value_of_sl_pips_one_lot}) is zero or negative for {symbol} (SL pips: {sl_pips}, Point value: {value_of_one_point}). Using min lot.")
        return props['volume_min']
    lot_size = amount_to_risk / value_of_sl_pips_one_lot
    lot_size = max(lot_size, props['volume_min'])
    if props['volume_max'] > 0: lot_size = min(lot_size, props['volume_max'])
    if props['volume_step'] > 0:
        lot_size = math.floor(lot_size / props['volume_step']) * props['volume_step']
        lot_precision = 0
        if props['volume_step'] < 1:
            step_str = format(props['volume_step'], '.8f').rstrip('0')
            if '.' in step_str: lot_precision = len(step_str.split('.')[1])
            else: lot_precision = 0
        lot_size = round(lot_size, lot_precision)
    lot_size = max(lot_size, props['volume_min'])
    if lot_size < props['volume_min']:
        logging.warning(f"Calculated lot_size {lot_size} for {symbol} is below min_volume {props['volume_min']}. Correcting to min_volume.")
        return props['volume_min']
    if props['volume_max'] > 0 and lot_size > props['volume_max']:
        logging.warning(f"Calculated lot_size {lot_size} for {symbol} is above max_volume {props['volume_max']}. Correcting to max_volume.")
        return props['volume_max']
    return lot_size

def place_trade_order(symbol, trade_type_mt5, lot_size, sl_price, tp_price, comment=""):
    props = SYMBOL_PROPERTIES[symbol]
    current_tick = mt5.symbol_info_tick(symbol)
    if not current_tick:
        logging.error(f"Could not get tick for {symbol} to place order.")
        return None, 0
    price = current_tick.ask if trade_type_mt5 == mt5.ORDER_TYPE_BUY else current_tick.bid
    if price == 0:
        logging.error(f"Price for {symbol} is 0 (ask: {current_tick.ask}, bid: {current_tick.bid}). Cannot place order.")
        return None, 0
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size,
        "type": trade_type_mt5, "price": price,
        "sl": round(sl_price, props['digits']), "tp": round(tp_price, props['digits']),
        "deviation": MAX_SLIPPAGE, "magic": MAGIC_NUMBER, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    symbol_info_obj = mt5.symbol_info(symbol)
    if symbol_info_obj and hasattr(symbol_info_obj, 'filling_modes'):
        if mt5.SYMBOL_FILLING_FOK in symbol_info_obj.filling_modes:
             request["type_filling"] = mt5.ORDER_FILLING_FOK
    else: logging.warning(f"Could not get symbol_info for {symbol} to determine best filling mode. Using default IOC.")

    logging.info(f"Order Request: {request}")
    order_result = mt5.order_send(request)
    position_id_from_deal = 0
    if order_result is None:
        logging.error(f"order_send failed for {symbol}, returned None. Last error: {mt5.last_error()}")
        return None, position_id_from_deal
    if order_result.retcode == mt5.TRADE_RETCODE_DONE:
        deal_id = order_result.deal
        if deal_id > 0:
            time.sleep(0.2)
            history_deal_tuple = mt5.history_deals_get(ticket=deal_id)
            if history_deal_tuple and len(history_deal_tuple) > 0:
                position_id_from_deal = history_deal_tuple[0].position_id
                logging.info(f"Order for {symbol} successfully placed. Order Ticket: {order_result.order}, Deal Ticket: {deal_id}, Associated Position ID: {position_id_from_deal}, Price: {order_result.price}, Volume: {order_result.volume}")
            else:
                logging.warning(f"Order for {symbol} placed (Order: {order_result.order}, Deal: {deal_id}), but could not immediately fetch deal history to get PositionID. LastError: {mt5.last_error()}. PositionID will be logged as {order_result.position if hasattr(order_result, 'position') else 'N/A from result'} or re-fetched later.")
                if hasattr(order_result, 'position') and order_result.position > 0 :
                    position_id_from_deal = order_result.position
        else:
            logging.warning(f"Order for {symbol} placed (Order: {order_result.order}), but DealID is 0 (Retcode: {order_result.retcode}, Comment: {order_result.comment}). Cannot determine PositionID from this result directly. Will attempt to log with PositionID {order_result.position if hasattr(order_result, 'position') else '0'}.")
            if hasattr(order_result, 'position') and order_result.position > 0 :
                    position_id_from_deal = order_result.position
        return order_result, position_id_from_deal
    else:
        logging.error(f"Order placement failed for {symbol}. Retcode: {order_result.retcode} ({mt5.trade_retcode_description(order_result.retcode)}), MT5 Error: {mt5.last_error()}, Comment: {order_result.comment}, Request: {order_result.request}")
        return order_result, position_id_from_deal

def modify_position_sltp(ticket, new_sl_price, current_tp_price, symbol):
    props = SYMBOL_PROPERTIES[symbol]
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": round(new_sl_price, props['digits']),
        "tp": round(current_tp_price, props['digits'])
    }
    logging.info(f"Modifying SL/TP for position ticket {ticket} on {symbol} to SL: {new_sl_price:.{props['digits']}f}, TP: {current_tp_price:.{props['digits']}f}")
    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"SL/TP for position ticket {ticket} modified successfully.")
        return True
    else:
        error_code_tuple = mt5.last_error()
        error_code = error_code_tuple[0] if isinstance(error_code_tuple, tuple) else error_code_tuple
        retcode_val = result.retcode if result else 'N/A'
        comment_val = result.comment if result else 'N/A'
        logging.error(f"SL/TP modification failed for position ticket {ticket}. Retcode: {retcode_val} (MT5 Error Code: {error_code}), Comment: {comment_val}, MT5 Error Desc: {mt5.trade_retcode_description(retcode_val) if result else error_code_tuple[1] if isinstance(error_code_tuple, tuple) else 'N/A'}")
        if error_code == 10025:
             logging.error(f"Invalid stops for TSL on {symbol}, T:{ticket}. New SL:{new_sl_price:.{props['digits']}f}, TP:{current_tp_price:.{props['digits']}f}. Current price might be too close or invalid SL/TP level.")
        return False

def get_active_portfolio_trade_by_magic():
    positions = mt5.positions_get(magic=MAGIC_NUMBER)
    if positions is None:
        logging.warning(f"Could not get positions by magic {MAGIC_NUMBER}: {mt5.last_error()}. Assuming no active trade for now.")
        return None
    return positions[0] if len(positions) > 0 else None

def manage_trailing_stop_loss_live():
    active_position = get_active_portfolio_trade_by_magic()
    if not active_position: return

    ticket = active_position.ticket
    symbol = active_position.symbol
    trade_type = active_position.type
    entry_price = active_position.price_open
    current_sl = active_position.sl
    current_tp = active_position.tp
    comment_str = active_position.comment

    initial_sl_pips_risked = 0.0
    trade_row_df = trade_history_df[(trade_history_df['PositionID'] == ticket) & pd.isna(trade_history_df['CloseTimeUTC'])]
    if not trade_row_df.empty:
        open_comment_from_log = trade_row_df.iloc[0]['OpenComment']
        if open_comment_from_log and isinstance(open_comment_from_log, str):
            try:
                if "SL" in open_comment_from_log:
                    parts = open_comment_from_log.split(';')
                    for part in parts:
                        if part.startswith("SL") and not part.startswith("SLPips:"):
                            initial_sl_pips_risked_str = part[2:]
                            initial_sl_pips_risked = float(initial_sl_pips_risked_str)
                            break
                    if initial_sl_pips_risked == 0.0 and "SLPips:" in open_comment_from_log:
                         initial_sl_pips_risked_str = open_comment_from_log.split("SLPips:")[1].split(";")[0]
                         initial_sl_pips_risked = float(initial_sl_pips_risked_str)
            except (IndexError, ValueError, TypeError) as e:
                logging.debug(f"TSL: Could not parse SL pips from logged OpenComment '{open_comment_from_log}' for position {ticket}. Error: {e}.")
    
    if initial_sl_pips_risked == 0.0 and comment_str and isinstance(comment_str, str):
        try:
            if "SL" in comment_str:
                parts = comment_str.split(';')
                for part in parts:
                    if part.startswith("SL") and not part.startswith("SLPips:"):
                        initial_sl_pips_risked_str = part[2:]
                        initial_sl_pips_risked = float(initial_sl_pips_risked_str)
                        break
        except (IndexError, ValueError, TypeError) as e:
            logging.debug(f"TSL: Could not parse SL pips from active position comment '{comment_str}' for position {ticket}. Error: {e}.")

    if initial_sl_pips_risked <= 0:
        logging.warning(f"TSL: Initial SL pips for position {ticket} ({symbol}) is {initial_sl_pips_risked}. TSL cannot proceed.")
        return

    props = SYMBOL_PROPERTIES[symbol]
    point = props['point']; digits = props['digits']
    current_tick = mt5.symbol_info_tick(symbol)
    if not current_tick:
        logging.warning(f"Could not get tick for {symbol} to manage TSL for position {ticket}.")
        return
    
    if current_sl == 0.0:
        logging.warning(f"TSL: Position {ticket} has no SL (current_sl is 0.0). TSL cannot be applied or activated.")
        return

    tsl_considered_active_or_at_be = False
    if trade_type == mt5.ORDER_TYPE_BUY:
        current_market_price_for_profit_check = current_tick.bid
        if current_sl >= entry_price: tsl_considered_active_or_at_be = True
        if not tsl_considered_active_or_at_be:
            activation_profit_pips = initial_sl_pips_risked * TSL_ACTIVATION_RR_RATIO
            activation_target_price = entry_price + (activation_profit_pips * point)
            if current_market_price_for_profit_check >= activation_target_price:
                new_sl_at_be = round(entry_price, digits)
                if new_sl_at_be > current_sl :
                    logging.info(f"TSL Activating for BUY {symbol} (Pos:{ticket}). Moving SL to BE: {new_sl_at_be:.{digits}f}, Keeping TP: {current_tp:.{digits}f}")
                    if modify_position_sltp(ticket, new_sl_at_be, current_tp, symbol):
                        tsl_considered_active_or_at_be = True
        if tsl_considered_active_or_at_be:
            potential_trailed_sl = round(current_market_price_for_profit_check - (initial_sl_pips_risked * point), digits)
            if potential_trailed_sl > current_sl:
                logging.info(f"Trailing SL for BUY {symbol} (Pos:{ticket}). New SL: {potential_trailed_sl:.{digits}f}, Keeping TP: {current_tp:.{digits}f} (Current SL: {current_sl:.{digits}f})")
                modify_position_sltp(ticket, potential_trailed_sl, current_tp, symbol)
    elif trade_type == mt5.ORDER_TYPE_SELL:
        current_market_price_for_profit_check = current_tick.ask
        if current_sl <= entry_price and current_sl != 0.0: tsl_considered_active_or_at_be = True
        if not tsl_considered_active_or_at_be:
            activation_profit_pips = initial_sl_pips_risked * TSL_ACTIVATION_RR_RATIO
            activation_target_price = entry_price - (activation_profit_pips * point)
            if current_market_price_for_profit_check <= activation_target_price:
                new_sl_at_be = round(entry_price, digits)
                if new_sl_at_be < current_sl or current_sl == 0.0 :
                    logging.info(f"TSL Activating for SELL {symbol} (Pos:{ticket}). Moving SL to BE: {new_sl_at_be:.{digits}f}, Keeping TP: {current_tp:.{digits}f}")
                    if modify_position_sltp(ticket, new_sl_at_be, current_tp, symbol):
                         tsl_considered_active_or_at_be = True
        if tsl_considered_active_or_at_be:
            potential_trailed_sl = round(current_market_price_for_profit_check + (initial_sl_pips_risked * point), digits)
            if potential_trailed_sl < current_sl:
                logging.info(f"Trailing SL for SELL {symbol} (Pos:{ticket}). New SL: {potential_trailed_sl:.{digits}f}, Keeping TP: {current_tp:.{digits}f} (Current SL: {current_sl:.{digits}f})")
                modify_position_sltp(ticket, potential_trailed_sl, current_tp, symbol)

def check_for_new_trade_signals():
    global daily_trade_counts, daily_start_balance_utc, last_checked_day_utc_for_reset
    global daily_risk_budget_currency_global, current_daily_risked_amount_global, daily_risk_budget_exceeded_today_global

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    current_day_str = now_utc.strftime('%Y-%m-%d')

    if last_checked_day_utc_for_reset != current_day_str:
        logging.info(f"New UTC Day: {current_day_str}. Resetting daily counters, start balance, and risk budget.")
        daily_trade_counts = {day_sym: 0 for day_sym in SYMBOLS_TO_TRADE}
        current_bal_for_reset = get_account_balance()
        if current_bal_for_reset is not None and current_bal_for_reset > 0:
            daily_start_balance_utc = current_bal_for_reset
            daily_risk_budget_currency_global = daily_start_balance_utc * (DAILY_MAX_ACCOUNT_RISK_PERCENT / 100.0)
            current_daily_risked_amount_global = 0.0
            daily_risk_budget_exceeded_today_global = False
            account_currency = mt5.account_info().currency if mt5.account_info() else ""
            logging.info(f"Daily Risk Budget set to: {daily_risk_budget_currency_global:.2f} {account_currency}")
        else:
            logging.error("Failed to get account balance for daily reset or balance is zero! Daily risk budget is 0.")
            daily_start_balance_utc = 0
            daily_risk_budget_currency_global = 0.0
            current_daily_risked_amount_global = 0.0
            daily_risk_budget_exceeded_today_global = True
        last_checked_day_utc_for_reset = current_day_str

    if daily_risk_budget_exceeded_today_global: return
    if daily_risk_budget_currency_global > 0 and current_daily_risked_amount_global >= daily_risk_budget_currency_global:
        if not daily_risk_budget_exceeded_today_global:
            logging.warning(
                f"Daily max risk budget of {DAILY_MAX_ACCOUNT_RISK_PERCENT:.2f}% reached. "
                f"Budget: {daily_risk_budget_currency_global:.2f}, Risked Today: {current_daily_risked_amount_global:.2f}. "
                f"No new trades will be opened for the rest of UTC day: {current_day_str}."
            )
            daily_risk_budget_exceeded_today_global = True
        return

    if get_active_portfolio_trade_by_magic() is not None: return
    current_hour_utc = now_utc.hour
    if not (TRADING_HOUR_START_UTC <= current_hour_utc < TRADING_HOUR_END_UTC): return

    for symbol in SYMBOLS_TO_TRADE:
        if symbol not in SYMBOL_PROPERTIES:
            logging.warning(f"Symbol {symbol} not in SYMBOL_PROPERTIES. Skipping.")
            continue
        props = SYMBOL_PROPERTIES[symbol]
        in_session = False
        symbol_specific_sessions = SYMBOL_SESSIONS.get(symbol)
        if symbol_specific_sessions:
            for start_h, end_h in symbol_specific_sessions:
                if start_h <= current_hour_utc < end_h:
                    in_session = True; break
            if not in_session: continue
        if daily_trade_counts.get(symbol, 0) >= MAX_TRADES_PER_SYMBOL_PER_DAY: continue

        df_m5 = get_latest_data_and_indicators(symbol, ENTRY_TIMEFRAME_MT5, N_BARS_FOR_INDICATORS + FRACTAL_LOOKFORWARD + 10)
        df_h1 = get_latest_data_and_indicators(symbol, FILTER_TIMEFRAME_MT5, N_BARS_FOR_INDICATORS + 10)
        min_bars_m5 = max(M5_EMA_LONG_PERIOD, (ADX_PERIOD * 2 if ADX_PERIOD else 0), (ATR_PERIOD + ATR_ROLLING_MEAN_PERIOD), FRACTAL_LOOKBACK + FRACTAL_LOOKFORWARD + 2) + 5
        min_bars_h1 = max(H1_EMA_LONG_PERIOD, (H1_MACD_SLOW + H1_MACD_SIGNAL if H1_MACD_SLOW and H1_MACD_SIGNAL else 0)) + 5
        if df_m5.empty or df_h1.empty or len(df_m5) < min_bars_m5 or len(df_h1) < min_bars_h1:
            logging.debug(f"Not enough data for {symbol}. M5 have {len(df_m5)} (need ~{min_bars_m5}), H1 have {len(df_h1)} (need ~{min_bars_h1})")
            continue

        m5_signal_candle = df_m5.iloc[-2]
        m5_prev_signal_candle = df_m5.iloc[-3]
        m5_fractal_ref_idx = -2 - FRACTAL_LOOKFORWARD
        if abs(m5_fractal_ref_idx) >= len(df_m5) :
            logging.debug(f"Not enough bars for fractal calculation at index {m5_fractal_ref_idx} for {symbol} (M5 len: {len(df_m5)})")
            continue
        m5_fractal_candle = df_m5.iloc[m5_fractal_ref_idx]
        h1_aligned_candles = df_h1[df_h1.index <= m5_signal_candle.name]
        if h1_aligned_candles.empty:
            logging.debug(f"No H1 candle found aligned with M5 signal candle time {m5_signal_candle.name} for {symbol}")
            continue
        h1_signal_candle = h1_aligned_candles.iloc[-1]

        adx_val = m5_signal_candle.get(f'ADX_{ADX_PERIOD}')
        if pd.isna(adx_val) or adx_val < ADX_THRESHOLD: continue
        atr_val = m5_signal_candle.get(f'ATR_{ATR_PERIOD}')
        average_atr_val = m5_signal_candle.get(f'ATR_{ATR_PERIOD}_SMA{ATR_ROLLING_MEAN_PERIOD}')
        if pd.isna(atr_val) or pd.isna(average_atr_val) or average_atr_val == 0: continue
        if not (average_atr_val * ATR_MULTIPLIER_LOW <= atr_val <= average_atr_val * ATR_MULTIPLIER_HIGH): continue

        h1_ema_short = h1_signal_candle.get(f'H1_EMA_{H1_EMA_SHORT_PERIOD}')
        h1_ema_mid = h1_signal_candle.get(f'H1_EMA_{H1_EMA_MID_PERIOD}')
        h1_ema_long = h1_signal_candle.get(f'H1_EMA_{H1_EMA_LONG_PERIOD}')
        h1_rsi_val = h1_signal_candle.get(f'H1_RSI_{H1_RSI_PERIOD}')
        h1_macd_hist = h1_signal_candle.get("H1_MACD_HIST")
        if any(pd.isna(v) for v in [h1_ema_short, h1_ema_mid, h1_ema_long, h1_rsi_val, h1_macd_hist]): continue
        h1_is_uptrend = h1_ema_short > h1_ema_mid > h1_ema_long
        h1_is_downtrend = h1_ema_long > h1_ema_mid > h1_ema_short
        h1_filter_bullish = h1_is_uptrend and h1_rsi_val > H1_RSI_BULL_THRESHOLD and h1_macd_hist > 0
        h1_filter_bearish = h1_is_downtrend and h1_rsi_val < H1_RSI_BEAR_THRESHOLD and h1_macd_hist < 0

        m5_ema_short_val = m5_signal_candle.get(f'EMA_{M5_EMA_SHORT_PERIOD}')
        m5_ema_mid_val = m5_signal_candle.get(f'EMA_{M5_EMA_MID_PERIOD}')
        m5_ema_long_val = m5_signal_candle.get(f'EMA_{M5_EMA_LONG_PERIOD}')
        m5_close_signal = m5_signal_candle['close']
        m5_low_signal = m5_signal_candle['low']; m5_high_signal = m5_signal_candle['high']
        if any(pd.isna(v) for v in [m5_ema_short_val, m5_ema_mid_val, m5_ema_long_val]): continue
        m5_buy_frac = m5_fractal_candle.get('fractal_down', False)
        m5_sell_frac = m5_fractal_candle.get('fractal_up', False)
        order_type_to_place = None; sl_reference_price = 0.0

        m5_is_uptrend_strong = m5_ema_short_val > m5_ema_mid_val > m5_ema_long_val
        m5_close_above_short_ema = m5_close_signal > m5_ema_short_val
        m5_prev_close_val = m5_prev_signal_candle['close']
        m5_prev_ema_short_val = m5_prev_signal_candle.get(f'EMA_{M5_EMA_SHORT_PERIOD}')
        if pd.isna(m5_prev_ema_short_val): continue
        m5_prev_close_below_short_ema = m5_prev_close_val < m5_prev_ema_short_val
        m5_curr_low_below_short_ema = m5_low_signal < m5_ema_short_val
        m5_pullback_long_condition = m5_prev_close_below_short_ema or m5_curr_low_below_short_ema
        m5_close_above_long_ema_crit = m5_close_signal > m5_ema_long_val

        if m5_is_uptrend_strong and m5_pullback_long_condition and m5_close_above_short_ema and m5_buy_frac and m5_close_above_long_ema_crit and h1_filter_bullish:
            order_type_to_place = mt5.ORDER_TYPE_BUY
            sl_candidate1 = m5_ema_mid_val
            sl_candidate2 = m5_ema_long_val
            sl_candidate3_fractal_low = m5_fractal_candle['low']
            if m5_low_signal >= m5_ema_mid_val : sl_reference_price = min(sl_candidate1, sl_candidate3_fractal_low)
            else: sl_reference_price = min(sl_candidate2, sl_candidate3_fractal_low)
            logging.info(f"BUY Signal Conditions Met for {symbol} at {m5_signal_candle.name}")
            logging.debug(f"BUY Details - M5 EMAs (S/M/L): {m5_ema_short_val:.{props['digits']}f}/{m5_ema_mid_val:.{props['digits']}f}/{m5_ema_long_val:.{props['digits']}f}. Fractal Low: {sl_candidate3_fractal_low:.{props['digits']}f}. SL Ref: {sl_reference_price:.{props['digits']}f}")

        if order_type_to_place is None:
            m5_is_downtrend_strong = m5_ema_long_val > m5_ema_mid_val > m5_ema_short_val
            m5_close_below_short_ema_sell = m5_close_signal < m5_ema_short_val
            m5_prev_close_above_short_ema_sell = m5_prev_close_val > m5_prev_ema_short_val
            m5_curr_high_above_short_ema_sell = m5_high_signal > m5_ema_short_val
            m5_pullback_short_condition = m5_prev_close_above_short_ema_sell or m5_curr_high_above_short_ema_sell
            m5_close_below_long_ema_crit = m5_close_signal < m5_ema_long_val
            if m5_is_downtrend_strong and m5_pullback_short_condition and m5_close_below_short_ema_sell and m5_sell_frac and m5_close_below_long_ema_crit and h1_filter_bearish:
                order_type_to_place = mt5.ORDER_TYPE_SELL
                sl_candidate1_sell = m5_ema_mid_val
                sl_candidate2_sell = m5_ema_long_val
                sl_candidate3_fractal_high = m5_fractal_candle['high']
                if m5_high_signal <= m5_ema_mid_val: sl_reference_price = max(sl_candidate1_sell, sl_candidate3_fractal_high)
                else: sl_reference_price = max(sl_candidate2_sell, sl_candidate3_fractal_high)
                logging.info(f"SELL Signal Conditions Met for {symbol} at {m5_signal_candle.name}")
                logging.debug(f"SELL Details - M5 EMAs (S/M/L): {m5_ema_short_val:.{props['digits']}f}/{m5_ema_mid_val:.{props['digits']}f}/{m5_ema_long_val:.{props['digits']}f}. Fractal High: {sl_candidate3_fractal_high:.{props['digits']}f}. SL Ref: {sl_reference_price:.{props['digits']}f}")

        if order_type_to_place is not None:
            logging.info(f"CONFIRMED TRADE SIGNAL: {'BUY' if order_type_to_place == mt5.ORDER_TYPE_BUY else 'SELL'} for {symbol} at {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            point = props['point']; digits = props['digits']
            current_tick_data = mt5.symbol_info_tick(symbol)
            if not current_tick_data:
                logging.warning(f"Could not get current tick for {symbol} to finalize trade. Skipping.")
                continue
            entry_price_now = current_tick_data.ask if order_type_to_place == mt5.ORDER_TYPE_BUY else current_tick_data.bid
            if entry_price_now == 0:
                 logging.warning(f"Entry price for {symbol} is 0. Skipping trade.")
                 continue
            sl_pips_calculated = 0.0
            if order_type_to_place == mt5.ORDER_TYPE_BUY:
                if sl_reference_price < entry_price_now: sl_pips_calculated = (entry_price_now - sl_reference_price) / point if point > 0 else 0
                else: sl_pips_calculated = 0
            else:
                if sl_reference_price > entry_price_now: sl_pips_calculated = (sl_reference_price - entry_price_now) / point if point > 0 else 0
                else: sl_pips_calculated = 0
            if sl_pips_calculated == 0 and atr_val > 0 and point > 0 :
                logging.warning(f"Primary SL calculation for {symbol} resulted in 0 pips (Entry: {entry_price_now}, SL Ref: {sl_reference_price}). Using ATR-based SL ({atr_val/point:.1f} pips) as fallback.")
                sl_pips_calculated = atr_val / point
            min_sl_pips_atr_factor = 1.0; min_sl_pips_absolute = 10.0
            if "JPY" in symbol.upper(): min_sl_pips_absolute = 10.0
            elif any(metal in symbol.upper() for metal in ["XAU", "XAG", "XPT"]): min_sl_pips_absolute = 100.0
            elif "OIL" in symbol.upper() : min_sl_pips_absolute = 100.0
            min_sl_pips_from_atr = (atr_val / point * min_sl_pips_atr_factor) if point > 0 and atr_val > 0 else min_sl_pips_absolute
            sl_pips_final = max(sl_pips_calculated, min_sl_pips_from_atr, min_sl_pips_absolute)
            if sl_pips_final <= 0:
                logging.warning(f"{symbol} final SL pips is {sl_pips_final:.2f}. Invalid SL. Skipping trade.")
                continue
            sl_price_final, tp_price_final = (0.0,0.0)
            if order_type_to_place == mt5.ORDER_TYPE_BUY:
                sl_price_final = entry_price_now - (sl_pips_final * point)
                tp_price_final = entry_price_now + (sl_pips_final * RISK_REWARD_RATIO * point)
            else:
                sl_price_final = entry_price_now + (sl_pips_final * point)
                tp_price_final = entry_price_now - (sl_pips_final * RISK_REWARD_RATIO * point)
            if (order_type_to_place == mt5.ORDER_TYPE_BUY and (sl_price_final >= entry_price_now or tp_price_final <= entry_price_now)) or \
               (order_type_to_place == mt5.ORDER_TYPE_SELL and (sl_price_final <= entry_price_now or tp_price_final >= entry_price_now)):
                logging.warning(f"Invalid SL/TP for {symbol}: E:{entry_price_now:.{digits}f}, SL:{sl_price_final:.{digits}f}, TP:{tp_price_final:.{digits}f}. SL_pips: {sl_pips_final:.1f}. Skipping.")
                continue
            current_account_balance = get_account_balance()
            if current_account_balance is None:
                logging.error(f"Could not get account balance for {symbol} before lot calculation. Skipping trade.")
                continue
            lot_size_final = calculate_dynamic_lot_size(symbol, sl_pips_final, current_account_balance, RISK_PER_TRADE_ACCOUNT_PERCENT)
            if lot_size_final < props['volume_min'] or lot_size_final == 0:
                 logging.warning(f"Calculated lot size {lot_size_final} for {symbol} is too small or zero. Using min volume {props['volume_min']}.")
                 lot_size_final = props['volume_min']
            if lot_size_final == 0:
                logging.error(f"Final lot size for {symbol} is 0. Cannot place trade.")
                continue
            value_of_one_point_trade = get_value_of_one_point(symbol)
            if value_of_one_point_trade is None or value_of_one_point_trade <= 0:
                logging.error(f"Could not determine valid point value for {symbol} to calculate risk amount. Skipping trade.")
                continue
            risked_amount_this_trade = lot_size_final * sl_pips_final * value_of_one_point_trade
            if daily_risk_budget_currency_global > 0 and \
               (current_daily_risked_amount_global + risked_amount_this_trade > daily_risk_budget_currency_global):
                logging.info(f"Potential trade for {symbol} (risk: {risked_amount_this_trade:.2f}) would exceed daily risk budget "
                             f"({daily_risk_budget_currency_global:.2f}). "
                             f"Currently risked: {current_daily_risked_amount_global:.2f}. Skipping trade.")
                continue
            bot_version_comment = "V1.4.2" # Max DD calc version
            sl_pips_comment_str = f"{sl_pips_final:.0f}"
            rr_comment_str = f"{RISK_REWARD_RATIO:.0f}" if RISK_REWARD_RATIO == int(RISK_REWARD_RATIO) else f"{RISK_REWARD_RATIO:.1f}"
            lot_precision_comment = 0
            if props['volume_step'] > 0 and props['volume_step'] < 1:
                step_str_comm = format(props['volume_step'], '.8f').rstrip('0')
                if '.' in step_str_comm: lot_precision_comment = len(step_str_comm.split('.')[1])
            lot_comment_str = f"{lot_size_final:.{lot_precision_comment}f}"
            trade_comment_str = f"{bot_version_comment};SL{sl_pips_comment_str};R{rr_comment_str};L{lot_comment_str}"
            MAX_COMMENT_LENGTH = 31
            if len(trade_comment_str) > MAX_COMMENT_LENGTH:
                trade_comment_str = f"SL{sl_pips_comment_str};R{rr_comment_str};L{lot_comment_str}"
                if len(trade_comment_str) > MAX_COMMENT_LENGTH:
                     trade_comment_str = f"SL{sl_pips_comment_str};R{rr_comment_str}"
                     if len(trade_comment_str) > MAX_COMMENT_LENGTH: trade_comment_str = trade_comment_str[:MAX_COMMENT_LENGTH]
                logging.warning(f"Trade comment was too long, truncated to: '{trade_comment_str}'")
            logging.info(f"Attempting to place {'BUY' if order_type_to_place == mt5.ORDER_TYPE_BUY else 'SELL'} order for {symbol}: "
                         f"Lots: {lot_size_final}, Entry: {entry_price_now:.{digits}f}, SL: {sl_price_final:.{digits}f} ({sl_pips_final:.1f} pips), "
                         f"TP: {tp_price_final:.{digits}f}, Risk: {risked_amount_this_trade:.2f}, Comment: '{trade_comment_str}'")
            order_placement_result, position_id_from_order = place_trade_order(
                symbol, order_type_to_place, lot_size_final, sl_price_final, tp_price_final, comment=trade_comment_str
            )
            if order_placement_result and order_placement_result.retcode == mt5.TRADE_RETCODE_DONE:
                log_opened_trade(order_placement_result, symbol, order_type_to_place, lot_size_final,
                                 sl_price_final, tp_price_final, trade_comment_str, risked_amount_this_trade,
                                 position_id_override=position_id_from_order)
                daily_trade_counts[symbol] = daily_trade_counts.get(symbol, 0) + 1
                current_daily_risked_amount_global += risked_amount_this_trade
                logging.info(f"Trade execution successful for {symbol}. Position ID: {position_id_from_order}. Order Ticket: {order_placement_result.order}. "
                             f"Daily count for {symbol}: {daily_trade_counts[symbol]}. "
                             f"Total daily risked: {current_daily_risked_amount_global:.2f}/{daily_risk_budget_currency_global:.2f}.")
                return
            else:
                retcode_info = f"Retcode: {order_placement_result.retcode}" if order_placement_result else "Result: None"
                logging.error(f"Failed to execute trade for {symbol} despite signal. {retcode_info}. See order request/result logs for details.")

def main_bot_loop():
    logging.info("Initializing Trading Bot...")
    global last_checked_day_utc_for_reset, daily_start_balance_utc, daily_trade_counts, SYMBOLS_TO_TRADE
    global daily_risk_budget_currency_global, current_daily_risked_amount_global, daily_risk_budget_exceeded_today_global

    if not initialize_mt5_for_live():
        logging.critical("Failed to initialize MT5. Bot cannot start.")
        return

    load_or_initialize_trade_history()
    check_and_log_closed_trades() # Log any trades closed while bot was offline, this will also save history once.

    now_utc_init = datetime.datetime.now(datetime.timezone.utc)
    last_checked_day_utc_for_reset = now_utc_init.strftime('%Y-%m-%d')
    daily_trade_counts = {symbol: 0 for symbol in SYMBOLS_TO_TRADE}
    initial_bal = get_account_balance()
    account_currency_main = mt5.account_info().currency if mt5.account_info() else ""

    if initial_bal is not None and initial_bal > 0:
        daily_start_balance_utc = initial_bal
        daily_risk_budget_currency_global = daily_start_balance_utc * (DAILY_MAX_ACCOUNT_RISK_PERCENT / 100.0)
        current_daily_risked_amount_global = 0.0
        daily_risk_budget_exceeded_today_global = False
        logging.info(f"Bot Started. Initial Daily Start Balance: {daily_start_balance_utc:.2f} {account_currency_main}. "
                     f"Daily Risk Budget: {daily_risk_budget_currency_global:.2f} {account_currency_main}.")
    else:
        logging.critical("CRITICAL: Could not get initial account balance at bot start or balance is zero! Daily risk budget is 0.")
        daily_start_balance_utc = 0
        daily_risk_budget_currency_global = 0.0
        current_daily_risked_amount_global = 0.0
        daily_risk_budget_exceeded_today_global = True

    last_signal_check_minute = -1
    last_closed_trade_check_time = time.time()

    logging.info("Bot Initialized. Starting Trading Loop...")
    try:
        while True:
            current_time_utc = datetime.datetime.now(datetime.timezone.utc)
            manage_trailing_stop_loss_live()
            if time.time() - last_closed_trade_check_time > 30:
                check_and_log_closed_trades() # This implicitly calls save_trade_history if trades were closed
                last_closed_trade_check_time = time.time()
            if current_time_utc.minute % 5 == 0 and current_time_utc.minute != last_signal_check_minute:
                if current_time_utc.second < 15 :
                    logging.info(f"--- Scheduled Signal Check at {current_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
                    check_for_new_trade_signals() # This implicitly calls save_trade_history if a trade is opened
                    last_signal_check_minute = current_time_utc.minute
            elif current_time_utc.minute != last_signal_check_minute :
                 last_signal_check_minute = -1
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Bot shutdown requested by user (Ctrl+C).")
    except Exception as e:
        logging.exception("Critical error in main trading loop:")
    finally:
        logging.info("Bot shutting down. Performing final check for closed trades...")
        check_and_log_closed_trades() # Final check and save
        active_trade = get_active_portfolio_trade_by_magic()
        if active_trade:
            logging.info(f"Bot shutting down. Active trade: Position {active_trade.ticket} on {active_trade.symbol}")
        else:
            logging.info("Bot shutting down. No active trades managed by this bot.")
        shutdown_mt5()
        logging.info("Bot shutdown complete.")

if __name__ == "__main__":
    main_bot_loop()