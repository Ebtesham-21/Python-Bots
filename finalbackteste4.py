import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, time
import pytz # For timezone handling
import math # For rounding steps
import decimal # For accurate rounding

# --- Strategy Parameters ---
STRATEGY_NAME = "BoringStrategy_v1.10_OrigPullback_VolFilter" # Updated name
SYMBOLS = ["EURUSD", "USDJPY"] # Keep optimized symbols
TIMEFRAME_STR = "M15"

# --- Trend Filter (Entry Timeframe) ---
EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200

# --- Higher Timeframe Filter ---
HTF_TIMEFRAME_STR = "H1"
HTF_TIMEFRAME_MT5 = None
HTF_EMA_FAST_PERIOD = 50
HTF_EMA_SLOW_PERIOD = 200

# --- ADX Trend Strength Filter (Entry Timeframe) ---
ADX_PERIOD = 14
ADX_THRESHOLD = 0 # ADX disabled

# --- Risk Management ---
RISK_PER_TRADE_PERCENT = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5 # Keep 1.5 ATR SL
TAKE_PROFIT_RR_RATIO = 3.0   # Keep 2.0 RR
ATR_PERIOD = 14

# --- Volatility Filter Parameters ---
ATR_MA_PERIOD = 20
LOW_VOL_THRESHOLD_MULT = 0.7
HIGH_VOL_THRESHOLD_MULT = 2.5

# --- Daily Risk Controls ---
MAX_LOSING_TRADES_PER_DAY = 3
MAX_DAILY_LOSS_PERCENT = 3.0
MAX_DAILY_PROFIT_PERCENT = 5.0

# --- Session Filter ---
SESSION_START_UTC_HOUR = 12 # Back to original session
SESSION_END_UTC_HOUR = 18
UTC = pytz.utc

# --- Backtest Period ---
START_DATE = datetime(2025, 1, 1, tzinfo=UTC)
END_DATE = datetime(2025, 4, 30, tzinfo=UTC)

# --- Initial Account Settings ---
INITIAL_BALANCE = 200.0

# --- Global Variables ---
account_balance = INITIAL_BALANCE
trades = []
active_trades = {}
daily_trade_stats = {}
TIMEFRAME_MT5 = None
HTF_TIMEFRAME_MT5 = None

# --- Timeframe Mapping ---
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
}

# --- Helper Functions ---
def initialize_mt5():
    """Initializes the MetaTrader 5 connection and sets timeframe constants."""
    global TIMEFRAME_MT5, HTF_TIMEFRAME_MT5
    term_info = mt5.terminal_info()
    if term_info:
        print("MT5 already initialized.")
        if TIMEFRAME_MT5 is None: TIMEFRAME_MT5 = TIMEFRAME_MAP.get(TIMEFRAME_STR)
        if HTF_TIMEFRAME_MT5 is None: HTF_TIMEFRAME_MT5 = TIMEFRAME_MAP.get(HTF_TIMEFRAME_STR)
        if TIMEFRAME_MT5 is None or HTF_TIMEFRAME_MT5 is None:
            print("Error: Invalid timeframe string(s) in config.")
            return False
        return True
    if not mt5.initialize(): print("initialize() failed, error code =", mt5.last_error()); return False
    print("MetaTrader 5 Initialized Successfully")
    acct_info = mt5.account_info()
    if acct_info:
        print(f"Account: {acct_info.login}, Server: {acct_info.server}, Balance: {acct_info.balance} {acct_info.currency}")
        TIMEFRAME_MT5 = TIMEFRAME_MAP.get(TIMEFRAME_STR)
        HTF_TIMEFRAME_MT5 = TIMEFRAME_MAP.get(HTF_TIMEFRAME_STR)
        if TIMEFRAME_MT5 is None or HTF_TIMEFRAME_MT5 is None:
            print("Error: Invalid timeframe string(s) in config.")
            mt5.shutdown(); return False
        print(f"Using Entry Timeframe: {TIMEFRAME_STR}, HTF Timeframe: {HTF_TIMEFRAME_STR}")
        return True
    else: print("Could not get account info. Error:", mt5.last_error()); mt5.shutdown(); return False

def shutdown_mt5():
    """Shuts down the MetaTrader 5 connection."""
    mt5.shutdown()
    print("MetaTrader 5 Shutdown")

def get_symbol_info(symbol):
    """Gets necessary symbol information."""
    info = mt5.symbol_info(symbol)
    if info is None: print(f"Symbol {symbol} not found, error code =", mt5.last_error()); return None
    if not info.visible:
        print(f"Symbol {symbol} not visible, enabling...")
        if not mt5.symbol_select(symbol, True): print(f"Failed to enable {symbol}, error code =", mt5.last_error()); return None
        info = mt5.symbol_info(symbol)
        if info is None: print(f"Failed to re-fetch info for {symbol} after enabling."); return None
    return info

def get_historical_data(symbol, timeframe_mt5, timeframe_str, start_dt, end_dt, calculate_emas=True, calculate_atr=False, calculate_adx=False):
    """Fetches historical data and calculates requested indicators."""
    if timeframe_mt5 is None: print(f"Error: MT5 TF constant not set for {timeframe_str}."); return None
    print(f"Fetching data for {symbol} ({timeframe_str}) from {start_dt} to {end_dt}...")
    periods = [EMA_SLOW_PERIOD, HTF_EMA_SLOW_PERIOD]
    if calculate_atr: periods.append(ATR_PERIOD)
    if calculate_adx: periods.append(ADX_PERIOD)
    if calculate_atr: periods.append(ATR_MA_PERIOD)
    required_warmup = max(periods) * 2 if periods else 200
    tf_int = timeframe_mt5 if isinstance(timeframe_mt5, int) else 0
    tf_d1_int = mt5.TIMEFRAME_D1 if isinstance(mt5.TIMEFRAME_D1, int) else 16408
    days_multiplier = 1 if tf_int < tf_d1_int else 1.5
    start_fetch_dt = start_dt - timedelta(days=required_warmup * days_multiplier)
    rates = mt5.copy_rates_range(symbol, timeframe_mt5, start_fetch_dt, end_dt)
    if rates is None or len(rates) == 0: print(f"No data returned for {symbol} ({timeframe_str}). Error: {mt5.last_error()}"); return None
    df = pd.DataFrame(rates)
    try: df['time'] = pd.to_datetime(df['time'], unit='s', utc=True); df.set_index('time', inplace=True)
    except Exception as e: print(f"Error converting timestamps for {symbol} ({timeframe_str}): {e}"); return None
    print(f"Data fetched for {symbol} ({timeframe_str}): {len(df)} bars (includes warmup)")
    try:
        if calculate_emas:
            df.ta.ema(length=EMA_FAST_PERIOD, append=True, col_names=(f'EMA_{EMA_FAST_PERIOD}',))
            df.ta.ema(length=EMA_SLOW_PERIOD, append=True, col_names=(f'EMA_{EMA_SLOW_PERIOD}',))
        if calculate_atr:
            atr_col_name = f'ATR_{ATR_PERIOD}'
            df.ta.atr(length=ATR_PERIOD, append=True, col_names=(atr_col_name,))
            atr_ma_col_name = f'ATR_MA_{ATR_MA_PERIOD}'
            df.ta.sma(close=atr_col_name, length=ATR_MA_PERIOD, append=True, col_names=(atr_ma_col_name,))
        if calculate_adx: df.ta.adx(length=ADX_PERIOD, append=True, col_names=(f'ADX_{ADX_PERIOD}', f'DMP_{ADX_PERIOD}', f'DMN_{ADX_PERIOD}'))
    except Exception as e: print(f"Error calculating indicators for {symbol} ({timeframe_str}): {e}"); pass
    df.dropna(inplace=True)
    start_dt_aware = start_dt if start_dt.tzinfo else pytz.utc.localize(start_dt)
    end_dt_aware = end_dt if end_dt.tzinfo else pytz.utc.localize(end_dt)
    df = df[(df.index >= start_dt_aware) & (df.index <= end_dt_aware)]
    print(f"Data after indicator calculation & date filter for {symbol} ({timeframe_str}): {len(df)} bars")
    return df

def check_session(current_time_utc):
    """Checks if the current time is within the allowed trading session."""
    return SESSION_START_UTC_HOUR <= current_time_utc.hour < SESSION_END_UTC_HOUR

def calculate_position_size(symbol_info, balance, risk_percent, sl_points_abs):
    """Calculates position size based on risk percentage and SL points."""
    if sl_points_abs <= 0: print(f"Warning: SL points <= 0 ({sl_points_abs}). Cannot size."); return 0.0
    required_attrs = ['point', 'trade_contract_size', 'trade_tick_value', 'trade_tick_size', 'volume_step', 'volume_min', 'volume_max', 'digits', 'name']
    for attr in required_attrs:
        if not hasattr(symbol_info, attr): print(f"Error: SymbolInfo missing '{attr}'. Cannot size."); return 0.0

    point=symbol_info.point; contract_size=symbol_info.trade_contract_size; trade_tick_value=symbol_info.trade_tick_value; trade_tick_size=symbol_info.trade_tick_size
    volume_step=symbol_info.volume_step; min_volume=symbol_info.volume_min; max_volume=symbol_info.volume_max; symbol_digits=symbol_info.digits

    value_per_point_per_lot = 0.0
    if point <= 0: print(f"Error: Symbol point size <= 0 ({point})."); return 0.0
    if trade_tick_value > 0 and trade_tick_size > 0:
        points_in_tick = trade_tick_size / point
        if points_in_tick > 0: value_per_point_per_lot = trade_tick_value / points_in_tick
        else: print(f"Error: points_in_tick <= 0."); return 0.0
    elif trade_tick_value <= 0 or trade_tick_size <=0:
        print(f"Warning: Using fallback value per point for {symbol_info.name}.")
        value_per_point_per_lot = contract_size * point
    if value_per_point_per_lot <= 0: print(f"Error: value_per_point_per_lot <= 0."); return 0.0

    risk_amount = balance * (risk_percent / 100.0)
    risk_value_total_per_lot = sl_points_abs * value_per_point_per_lot
    if risk_value_total_per_lot <= 0: print(f"Warning: risk_value_total_per_lot <= 0."); return 0.0

    volume = risk_amount / risk_value_total_per_lot

    if not (isinstance(min_volume, (int, float)) and isinstance(max_volume, (int, float))): print(f"Error: Invalid min/max volume."); return 0.0
    if volume < min_volume:
        min_vol_risk = min_volume * risk_value_total_per_lot
        if min_vol_risk > risk_amount * 1.100001: print(f"Warning: Min volume risk (${min_vol_risk:.2f}) > target risk (${risk_amount:.2f}). Skipping."); return 0.0
        else: print(f"Info: Volume < min. Using min volume. Actual risk: ${min_vol_risk:.2f}"); volume = min_volume
    elif volume > max_volume: print(f"Info: Volume > max. Using max volume."); volume = max_volume

    if not isinstance(volume_step, (int, float)): print(f"Error: Invalid volume step type."); return 0.0
    if volume_step > 0:
        volume = float(decimal.Decimal(str(volume)).quantize(decimal.Decimal(str(volume_step)), rounding=decimal.ROUND_FLOOR))
        if volume < min_volume: print(f"Info: Vol after step < min. Using min."); volume = min_volume
    elif volume_step == 0: print(f"Warning: Volume step is zero."); volume = round(volume, 8)
    else: print(f"Error: Negative volume step."); return 0.0

    if volume < (min_volume - 1e-9) or volume <= 1e-9:
         print(f"Error: Final volume invalid ({volume:.8f}, Min: {min_volume}). Cannot place trade.")
         return 0.0

    if volume_step > 0:
        try:
            step_decimal = decimal.Decimal(str(volume_step))
            step_decimals = abs(step_decimal.as_tuple().exponent) if step_decimal.is_finite() and step_decimal.as_tuple().exponent < 0 else 0
            volume_str = str(volume)
            volume = float(decimal.Decimal(volume_str).quantize(decimal.Decimal('1e-' + str(step_decimals)), rounding=decimal.ROUND_DOWN))
        except Exception as e:
            print(f"Warning: Decimals error during rounding {e}. Using round(8).")
            volume = round(volume, 8)
    else:
        volume = round(volume, 8)

    if volume < (min_volume - 1e-9) or volume <= 1e-9:
        print(f"Error: Final volume invalid after rounding ({volume:.8f}, Min: {min_volume}). Cannot place trade.")
        return 0.0
    return volume

def simulate_trade_outcome(entry_price, sl_price, tp_price, direction, symbol_info, volume, bars_df, entry_index):
    """Simulates if the trade hit SL or TP by checking subsequent bars."""
    required_attrs = ['point', 'trade_contract_size', 'trade_tick_value', 'trade_tick_size', 'digits', 'name']
    for attr in required_attrs:
        if not hasattr(symbol_info, attr):
            symbol_name = getattr(symbol_info, 'name', 'UNK'); print(f"Error sim: {symbol_name} missing '{attr}'.");
            entry_time = bars_df.index[entry_index] if entry_index < len(bars_df.index) else pd.Timestamp.now(tz=UTC)
            return "ERROR_SIM", entry_price, entry_time, 0.0

    point=symbol_info.point; contract_size=symbol_info.trade_contract_size; trade_tick_value=symbol_info.trade_tick_value; trade_tick_size=symbol_info.trade_tick_size; symbol_digits=symbol_info.digits
    pnl = 0.0; exit_price = None; exit_time = None; outcome = "UNKNOWN"
    value_per_point_per_lot = 0.0
    if point <= 0: print(f"Error sim: point <= 0.")
    elif trade_tick_value > 0 and trade_tick_size > 0:
        points_in_tick = trade_tick_size / point
        if points_in_tick > 0: value_per_point_per_lot = trade_tick_value / points_in_tick
        else: print(f"Error sim: points_in_tick <= 0.")
    elif trade_tick_value <= 0 or trade_tick_size <=0:
        print(f"Warning sim: Using fallback value per point."); value_per_point_per_lot = contract_size * point
    if value_per_point_per_lot <= 0: print(f"Error sim: value_per_point_per_lot <= 0.")

    for i in range(entry_index + 1, len(bars_df)):
        if i >= len(bars_df): print(f"Warning sim: index out of bounds."); break
        try: current_bar = bars_df.iloc[i]; high = current_bar['high']; low = current_bar['low']; exit_time = current_bar.name
        except IndexError: print(f"Error sim: index {i}."); outcome = "ERROR_SIM_INDEX"; break

        pnl_points = 0
        if direction == "LONG":
            if low <= sl_price:
                exit_price = sl_price; outcome = "SL_HIT"
                if point != 0: pnl_points = (exit_price - entry_price) / point; pnl = pnl_points * value_per_point_per_lot * volume
                break
            elif high >= tp_price:
                exit_price = tp_price; outcome = "TP_HIT"
                if point != 0: pnl_points = (exit_price - entry_price) / point; pnl = pnl_points * value_per_point_per_lot * volume
                break
        elif direction == "SHORT":
            if high >= sl_price:
                exit_price = sl_price; outcome = "SL_HIT"
                if point != 0: pnl_points = (entry_price - exit_price) / point; pnl = pnl_points * value_per_point_per_lot * volume
                break
            elif low <= tp_price:
                exit_price = tp_price; outcome = "TP_HIT"
                if point != 0: pnl_points = (entry_price - exit_price) / point; pnl = pnl_points * value_per_point_per_lot * volume
                break

    if outcome == "UNKNOWN":
        if not bars_df.empty and entry_index < len(bars_df) -1 :
             last_bar = bars_df.iloc[-1]; exit_price = last_bar['close']; exit_time = last_bar.name; pnl_points = 0
             if point != 0:
                  if direction == "LONG": pnl_points = (exit_price - entry_price) / point
                  elif direction == "SHORT": pnl_points = (entry_price - exit_price) / point
             pnl = pnl_points * value_per_point_per_lot * volume; outcome = "END_OF_DATA"
        elif not bars_df.empty and entry_index >= len(bars_df) -1 :
             outcome = "NO_DATA_AFTER_ENTRY"; pnl = 0; exit_price = entry_price
             exit_time = bars_df.index[entry_index] if entry_index < len(bars_df) else pd.Timestamp.now(tz=UTC)
        else:
             outcome = "ERROR_NO_DATA_FOR_EXIT"; pnl = 0; exit_price = entry_price
             exit_time = pd.Timestamp.now(tz=UTC)

    if exit_price is None: exit_price = entry_price
    if exit_time is None:
         exit_time = bars_df.index[entry_index] if entry_index >= 0 and entry_index < len(bars_df.index) else pd.Timestamp.now(tz=UTC)
    if isinstance(exit_time, pd.Timestamp) and exit_time.tzinfo is None: exit_time = exit_time.tz_localize(UTC)
    elif isinstance(exit_time, datetime) and exit_time.tzinfo is None: exit_time = UTC.localize(exit_time)
    pnl = round(pnl, 2)
    return outcome, exit_price, exit_time, pnl

# --- Main Backtesting Function (Vol Filter & Orig Pullback) ---
def run_backtest():
    global account_balance, trades, active_trades, daily_trade_stats, INITIAL_BALANCE, START_DATE, END_DATE, SYMBOLS, TIMEFRAME_STR, HTF_TIMEFRAME_STR, ADX_THRESHOLD, STOP_LOSS_ATR_MULTIPLIER, TAKE_PROFIT_RR_RATIO, SESSION_START_UTC_HOUR, SESSION_END_UTC_HOUR

    account_balance = INITIAL_BALANCE; trades = []; active_trades = {}; daily_trade_stats = {}
    if TIMEFRAME_MT5 is None or HTF_TIMEFRAME_MT5 is None: print("Error: Timeframes not set."); return None

    print(f"\n--- Starting Backtest: {STRATEGY_NAME} ---")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}, Symbols: {', '.join(SYMBOLS)}")
    print(f"Settings: Entry={TIMEFRAME_STR}, HTF={HTF_TIMEFRAME_STR}, ADX>{ADX_THRESHOLD}, SL={STOP_LOSS_ATR_MULTIPLIER}*ATR, RR={TAKE_PROFIT_RR_RATIO}")
    print(f"Initial Balance: ${INITIAL_BALANCE:.2f}, Risk: {RISK_PER_TRADE_PERCENT:.2f}%, Session: {SESSION_START_UTC_HOUR}-{SESSION_END_UTC_HOUR} UTC")
    print(f"Volatility Filter: ATR < {LOW_VOL_THRESHOLD_MULT}*ATR_MA or ATR > {HIGH_VOL_THRESHOLD_MULT}*ATR_MA")
    print("-" * 50)

    all_symbol_data = {}; symbol_infos = {}
    print("Fetching data and calculating indicators...")
    for symbol in SYMBOLS:
        info = get_symbol_info(symbol)
        if not info:
            print(f"Excluding {symbol}: Cannot get info.")
            continue
        symbol_infos[symbol] = info

        df_m15 = get_historical_data(symbol, TIMEFRAME_MT5, TIMEFRAME_STR, START_DATE, END_DATE, calculate_emas=True, calculate_atr=True, calculate_adx=True)
        if df_m15 is None or df_m15.empty:
            print(f"Excluding {symbol}: M15 data failed.")
            if symbol in symbol_infos: del symbol_infos[symbol]
            continue

        df_htf = get_historical_data(symbol, HTF_TIMEFRAME_MT5, HTF_TIMEFRAME_STR, START_DATE, END_DATE, calculate_emas=True, calculate_atr=False, calculate_adx=False)
        if df_htf is None or df_htf.empty:
            print(f"Excluding {symbol}: HTF data failed.")
            if symbol in symbol_infos: del symbol_infos[symbol]
            continue

        htf_ema_fast_col=f'EMA_{HTF_EMA_FAST_PERIOD}_HTF'; htf_ema_slow_col=f'EMA_{HTF_EMA_SLOW_PERIOD}_HTF'
        df_htf = df_htf.rename(columns={f'EMA_{HTF_EMA_FAST_PERIOD}': htf_ema_fast_col, f'EMA_{HTF_EMA_SLOW_PERIOD}': htf_ema_slow_col})

        print(f"Aligning {HTF_TIMEFRAME_STR} to {TIMEFRAME_STR} for {symbol}...")
        if df_m15.index.tz is None: df_m15.index = df_m15.index.tz_localize(UTC)
        if df_htf.index.tz is None: df_htf.index = df_htf.index.tz_localize(UTC)

        htf_cols_to_align = [col for col in [htf_ema_fast_col, htf_ema_slow_col] if col in df_htf.columns]
        if not htf_cols_to_align:
             print(f"Excluding {symbol}: HTF EMA columns missing.")
             if symbol in symbol_infos: del symbol_infos[symbol]
             continue

        df_htf_aligned = df_htf[htf_cols_to_align].reindex(df_m15.index, method='ffill')
        df_combined = pd.concat([df_m15, df_htf_aligned], axis=1)
        df_combined.dropna(inplace=True)

        if df_combined.empty:
            print(f"Excluding {symbol}: No combined data.")
            if symbol in symbol_infos: del symbol_infos[symbol]
            continue

        all_symbol_data[symbol] = df_combined
        print(f"Data prepared for {symbol}: {len(df_combined)} bars")

    valid_symbols = list(all_symbol_data.keys())
    if not valid_symbols: print("\nNo valid data for any symbol. Exiting."); return None
    print(f"\nProceeding with symbols: {', '.join(valid_symbols)}")

    try:
        for sym_key, sym_df in all_symbol_data.items():
            if sym_df.index.tz is None: print(f"Warning: Localizing index {sym_key}."); sym_df.index = sym_df.index.tz_localize(UTC)
        common_start_time = max(df.index.min() for df in all_symbol_data.values())
        common_end_time = min(df.index.max() for df in all_symbol_data.values())
        if common_start_time >= common_end_time: print("\nError: Invalid common date range."); return None
    except ValueError: print("\nError determining common date range."); return None
    print(f"\nUsing common data range: {common_start_time} to {common_end_time}")

    first_valid_symbol = valid_symbols[0]
    if all_symbol_data[first_valid_symbol].index.tz is None: all_symbol_data[first_valid_symbol].index = all_symbol_data[first_valid_symbol].index.tz_localize(UTC)
    common_time_index_df = all_symbol_data[first_valid_symbol].loc[common_start_time:common_end_time]
    if common_time_index_df.empty: print(f"\nError: No common index data."); return None
    common_time_index = common_time_index_df.index
    print(f"Generated common time index: {len(common_time_index)} timestamps.")

    current_day_str = ""; balance_at_day_start = account_balance
    print("\nStarting main backtest loop...")
    for timestamp in common_time_index:
        if timestamp.tzinfo is None: timestamp = timestamp.tz_localize(UTC)
        day_str = timestamp.strftime('%Y-%m-%d')
        if day_str != current_day_str:
            if current_day_str in daily_trade_stats:
                 day_stats = daily_trade_stats[current_day_str]
                 pnl_today = float(day_stats.get('pnl', 0.0))
                 print(f"{current_day_str}: Losses={day_stats.get('losses',0)}, PnL={pnl_today:.2f}, End Bal={account_balance:.2f}")
            current_day_str = day_str
            if current_day_str not in daily_trade_stats: daily_trade_stats[current_day_str] = {'losses': 0, 'pnl': 0.0, 'limit_reported': False}
            balance_at_day_start = account_balance

        today_stats = daily_trade_stats.get(current_day_str, {'losses': 0, 'pnl': 0.0, 'limit_reported': False})
        daily_pnl_today = float(today_stats.get('pnl', 0.0)); daily_loss_amount = abs(min(0, daily_pnl_today)); daily_profit_amount = max(0, daily_pnl_today)
        max_loss_limit = 0.0; max_profit_limit = 0.0
        if balance_at_day_start > 0: max_loss_limit = balance_at_day_start*(MAX_DAILY_LOSS_PERCENT/100.0); max_profit_limit = balance_at_day_start*(MAX_DAILY_PROFIT_PERCENT/100.0)
        losses_today = today_stats.get('losses', 0); day_limit_reached = False; limit_reason = ""
        if losses_today >= MAX_LOSING_TRADES_PER_DAY: day_limit_reached=True; limit_reason=f"Max Losses ({MAX_LOSING_TRADES_PER_DAY})"
        elif max_loss_limit > 0 and daily_loss_amount >= max_loss_limit: day_limit_reached=True; limit_reason=f"Max Loss % ({MAX_DAILY_LOSS_PERCENT}%)"
        elif max_profit_limit > 0 and daily_profit_amount >= max_profit_limit: day_limit_reached=True; limit_reason=f"Max Profit % ({MAX_DAILY_PROFIT_PERCENT}%)"
        if day_limit_reached:
            if not today_stats.get('limit_reported', False):
                today_stats['limit_reported'] = True; daily_trade_stats[current_day_str] = today_stats
            continue

        for symbol in valid_symbols:
            df_combined = all_symbol_data[symbol]
            symbol_info = symbol_infos[symbol]
            if df_combined.index.tz is None: df_combined.index = df_combined.index.tz_localize(UTC)
            if timestamp not in df_combined.index: continue

            try:
                current_bar_index = df_combined.index.get_loc(timestamp)
                if current_bar_index < 1: continue
                current_bar = df_combined.iloc[current_bar_index]
                previous_bar = df_combined.iloc[current_bar_index - 1]
            except (KeyError, IndexError): continue

            if not check_session(timestamp): continue

            try:
                ema_fast_key=f'EMA_{EMA_FAST_PERIOD}'; ema_slow_key=f'EMA_{EMA_SLOW_PERIOD}'
                atr_key=f'ATR_{ATR_PERIOD}'; adx_key=f'ADX_{ADX_PERIOD}'
                htf_ema_fast_key=f'EMA_{HTF_EMA_FAST_PERIOD}_HTF'; htf_ema_slow_key=f'EMA_{HTF_EMA_SLOW_PERIOD}_HTF'
                atr_ma_key = f'ATR_MA_{ATR_MA_PERIOD}'

                required_cols_curr = [ema_fast_key, ema_slow_key, atr_key, adx_key, atr_ma_key, htf_ema_fast_key, htf_ema_slow_key, 'open', 'close', 'high', 'low']
                required_cols_prev = ['close', ema_fast_key, 'high', 'low', 'open'] # Need open for candle check
                if not all(k in current_bar for k in required_cols_curr) or not all(k in previous_bar for k in required_cols_prev): continue

                ema_fast = current_bar[ema_fast_key]; ema_slow = current_bar[ema_slow_key]
                atr = current_bar[atr_key]; adx = current_bar[adx_key] # Read but not used
                atr_ma = current_bar[atr_ma_key]
                curr_open=current_bar['open']; curr_close=current_bar['close']
                curr_high=current_bar['high']; curr_low=current_bar['low']
                prev_close=previous_bar['close']; prev_ema_fast=previous_bar[ema_fast_key]
                prev_high=previous_bar['high']; prev_low=previous_bar['low'];
                htf_ema_fast = current_bar[htf_ema_fast_key]; htf_ema_slow = current_bar[htf_ema_slow_key]

                if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(atr) or atr <= 0 or pd.isna(adx) \
                   or pd.isna(atr_ma) or atr_ma <= 0 \
                   or pd.isna(curr_open) or pd.isna(curr_close) or pd.isna(curr_high) or pd.isna(curr_low)\
                   or pd.isna(htf_ema_fast) or pd.isna(htf_ema_slow) \
                   or pd.isna(prev_ema_fast) or pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(prev_close) \
                   or pd.isna(previous_bar['open']): # Added check for prev_open
                    continue
            except KeyError as e: print(f"Warning: KeyError indicators {symbol} {timestamp}: {e}"); continue

            # --- Volatility Filter Check ---
            is_low_vol = atr < atr_ma * LOW_VOL_THRESHOLD_MULT
            is_high_vol = atr > atr_ma * HIGH_VOL_THRESHOLD_MULT
            if is_low_vol or is_high_vol: continue

            # --- Signal Logic ---
            long_signal = False; short_signal = False
            m15_uptrend = ema_fast > ema_slow; m15_downtrend = ema_fast < ema_slow
            htf_uptrend = htf_ema_fast > htf_ema_slow; htf_downtrend = htf_ema_fast < htf_ema_slow
            # ADX check disabled

            # --- *** ORIGINAL PULLBACK LOGIC *** ---
            max_pullback_dist_atr = 0.5
            pullback_touch_or_below_ema = prev_close <= prev_ema_fast + atr * 0.1
            pullback_not_too_far_below = prev_close >= prev_ema_fast - atr * max_pullback_dist_atr
            pullback_touch_or_above_ema = prev_close >= prev_ema_fast - atr * 0.1
            pullback_not_too_far_above = prev_close <= prev_ema_fast + atr * max_pullback_dist_atr
            pulled_back_to_ema_long = m15_uptrend and pullback_touch_or_below_ema and pullback_not_too_far_below
            pulled_back_to_ema_short = m15_downtrend and pullback_touch_or_above_ema and pullback_not_too_far_above
            # --- End Original Pullback ---

            # Original Candle Confirmation
            is_bullish_candle = curr_close > curr_open
            is_strong_close_bull = curr_close > prev_high
            is_bearish_candle = curr_close < curr_open
            is_strong_close_bear = curr_close < prev_low

            # Combined Signal Check (Orig Pullback, Orig Entry)
            if m15_uptrend and htf_uptrend and pulled_back_to_ema_long and is_bullish_candle and is_strong_close_bull:
                long_signal = True
            elif m15_downtrend and htf_downtrend and pulled_back_to_ema_short and is_bearish_candle and is_strong_close_bear:
                short_signal = True

            # --- Execute Trade (Simulated) ---
            if long_signal or short_signal:
                entry_price = curr_close
                point = symbol_info.point; digits = symbol_info.digits
                if point <= 0 or atr <= 0: print(f"{timestamp}|{symbol}|Error: Invalid point/ATR. Skip."); continue

                sl_price=0.0; tp_price=0.0; sl_dist_price=0.0; direction=""

                if long_signal:
                    direction = "LONG"; signal_candle_low = curr_low
                    sl_price = signal_candle_low - atr * STOP_LOSS_ATR_MULTIPLIER # Using 1.5
                    if sl_price >= entry_price: sl_price = signal_candle_low - point
                    if sl_price >= entry_price: print(f"{timestamp}|{symbol}|{direction}|SL Err(>=Entry). Skip."); continue
                    sl_dist_price = entry_price - sl_price
                    tp_price = entry_price + sl_dist_price * TAKE_PROFIT_RR_RATIO # Using 2.0
                else: # Short
                    direction = "SHORT"; signal_candle_high = curr_high
                    sl_price = signal_candle_high + atr * STOP_LOSS_ATR_MULTIPLIER # Using 1.5
                    if sl_price <= entry_price: sl_price = signal_candle_high + point
                    if sl_price <= entry_price: print(f"{timestamp}|{symbol}|{direction}|SL Err(<=Entry). Skip."); continue
                    sl_dist_price = sl_price - entry_price
                    tp_price = entry_price - sl_dist_price * TAKE_PROFIT_RR_RATIO # Using 2.0

                if sl_dist_price <= 0: print(f"{timestamp}|{symbol}|{direction}|SL Dist Err(<=0). Skip."); continue
                sl_points_abs = round(sl_dist_price / point)
                if sl_points_abs <= 0: print(f"{timestamp}|{symbol}|{direction}|SL Points Err(<=0). Skip."); continue

                position_size = calculate_position_size(symbol_info, account_balance, RISK_PER_TRADE_PERCENT, sl_points_abs)

                if position_size is not None and position_size > 0:
                    vol_step=symbol_info.volume_step; disp_dec=2
                    if isinstance(vol_step,(int,float)) and vol_step>0:
                         try:
                             step_dec=decimal.Decimal(str(vol_step)); disp_dec=abs(step_dec.as_tuple().exponent) if step_dec.is_finite() and step_dec.as_tuple().exponent<0 else 0; disp_dec=max(2,disp_dec)
                         except: disp_dec=2

                    print(f"\n{timestamp} | {symbol} | {direction} ENTRY (HTF:{'Up' if htf_uptrend else 'Down'}, OrigPullback, VolOK)") # Updated Log
                    print(f"  Entry={entry_price:.{digits}f}, SL={sl_price:.{digits}f}, TP={tp_price:.{digits}f} ({sl_points_abs} pts, RR:{TAKE_PROFIT_RR_RATIO})")
                    print(f"  ATR={atr:.{digits}f}, ATR_MA={atr_ma:.{digits}f}")
                    print(f"  Balance: {account_balance:.2f}, Risk%: {RISK_PER_TRADE_PERCENT}, Size={position_size:.{disp_dec}f}")

                    outcome, exit_price, exit_time, pnl = simulate_trade_outcome(
                        entry_price, sl_price, tp_price, direction, symbol_info, position_size,
                        df_combined, current_bar_index)

                    trade_data = {
                        'symbol': symbol, 'entry_time': timestamp, 'entry_price': entry_price, 'direction': direction,
                        'sl_price': sl_price, 'tp_price': tp_price, 'volume': position_size, 'outcome': outcome,
                        'exit_time': exit_time, 'exit_price': exit_price, 'pnl': pnl, 'balance_before': account_balance }
                    trades.append(trade_data)

                    balance_before = account_balance; account_balance += pnl
                    if isinstance(exit_time, pd.Timestamp) and exit_time.tzinfo is None: exit_time = exit_time.tz_localize(UTC)
                    elif isinstance(exit_time, datetime) and exit_time.tzinfo is None: exit_time = UTC.localize(exit_time)
                    exit_day_str = exit_time.strftime('%Y-%m-%d')
                    if exit_day_str not in daily_trade_stats: daily_trade_stats[exit_day_str] = {'losses': 0, 'pnl': 0.0, 'limit_reported': False}
                    current_exit_day_stats = daily_trade_stats[exit_day_str]
                    current_exit_day_stats['pnl'] = float(current_exit_day_stats.get('pnl', 0.0)) + pnl
                    if pnl < 0: current_exit_day_stats['losses'] = current_exit_day_stats.get('losses', 0) + 1
                    daily_trade_stats[exit_day_str] = current_exit_day_stats

                    print(f"  Outcome: {outcome} @{exit_time} | Exit={exit_price:.{digits}f} | PnL={pnl:.2f} | New Bal={account_balance:.2f}")

                    if account_balance <= 0: print(f"\n{timestamp} | BANKRUPT! Balance <= 0."); break
            # End trade execution
        # End symbol loop
        if account_balance <= 0: break # End time loop
    # End time loop

    print("\n--- Backtest loop finished ---")
    if account_balance > 0 and current_day_str and current_day_str in daily_trade_stats:
        day_stats = daily_trade_stats[current_day_str]
        pnl_today = float(day_stats.get('pnl', 0.0))
        print(f"Final Day ({current_day_str}): Losses={day_stats.get('losses',0)}, PnL={pnl_today:.2f}, Final Bal={account_balance:.2f}")
    return trades

# --- Results Analysis Function (with per-symbol breakdown) ---
def analyze_results(trades_list, initial_balance):
    """Analyzes the results of the backtest, including per-symbol performance."""
    if not trades_list: print("No trades executed."); return
    print("\n--- Analyzing Results ---")
    results_df = pd.DataFrame(trades_list)
    for col in ['pnl', 'balance_before', 'entry_price', 'exit_price', 'volume', 'sl_price', 'tp_price']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    results_df['entry_time'] = pd.to_datetime(results_df['entry_time'], errors='coerce', utc=True)
    results_df['exit_time'] = pd.to_datetime(results_df['exit_time'], errors='coerce', utc=True)

    results_df.dropna(subset=['pnl', 'balance_before', 'entry_time', 'exit_time'], inplace=True)
    if results_df.empty: print("No valid trades after cleaning."); return

    results_df.sort_values(by='exit_time', inplace=True); results_df.reset_index(drop=True, inplace=True)
    results_df['pnl_percent'] = results_df.apply(lambda row: (row['pnl']/row['balance_before'])*100 if row['balance_before']!=0 else 0, axis=1)

    # --- Overall Metrics ---
    total_trades = len(results_df); winning_trades = results_df[results_df['pnl'] > 0]; losing_trades = results_df[results_df['pnl'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = results_df['pnl'].sum(); sum_losses = losing_trades['pnl'].sum(); sum_wins = winning_trades['pnl'].sum()
    profit_factor = abs(sum_wins / sum_losses) if sum_losses != 0 else float('inf') if sum_wins > 0 else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0

    results_df['cumulative_pnl'] = results_df['pnl'].cumsum()
    results_df['account_equity'] = initial_balance + results_df['cumulative_pnl']
    results_df['peak_equity'] = results_df['account_equity'].cummax()
    results_df['drawdown_amount'] = results_df['peak_equity'] - results_df['account_equity']
    max_dd_amount = results_df['drawdown_amount'].max() if not results_df['drawdown_amount'].empty else 0.0
    max_dd_percent = 0.0
    if max_dd_amount > 0 and not results_df.empty:
         try: peak_at_max_dd = results_df.loc[results_df['drawdown_amount'].idxmax()]['peak_equity']; max_dd_percent = (max_dd_amount / peak_at_max_dd)*100 if peak_at_max_dd>0 else 0
         except KeyError: max_dd_percent = 0.0

    longest_win_streak = 0; longest_loss_streak = 0
    if total_trades > 0:
        results_df['win'] = results_df['pnl'] > 0
        results_df['streak_group'] = results_df['win'].ne(results_df['win'].shift()).cumsum()
        streaks = results_df.groupby('streak_group').agg(is_win_streak=('win', 'first'), streak_len=('win', 'size'))
        win_streaks = streaks[streaks['is_win_streak']==True]['streak_len']; loss_streaks = streaks[streaks['is_win_streak']==False]['streak_len']
        longest_win_streak = win_streaks.max() if not win_streaks.empty else 0; longest_loss_streak = loss_streaks.max() if not loss_streaks.empty else 0

    # --- Print Overall Summary ---
    print("\n--- Overall Backtest Results ---")
    start_dt_str = results_df['entry_time'].min().strftime('%Y-%m-%d') if not results_df.empty else START_DATE.strftime('%Y-%m-%d')
    end_dt_str = results_df['exit_time'].max().strftime('%Y-%m-%d') if not results_df.empty else END_DATE.strftime('%Y-%m-%d')

    print(f"Period Analyzed: {start_dt_str} to {end_dt_str}")
    # *** UPDATED Description ***
    print(f"Strategy: {STRATEGY_NAME} (Entry:{TIMEFRAME_STR}, HTF:{HTF_TIMEFRAME_STR}, No ADX, SL:{STOP_LOSS_ATR_MULTIPLIER}*ATR, RR:{TAKE_PROFIT_RR_RATIO}, OrigPullback, VolFilter)")
    print(f"Total Trades: {total_trades}, Wins: {len(winning_trades)}, Losses: {len(losing_trades)}, Win Rate: {win_rate:.2f}%")
    print("-" * 30)
    print(f"Total Net PnL: ${total_pnl:.2f} ({ (total_pnl / initial_balance) * 100:.2f}%)")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}, Avg W/L Ratio: {avg_win_loss_ratio:.2f}")
    print("-" * 30)
    print(f"Max Drawdown: ${max_dd_amount:.2f} ({max_dd_percent:.2f}%)")
    print(f"Longest Win Streak: {longest_win_streak}, Longest Loss Streak: {longest_loss_streak}")
    print("-" * 30)
    print(f"Initial Balance: ${initial_balance:.2f}, Final Balance: ${initial_balance + total_pnl:.2f}")
    print("-" * 50)

    # --- Per Symbol Analysis ---
    print("\n--- Performance By Symbol ---")
    symbol_groups = results_df.groupby('symbol')
    summary_data = []
    for symbol, group in symbol_groups:
        s_total_trades = len(group)
        s_winning_trades = group[group['pnl'] > 0]
        s_losing_trades = group[group['pnl'] <= 0]
        s_win_rate = (len(s_winning_trades) / s_total_trades) * 100 if s_total_trades > 0 else 0
        s_total_pnl = group['pnl'].sum()
        s_sum_losses = s_losing_trades['pnl'].sum()
        s_sum_wins = s_winning_trades['pnl'].sum()
        s_profit_factor = abs(s_sum_wins / s_sum_losses) if s_sum_losses != 0 else float('inf') if s_sum_wins > 0 else 0
        summary_data.append({
            'Symbol': symbol,
            'Trades': s_total_trades,
            'Win Rate (%)': f"{s_win_rate:.2f}",
            'Total PnL ($)': f"{s_total_pnl:.2f}",
            'Profit Factor': f"{s_profit_factor:.2f}"
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("-" * 50)

    # --- Save Results ---
    try:
        safe_name = "".join(c if c.isalnum() else "_" for c in STRATEGY_NAME)
        csv_filename = f"{safe_name}_results_{start_dt_str}_to_{end_dt_str}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
    except Exception as e: print(f"Error saving CSV: {e}")

# --- Execute ---
if __name__ == "__main__":
    mt5_init = False
    try:
        if initialize_mt5():
            mt5_init = True
            # *** Set parameters for the v1.10 run ***
            STRATEGY_NAME = "BoringStrategy_v1.10_OrigPullback_VolFilter"
            SYMBOLS = ["EURUSD", "USDJPY"]
            ADX_THRESHOLD = 0
            TAKE_PROFIT_RR_RATIO = 2.0
            STOP_LOSS_ATR_MULTIPLIER = 1.5
            SESSION_START_UTC_HOUR = 12 # Original session
            SESSION_END_UTC_HOUR = 18

            exec_trades = run_backtest()
            if exec_trades is not None:
                if isinstance(exec_trades, list):
                     analyze_results(exec_trades, INITIAL_BALANCE)
                else: print("Backtest returned unexpected type.")
            else: print("Backtest returned None.")
        else: print("MT5 init failed. Exiting.")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR ---"); import traceback; print(traceback.format_exc()); print(f"Error: {e}")
    finally:
        if mt5_init and mt5.terminal_info(): shutdown_mt5()
        elif not mt5_init: print("\nMT5 not initialized.")
        else: print("\nMT5 connection lost before script end.")
        print("\nScript finished.")