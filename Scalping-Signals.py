# pip install requests pandas numpy ta-lib
import time, math, requests
import pandas as pd
import numpy as np
from datetime import datetime

# ---------- Config (Advanced for high win rate) ----------
# Last updated: November 3, 2025
SYMBOLS = [
    'BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','DOGEUSDT','XRPUSDT','ADAUSDT','LINKUSDT',
    'AVAXUSDT','NEARUSDT','SEIUSDT','APTUSDT','ARBUSDT','OPUSDT','DOTUSDT','ATOMUSDT','SUIUSDT','ZENUSDT',
    'MATICUSDT','LTCUSDT','INJUSDT','FETUSDT','RNDRUSDT','UNIUSDT','FTMUSDT','SANDUSDT','MANAUSDT','GMTUSDT',
    'TRXUSDT','BCHUSDT','ETCUSDT','XLMUSDT','FILUSDT','EOSUSDT','TONUSDT','AAVEUSDT','IMXUSDT','LDOUSDT',
    'SHIBUSDT','PEPEUSDT'
]
# Extended-tier control (staggered scanning cadence)
EXTENDED_SYMBOLS = [
    'TRXUSDT','BCHUSDT','ETCUSDT','XLMUSDT','FILUSDT','EOSUSDT','TONUSDT','AAVEUSDT','IMXUSDT','LDOUSDT',
    'SHIBUSDT','PEPEUSDT'
]
EXTENDED_SCAN_EVERY = 2  # scan extended symbols every N cycles
SCAN_CYCLE = 0
PRIMARY_IV = '1m'
CONFIRM_IV = '5m'
TREND_IV = '15m'
LOOKBACK = 400
SLEEP_SEC = 2

# Advanced Indicators
USE_TREND = True         # require 15m trend alignment
USE_CONFIRM = True       # require 5m confirmation
USE_MARKET_REGIME = True # adapt to market conditions
USE_SCORE_SYSTEM = True  # use scoring instead of binary decisions

# Volatility Settings
ATR_LEN = 14
ATR_MIN_PCT = 0.05       # balanced volatility requirement
ATR_FILTER_MULT = 1.2    # for dynamic ATR filtering
ATR_THRESHOLD = {
    'low': 0.04,         # low volatility threshold
    'moderate': 0.05,    # moderate volatility threshold
    'high': 0.08         # high volatility threshold
}

# Volume Settings
VOL_MA_LEN = 20
VOL_MIN_MULT = 1.2       # require 20% volume surge
VOL_PROFILE_BARS = 100   # for volume profile analysis
VOL_THRESHOLD = {
    'low': 1.0,          # low volume threshold
    'moderate': 1.2,     # moderate volume threshold
    'high': 1.5          # high volume threshold
}

# Moving Averages
EMA_FAST = 8
EMA_MID = 21
EMA_SLOW = 50
EMA_TREND = 200
HAMA_FAST = 9            # Heikin-Ashi EMA fast
HAMA_SLOW = 21           # Heikin-Ashi EMA slow

# Oscillators
RSI_LEN = 14
RSI_SMOOTH = 3           # Smoothed RSI
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
CCI_LEN = 20             # Commodity Channel Index
# Additional Indicator Config
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_LEN = 20
BB_STD = 2.0
KC_EMA_LEN = 20
KC_MULT = 1.5
CHOP_LEN = 14
MFI_LEN = 14
OBV_SMOOTH = 10
SQUEEZE_LEN = 20
ATR_BAND_MULT = 1.5
ADX_LEN = 14
ADX_THRESHOLD = {
    'weak': 15,
    'moderate': 22,      # slightly reduced for more signals
    'strong': 35
}
EMA_SLOPE_MIN = 0.018    # slightly reduced momentum requirement

# Market Structure
SR_LOOKBACK = 60
SWING_WINDOW = 3
BREAK_K = 0.0018         # breakout threshold (0.18%)
RETEST_BARS = 6
LIQUIDITY_ZONES = True   # identify liquidity zones
LIQUIDITY_LOOKBACK = 50  # bars to look back for liquidity zones

# Risk Management
SL_ATR = 2.0
TP_RR = [1.1, 2.16, 3.9]
TP1_R = 0.9
TP2_R = 2.5
TP3_R = 4.0
MAX_SL_PCT = 1.5

# Signal Quality
MIN_SCORE = 60
MIN_SCORE_VOLATILE = 65
CONSECUTIVE_SIGNALS = 1
LONG_BIAS = 0.9
COOLDOWN_MIN = 3

# Debugging
DEBUG_LOG = False
QUIET_ERRORS = True

# Market Hours (avoid low liquidity periods)
AVOID_LOW_LIQUIDITY = True
LOW_LIQUIDITY_HOURS = [
    (0, 2),
    (22, 24)
]

BASE = 'https://fapi.binance.com'

# ---------- Helper Functions ----------
def klines(symbol, interval, lookback):
    # Binance futures klines API expects a 'limit' parameter
    limit = lookback
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    last_error = None
    for attempt in range(4):
        try:
            r = requests.get(f'{BASE}/fapi/v1/klines', params=params, timeout=25)
            r.raise_for_status()
            cols = ['open_time','open','high','low','close','volume','close_time','qav','trades','taker_b','taker_q','ignore']
            df = pd.DataFrame(r.json(), columns=cols)
            for c in ['open','high','low','close','volume']:
                df[c] = df[c].astype(float)
            return df
        except Exception as e:
            last_error = e
            time.sleep(1 + attempt*1.5)
    if last_error:
        raise last_error
    raise Exception(f"Failed to fetch {symbol} after 4 attempts")

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def sma(s, n): return s.rolling(window=n).mean()

def rsi(series, n, smooth=None):
    d = series.diff()
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean()
    rs = ru / (rd + 1e-9)
    rsi_raw = 100 - (100/(1+rs))
    if smooth and smooth > 1:
        return rsi_raw.rolling(window=smooth).mean()
    return rsi_raw

def atr(df, n):
    h,l,c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=ADX_LEN):
    h,l,c = df['high'], df['low'], df['close']
    up = h.diff()
    dn = -l.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(n).mean()
    pdi = 100 * pd.Series(plus_dm).rolling(n).mean() / (atr_ + 1e-9)
    mdi = 100 * pd.Series(minus_dm).rolling(n).mean() / (atr_ + 1e-9)
    dx = (abs(pdi - mdi) / (pdi + mdi + 1e-9)) * 100
    adx_val = dx.rolling(n).mean()
    return adx_val, pdi, mdi

def stochastic(df, k_period=STOCH_K, d_period=STOCH_D, smooth=STOCH_SMOOTH):
    high_roll = df['high'].rolling(window=k_period).max()
    low_roll = df['low'].rolling(window=k_period).min()
    fast_k = 100 * (df['close'] - low_roll) / (high_roll - low_roll + 1e-9)
    fast_d = fast_k.rolling(window=d_period).mean()
    if smooth and smooth > 1:
        slow_k = fast_k.rolling(window=smooth).mean()
        slow_d = fast_d.rolling(window=smooth).mean()
        return slow_k, slow_d
    return fast_k, fast_d

def cci(df, n=CCI_LEN):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    tp_sma = tp.rolling(window=n).mean()
    md = tp.rolling(window=n).apply(lambda x: abs(x - x.mean()).mean())
    cci_val = (tp - tp_sma) / (0.015 * md + 1e-9)
    return cci_val

def vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = (tp * df['volume']).cumsum()
    vv = df['volume'].cumsum().replace(0, np.nan)
    return pv / vv

def choppiness_index(df, n=CHOP_LEN):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()], axis=1).max(axis=1)
    tr_sum = tr.rolling(n).sum()
    range_sum = (h.rolling(n).max() - l.rolling(n).min())
    
    # Initialize chop series
    chop = pd.Series(index=df.index, dtype=float)
    
    # Where range is significant, calculate chop
    valid_mask = range_sum > 1e-9
    # Avoid log(0) by checking tr_sum as well
    valid_mask &= (tr_sum > 1e-9)
    
    chop.loc[valid_mask] = 100 * np.log10(tr_sum[valid_mask] / range_sum[valid_mask]) / np.log10(n)
    
    # Where range or tr_sum is negligible, market is flat -> max choppiness
    chop.fillna(100.0, inplace=True)
    
    return chop

def money_flow_index(df, n=MFI_LEN):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    mf = tp * df['volume']
    pos = np.where(tp > tp.shift(1), mf, 0.0)
    neg = np.where(tp < tp.shift(1), mf, 0.0)
    pmf = pd.Series(pos, index=df.index).rolling(n).sum()
    nmf = pd.Series(neg, index=df.index).rolling(n).sum()
    return 100 - (100/(1 + (pmf / (nmf + 1e-9))))

def on_balance_volume(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def heikin_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Initialize first HA open with first price open
    ha_df.at[0, 'ha_open'] = df.at[0, 'open']
    
    # Calculate HA open for the rest of the data
    for i in range(1, len(df)):
        ha_df.at[i, 'ha_open'] = (ha_df.at[i-1, 'ha_open'] + ha_df.at[i-1, 'ha_close']) / 2
    
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    return ha_df

def last_swing_levels(df, win=SWING_WINDOW, lookback=SR_LOOKBACK):
    hi_idx = None; lo_idx = None
    win = int(lookback/2)
    for i in range(len(df)-2, max(len(df)-lookback-1, win), -1):
        seg = df['high'].iloc[i-win:i+win+1]
        if df['high'].iloc[i] == seg.max():
            hi_idx = i; break
    for i in range(len(df)-2, max(len(df)-lookback-1, win), -1):
        seg = df['low'].iloc[i-win:i+win+1]
        if df['low'].iloc[i] == seg.min():
            lo_idx = i; break
    swing_high = df['high'].iloc[hi_idx] if hi_idx is not None else None
    swing_low  = df['low'].iloc[lo_idx]  if lo_idx is not None else None
    return swing_high, swing_low, hi_idx, lo_idx

def find_order_blocks(df, lookback=30):
    """Find order blocks (areas of high volume and price rejection)"""
    blocks = {'buy': [], 'sell': []}
    
    for i in range(SWING_WINDOW, min(lookback, len(df)-SWING_WINDOW)):
        idx = len(df) - i - 1
        
        # Check for bullish order block (price rejection up)
        if (df['close'].iloc[idx] > df['open'].iloc[idx] and  # Bullish candle
            df['volume'].iloc[idx] > df['vol_ma'].iloc[idx] * 1.5 and  # High volume
            df['low'].iloc[idx+1:idx+4].min() > df['low'].iloc[idx]):  # Price respected as support
            blocks['buy'].append((idx, df['low'].iloc[idx], df['high'].iloc[idx]))
        
        # Check for bearish order block (price rejection down)
        if (df['close'].iloc[idx] < df['open'].iloc[idx] and  # Bearish candle
            df['volume'].iloc[idx] > df['vol_ma'].iloc[idx] * 1.5 and  # High volume
            df['high'].iloc[idx+1:idx+4].max() < df['high'].iloc[idx]):  # Price respected as resistance
            blocks['sell'].append((idx, df['low'].iloc[idx], df['high'].iloc[idx]))
    
    # Sort by recency and limit to top 3
    blocks['buy'] = sorted(blocks['buy'], key=lambda x: x[0], reverse=True)[:3]
    blocks['sell'] = sorted(blocks['sell'], key=lambda x: x[0], reverse=True)[:3]
    
    return blocks

def detect_market_regime(df):
    """Detect current market regime (trending, ranging, volatile)"""
    # Calculate key metrics
    close = df['close']
    atr_val = atr(df, ATR_LEN).iloc[-1]
    atr_pct = atr_val / close.iloc[-1] * 100
    adx_val = adx(df)[0].iloc[-1]
    
    # Calculate price movement efficiency
    price_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
    path_length = df['close'].diff().abs().rolling(20).sum()
    efficiency = (price_range / (path_length + 1e-9)).iloc[-1]
    
    # Determine regime
    if adx_val >= ADX_THRESHOLD['strong']:
        regime = 'strong_trend'
    elif adx_val >= ADX_THRESHOLD['moderate']:
        regime = 'moderate_trend'
    elif efficiency < 0.3:
        regime = 'choppy'
    elif atr_pct > ATR_MIN_PCT * 1.5:
        regime = 'volatile'
    else:
        regime = 'ranging'
        
    return regime

def enrich(df):
    df = df.copy()
    
    # Moving averages
    df['ema_fast'] = ema(df['close'], EMA_FAST)
    df['ema_mid'] = ema(df['close'], EMA_MID)
    df['ema_slow'] = ema(df['close'], EMA_SLOW)
    df['ema_trend'] = ema(df['close'], EMA_TREND)
    df['ema_fast_slope'] = df['ema_fast'].diff()
    df['ema_mid_slope'] = df['ema_mid'].diff()
    
    # Heikin-Ashi transformation
    ha = heikin_ashi(df)
    df['ha_open'] = ha['ha_open']
    df['ha_high'] = ha['ha_high']
    df['ha_low'] = ha['ha_low']
    df['ha_close'] = ha['ha_close']
    df['ha_ema_fast'] = ema(df['ha_close'], HAMA_FAST)
    df['ha_ema_slow'] = ema(df['ha_close'], HAMA_SLOW)
    df['ha_ema_fast_slope'] = df['ha_ema_fast'].diff()
    
    # Oscillators
    df['rsi'] = rsi(df['close'], RSI_LEN)
    df['rsi_smooth'] = rsi(df['close'], RSI_LEN, RSI_SMOOTH)
    stoch_k, stoch_d = stochastic(df)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['cci'] = cci(df)
    
    # Trend and volatility
    df['atr'] = atr(df, ATR_LEN)
    adx_val, pdi, mdi = adx(df)
    df['adx'] = adx_val
    df['pdi'] = pdi
    df['mdi'] = mdi
    
    # Volume analysis
    df['vol_ma'] = df['volume'].rolling(VOL_MA_LEN).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['vwap'] = vwap(df)
    # Additional indicators for high-precision scalp mode
    # MACD
    macd_fast = ema(df['close'], MACD_FAST)
    macd_slow = ema(df['close'], MACD_SLOW)
    df['macd'] = macd_fast - macd_slow
    df['macd_signal'] = ema(df['macd'], MACD_SIGNAL)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # Bollinger Bands
    bb_mid = sma(df['close'], BB_LEN)
    bb_std = df['close'].rolling(BB_LEN).std()
    df['bb_mid'] = bb_mid
    df['bb_upper'] = bb_mid + BB_STD * bb_std
    df['bb_lower'] = bb_mid - BB_STD * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-9)
    # Keltner Channels
    kc_mid = ema(df['close'], KC_EMA_LEN)
    df['kc_mid'] = kc_mid
    df['kc_upper'] = kc_mid + KC_MULT * df['atr']
    df['kc_lower'] = kc_mid - KC_MULT * df['atr']
    df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    # Choppiness Index
    df['chop'] = choppiness_index(df, CHOP_LEN)
    # Money Flow Index
    df['mfi'] = money_flow_index(df, MFI_LEN)
    # On-Balance Volume and slope
    df['obv'] = on_balance_volume(df).rolling(OBV_SMOOTH).mean()
    df['obv_slope'] = df['obv'] - df['obv'].shift(1)
    # ATR band directional cue
    df['atr_band_upper'] = df['ema_mid'] + ATR_BAND_MULT * df['atr']
    df['atr_band_lower'] = df['ema_mid'] - ATR_BAND_MULT * df['atr']
    df['atr_band_dir'] = np.where(df['close'] > df['atr_band_upper'], 1,
                                  np.where(df['close'] < df['atr_band_lower'], -1, 0))
    return df

def trend_ok(df15):
    """Advanced trend detection using multiple indicators"""
    # Calculate EMAs for trend direction
    ema_trend = ema(df15['close'], EMA_TREND)
    ema_now = ema_trend.iloc[-2]
    ema_prev = ema_trend.iloc[-3]
    c2 = df15['close'].iloc[-2]
    
    # Calculate ADX for trend strength
    adx_val = adx(df15)[0].iloc[-2]
    
    # Determine trend direction and strength
    up = (ema_now > ema_prev and c2 > ema_now and 
          df15['ema_fast'].iloc[-2] > df15['ema_mid'].iloc[-2])
    
    down = (ema_now < ema_prev and c2 < ema_now and 
            df15['ema_fast'].iloc[-2] < df15['ema_mid'].iloc[-2])
    
    # Add trend strength component
    strong_trend = adx_val >= ADX_THRESHOLD['moderate']
    
    return up and strong_trend, down and strong_trend

def is_low_liquidity_period():
    """Check if current time is in low liquidity period"""
    if not AVOID_LOW_LIQUIDITY:
        return False
        
    now_utc = datetime.utcnow().hour
    for start, end in LOW_LIQUIDITY_HOURS:
        if start <= now_utc < end:
            return True
    return False

# ---------- Signal logic ----------
_last_emit = {}
_signal_history = {}

def cooldown_ok(sym):
    now_min = int(time.time() // 60)
    last = _last_emit.get(sym, -9999)
    return (now_min - last) >= COOLDOWN_MIN

def mark_emit(sym):
    _last_emit[sym] = int(time.time() // 60)
    
def track_signal(sym, side, score):
    """Track signal history for consecutive signal confirmation"""
    now_min = int(time.time() // 60)
    if sym not in _signal_history:
        _signal_history[sym] = []
    
    # Add new signal to history
    _signal_history[sym].append((now_min, side, score))
    
    # Clean up old signals (older than 10 minutes)
    _signal_history[sym] = [s for s in _signal_history[sym] if now_min - s[0] <= 10]
    
    # Check for consecutive signals
    if len(_signal_history[sym]) >= CONSECUTIVE_SIGNALS:
        recent_signals = _signal_history[sym][-CONSECUTIVE_SIGNALS:]
        sides = [s[1] for s in recent_signals]
        return all(s == side for s in sides)
    
    return False

def calculate_signal_score(d1, d5, d15, regime, row, conf, swing_high, swing_low, 
                                           breakout_up, breakout_dn, retest_up, retest_dn, signal_type):
    """Calculate a signal quality score (0-100) based on multiple factors"""
    score = 0
    components = []
    
    # 1. Trend Alignment & VWAP (up to 20 points)
    if signal_type == 'long':
        if row['ema_fast'] > row['ema_mid']: score += 4; components.append("EMA Fast>Mid")
        if row['ema_mid'] > row['ema_slow']: score += 4; components.append("EMA Mid>Slow")
        if row['close'] > row['ema_trend']: score += 4; components.append("Price>EMA200")
        if row['close'] > row['vwap']: score += 8; components.append("Price>VWAP")
    else:  # short
        if row['ema_fast'] < row['ema_mid']: score += 4; components.append("EMA Fast<Mid")
        if row['ema_mid'] < row['ema_slow']: score += 4; components.append("EMA Mid<Slow")
        if row['close'] < row['ema_trend']: score += 4; components.append("Price<EMA200")
        if row['close'] < row['vwap']: score += 8; components.append("Price<VWAP")
    
    # 2. Oscillator Status (RSI, Stochastics) - up to 15 points
    rsi_now = row['rsi']
    stoch_k_now = row['stoch_k']
    stoch_d_now = row['stoch_d']
    if signal_type == 'long':
        if rsi_now < 30: score += 5; components.append("RSI Oversold")
        elif rsi_now < 50: score += 3; components.append("RSI Neutral Bullish")
        elif rsi_now < 70: score += 1; components.append("RSI Bullish")
        else: score -= 10; components.append("RSI Overbought Penalty")
        if stoch_k_now > stoch_d_now and stoch_k_now > 70: score += 5; components.append("Stoch Bullish Momentum")
        elif stoch_k_now > stoch_d_now and stoch_k_now > 50: score += 2; components.append("Stoch Bullish Crossover")
    else:
        if rsi_now > 70: score += 5; components.append("RSI Overbought")
        elif rsi_now > 60: score += 3; components.append("RSI Strong Bearish")
        elif rsi_now > 50: score += 1; components.append("RSI Bearish")
        if stoch_k_now < stoch_d_now and stoch_k_now < 30: score += 5; components.append("Stoch Bearish Momentum")
        elif stoch_k_now < stoch_d_now and stoch_k_now < 50: score += 2; components.append("Stoch Bearish Crossover")

    # 2b. Divergence penalty (up to -8 points)
    rsi_prev = d1['rsi'].iloc[-3] if 'rsi' in d1.columns and len(d1) >= 3 else rsi_now
    price_prev_high = d1['high'].iloc[-3] if len(d1) >= 3 else row['high']
    price_prev_low = d1['low'].iloc[-3] if len(d1) >= 3 else row['low']
    if signal_type == 'long':
        if row['high'] > price_prev_high and rsi_now < rsi_prev:
            score -= 8; components.append("Bearish RSI Divergence")
    else:
        if row['low'] < price_prev_low and rsi_now > rsi_prev:
            score -= 8; components.append("Bullish RSI Divergence")

    # 3. Momentum (up to 15 points)
    if signal_type == 'long':
        score += min(10, max(0, row['ema_fast_slope'] / 0.001))
        if row['ha_close'] > row['ha_open']: score += 5
    else:
        score += min(10, max(0, -row['ema_fast_slope'] / 0.001))
        if row['ha_close'] < row['ha_open']: score += 5
    
    # 4. Volume (up to 10 points)
    score += min(10, max(0, (row['vol_ratio'] - 1.5) * 10))
    
    # 4b. Volume quality (MFI/OBV) - up to 8 points
    mfi_now = row['mfi'] if 'mfi' in row.index else 50
    obv_slope_now = row['obv_slope'] if 'obv_slope' in row.index else 0
    if signal_type == 'long':
        if mfi_now > 60: score += 4; components.append("MFI Strong Inflow")
        if obv_slope_now > 0: score += 4; components.append("OBV Rising")
    else:
        if mfi_now < 40: score += 4; components.append("MFI Strong Outflow")
        if obv_slope_now < 0: score += 4; components.append("OBV Falling")
    
    # 5. MACD confirmation (up to 10 points)
    if signal_type == 'long':
        if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0: score += 10; components.append("MACD Bullish")
    else:
        if row['macd'] < row['macd_signal'] and row['macd_hist'] < 0: score += 10; components.append("MACD Bearish")
    
    # 6. Breakout/Retest (up to 15 points)
    if breakout_up and signal_type == 'long': score += 10
    if retest_up and signal_type == 'long': score += 15
    if breakout_dn and signal_type == 'short': score += 10
    if retest_dn and signal_type == 'short': score += 15
    
    # 7. Squeeze release (up to 10 points)
    bb_width_prev = d1['bb_width'].iloc[-3] if 'bb_width' in d1.columns and len(d1) >= 3 else None
    if bb_width_prev is not None and not bool(row['squeeze_on']):
        width_expanding = row['bb_width'] > bb_width_prev
        if signal_type == 'long' and width_expanding and row['close'] > row['bb_upper']:
            score += 10; components.append("Squeeze Release Up")
        if signal_type == 'short' and width_expanding and row['close'] < row['bb_lower']:
            score += 10; components.append("Squeeze Release Down")
    
    # 8. Confirmation timeframe (up to 10 points)
    if signal_type == 'long':
        if conf['ema_fast'] > conf['ema_slow']: score += 7
        if conf['rsi'] > 55: score += 3
        if 'macd_hist' in conf.index and conf['macd_hist'] > 0: score += 3; components.append("HTF MACD Bullish")
    else:
        if conf['ema_fast'] < conf['ema_slow']: score += 7
        if conf['rsi'] < 45: score += 3
        if 'macd_hist' in conf.index and conf['macd_hist'] < 0: score += 3; components.append("HTF MACD Bearish")
    
    # 9. Choppiness adjustment
    if 'chop' in d1.columns:
        chop_now = row['chop']
        if chop_now < 38: score += 5; components.append("Trending")
        elif chop_now > 62: score -= min(15, (chop_now - 62)) ; components.append("Choppy")
    
    # 10. Order book (up to 10 points)
    ob_imb = row['ob_imbalance'] if 'ob_imbalance' in row.index else 0
    spread_bp = row['ob_spread_bp'] if 'ob_spread_bp' in row.index else 999
    if signal_type == 'long':
        if ob_imb > 0.1: score += 7; components.append("OB Bid Imbalance")
        if spread_bp < 5: score += 3; components.append("Tight Spread")
    else:
        if ob_imb < -0.1: score += 7; components.append("OB Ask Imbalance")
        if spread_bp < 5: score += 3; components.append("Tight Spread")
    
    # Market regime adjustment
    if regime == 'strong_trend':
        score *= 1.1
    elif regime == 'choppy':
        score *= 0.8
    
    # Trend Strength (15 points max)
    adx_val = row['adx']
    if adx_val >= 40:
        score += 15
    elif adx_val >= 30:
        score += 10
    elif adx_val >= 20:
        score += 5
    
    # Microstructure alignment: order book imbalance and spread quality
    ob_imb = row.get('ob_imbalance', 0.0)
    spread_bp = row.get('ob_spread_bp', 999)
    if signal_type == 'long':
        if ob_imb > 0.07:
            score += 5
        elif ob_imb > 0.03:
            score += 3
        else:
            score -= 7
    elif signal_type == 'short':
        if ob_imb < -0.07:
            score += 5
        elif ob_imb < -0.03:
            score += 3
        else:
            score -= 7

    # Penalize wide spreads typical of illiquid or manipulative conditions
    if spread_bp > 16:
        score -= 8
    elif spread_bp > 12:
        score -= 4

    return min(100, score)

def make_signal(sym):
    # Skip during low liquidity periods
    if is_low_liquidity_period():
        return None
    
    # Get data for multiple timeframes
    d1 = enrich(klines(sym, PRIMARY_IV, LOOKBACK))
    d5 = enrich(klines(sym, CONFIRM_IV, LOOKBACK))
    d15 = enrich(klines(sym, TREND_IV, LOOKBACK))
    
    # Detect market regime
    regime = detect_market_regime(d5)
    
    # Get trend direction
    upTrend, downTrend = trend_ok(d15)

    # Get current candle data
    row = d1.iloc[-2]
    if any(math.isnan(x) for x in [row['atr'], row['rsi'], row['ema_fast'], row['ema_slow'], row['vol_ma'], row['adx']]):
        return None

    price = row['close']
    atr_v = row['atr']
    atr_pct = atr_v/price*100

    # Dynamic ATR filter based on market regime
    atr_min = ATR_MIN_PCT
    if regime == 'volatile':
        atr_min *= 0.8  # Lower threshold in volatile markets
    elif regime == 'choppy':
        atr_min *= 1.2  # Higher threshold in choppy markets

    # Price action analysis
    candle_body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    body_ratio = candle_body / (candle_range + 1e-9)
    
    # Confirmation timeframe
    conf = d5.iloc[-2]
    if any(math.isnan(x) for x in [conf['ema_fast'], conf['ema_slow'], conf['rsi'], conf['adx']]):
        return None
        
    # Confirmation conditions
    conf_long = (conf['ema_fast'] > conf['ema_slow'] and 
                conf['rsi'] > 52 and 
                conf['adx'] >= ADX_THRESHOLD['weak'])
                
    conf_short = (conf['ema_fast'] < conf['ema_slow'] and 
                 conf['rsi'] < 45 and 
                 conf['adx'] >= ADX_THRESHOLD['moderate'])

    # Order book snapshot (Binance Futures depth)
    ob_imbalance = 0.0; ob_spread_bp = 999.0
    try:
        depth = requests.get(f'{BASE}/fapi/v1/depth', params={'symbol': sym, 'limit': 20}, timeout=5)
        if depth.ok:
            djson = depth.json()
            bids = djson.get('bids', [])
            asks = djson.get('asks', [])
            if bids and asks:
                bid_qty = sum(float(q) for _, q in bids)
                ask_qty = sum(float(q) for _, q in asks)
                ob_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-9)
                best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
                mid = (best_bid + best_ask)/2
                ob_spread_bp = ((best_ask - best_bid)/(mid + 1e-9))*10000
    except Exception:
        pass
    # Use .loc to avoid SettingWithCopyWarning
    d1.loc[d1.index[-2], 'ob_imbalance'] = ob_imbalance
    d1.loc[d1.index[-2], 'ob_spread_bp'] = ob_spread_bp
    # Update the row reference to include the new columns
    row = d1.iloc[-2]

    # Market structure analysis
    swing_high, swing_low, hi_idx, lo_idx = last_swing_levels(d1)
    
    # Breakout detection
    breakout_up = (swing_high is not None) and (row['close'] > swing_high * (1 + BREAK_K))
    breakout_dn = (swing_low is not None) and (row['close'] < swing_low * (1 - BREAK_K))
    
    # Recent price action for retest analysis
    recent_lows = d1['low'].iloc[-RETEST_BARS-2:-2].min() if swing_high is not None else None
    recent_highs = d1['high'].iloc[-RETEST_BARS-2:-2].max() if swing_low is not None else None
    
    # Strong breakout or retest conditions
    strong_breakout_up = breakout_up and (row['close'] > swing_high * (1 + BREAK_K * 1.5)) if swing_high is not None else False
    strong_breakout_dn = breakout_dn and (row['close'] < swing_low * (1 - BREAK_K * 1.5)) if swing_low is not None else False
    
    retest_up = (breakout_up and recent_lows is not None and recent_lows <= swing_high * (1 + BREAK_K/3)) if (swing_high is not None and breakout_up) else False
    retest_dn = (breakout_dn and recent_highs is not None and recent_highs >= swing_low * (1 - BREAK_K/3)) if (swing_low is not None and breakout_dn) else False
    
    # Momentum conditions
    slope_ok_long = row['ema_fast_slope'] >= EMA_SLOPE_MIN
    slope_ok_short = row['ema_fast_slope'] <= -EMA_SLOPE_MIN
    
    # VWAP conditions
    vwap_long = row['close'] >= row['vwap']
    vwap_short = row['close'] <= row['vwap']
    
    # Heikin-Ashi confirmation
    ha_trend_long = row['ha_ema_fast'] > row['ha_ema_slow'] and row['ha_close'] > row['ha_open']
    ha_trend_short = row['ha_ema_fast'] < row['ha_ema_slow'] and row['ha_close'] < row['ha_open']

    # Core signal conditions
    long_core = (row['ema_fast'] > row['ema_mid'] and 
                row['ema_mid'] > row['ema_slow'] and 
                45 <= row['rsi'] <= 85 and
                row['stoch_k'] > row['stoch_d'] and
                slope_ok_long and 
                vwap_long and 
                row['adx'] >= ADX_THRESHOLD['weak'] and
                body_ratio > 0.4 and
                ha_trend_long and
                (not USE_TREND or upTrend) and
                (not USE_CONFIRM or conf_long))
                
    short_core = (row['ema_fast'] < row['ema_mid'] and 
                 row['ema_mid'] < row['ema_slow'] and 
                 15 <= row['rsi'] <= 55 and
                 row['stoch_k'] < row['stoch_d'] and
                 slope_ok_short and 
                 vwap_short and 
                 row['adx'] >= ADX_THRESHOLD['moderate'] and
                 body_ratio > 0.4 and
                 ha_trend_short and
                 (not USE_TREND or downTrend) and
                 (not USE_CONFIRM or conf_short))

    # Volume condition
    vol_ok = row['volume'] >= VOL_MIN_MULT * row['vol_ma']
    
    # ATR condition
    atr_ok = atr_pct >= atr_min

    # Signal conditions with scoring system
    if USE_SCORE_SYSTEM:
        # Calculate signal scores
        long_score = calculate_signal_score(d1, d5, d15, regime, row, conf, swing_high, swing_low, 
                                           breakout_up, False, retest_up, False, 'long')
        # Apply LONG_BIAS to favor long signals (lower threshold for longs)
        # long_score = long_score / LONG_BIAS # This inflated score > 100. Now applying bias to threshold.
                                           
        short_score = calculate_signal_score(d1, d5, d15, regime, row, conf, swing_high, swing_low, 
                                            False, breakout_dn, False, retest_dn, 'short')
        
        # Dynamic score threshold based on market regime
        min_score_required = MIN_SCORE_VOLATILE if regime == 'volatile' else MIN_SCORE
        if sym in EXTENDED_SYMBOLS:
            min_score_required = min(100, min_score_required + 2)
        min_long_score_req = min_score_required * LONG_BIAS
        
        # Time-based filter to avoid low liquidity periods
        current_hour = datetime.utcnow().hour
        if AVOID_LOW_LIQUIDITY:
            # Check if current hour is in low liquidity periods
            liquidity_ok = True
            for start_hour, end_hour in LOW_LIQUIDITY_HOURS:
                if start_hour <= current_hour < end_hour:
                    liquidity_ok = False
                    break
            if not liquidity_ok:
                if DEBUG_LOG:
                    print(f"DBG {sym}: skipped due to low-liquidity hour UTC={current_hour}")
                return None  # Skip signals during low liquidity periods
        
        # Generate signals based on score threshold with directional sanity checks
        pdi_now = row['pdi']; mdi_now = row['mdi']
        # Adaptive momentum override for continuation moves without fresh breakout
        alt_long_ok = (
            upTrend and vwap_long and slope_ok_long and pdi_now > mdi_now and
            row['macd_hist'] >= 0 and row['macd_hist'] > d1['macd_hist'].iloc[-3] and
            row['bb_width'] > d1['bb_width'].iloc[-3]
        )
        alt_short_ok = (
            downTrend and vwap_short and slope_ok_short and mdi_now > pdi_now and
            row['macd_hist'] <= 0 and row['macd_hist'] < d1['macd_hist'].iloc[-3] and
            row['bb_width'] > d1['bb_width'].iloc[-3]
        )
        # Enhanced microstructure guard with volume spike detection
        vol_ratio = row['vol_ratio'] if 'vol_ratio' in row.index else (row['volume']/(row['vol_ma']+1e-9))
        
        # Volume spike detection - check if current volume is significantly above recent average
        recent_vol_avg = d1['volume'].iloc[-10:-1].mean() if len(d1) > 10 else row['vol_ma']
        vol_spike = row['volume'] > (recent_vol_avg * 2.0)  # 2x recent average indicates potential manipulation

        # Higher-timeframe directional alignment (approximate 5m trend using slower MAs on 1m)
        # Use rolling means to avoid dependency on precomputed EMA columns
        htf_ma_short = d1['close'].rolling(20).mean().iloc[-1] if len(d1) >= 20 else d1['close'].iloc[-1]
        htf_ma_long = d1['close'].rolling(50).mean().iloc[-1] if len(d1) >= 50 else d1['close'].iloc[-1]
        htf_up = htf_ma_short > htf_ma_long
        htf_down = htf_ma_short < htf_ma_long
        
        # Enhanced spread and volume guards (tiered by symbol liquidity)
        is_extended = sym in EXTENDED_SYMBOLS
        tight_spread = row.get('ob_spread_bp', 999) < (12 if is_extended else 15)
        healthy_volume = (vol_ratio >= (1.2 if is_extended else 1.1)) and not vol_spike  # Good volume but not extreme spike

        # Order book directional alignment
        ob_imbalance = row.get('ob_imbalance', 0.0)
        ob_ok_long = ob_imbalance > 0.03
        ob_ok_short = ob_imbalance < -0.03
        ob_strong_long = ob_imbalance > 0.07  # Stricter threshold for strong confirmation
        ob_strong_short = ob_imbalance < -0.07

        # Do not allow alt overrides to bypass microstructure guard
        spread_guard_long = tight_spread or healthy_volume
        
        long_score_ok = (long_score >= min_long_score_req)
        long_score_with_ob_confirm = (long_score >= min_long_score_req - 5 and ob_strong_long)

        long_ok = (
             (long_score_ok or long_score_with_ob_confirm)
             and upTrend
             and htf_up
             and (not USE_CONFIRM or conf_long)
             and row['adx'] >= ADX_THRESHOLD['weak']
             and vwap_long
             and slope_ok_long
             and pdi_now > mdi_now
             and spread_guard_long
             and (atr_ok and vol_ok)
             and (strong_breakout_up or retest_up or ha_trend_long)
             and ob_ok_long
        )

        short_score_ok = (short_score >= min_score_required)
        short_score_with_ob_confirm = (short_score >= max(0, min_score_required - 5) and ob_strong_short)

        spread_guard_short = (tight_spread if is_extended else (tight_spread or healthy_volume))
        short_ok = (
             (short_score_ok or short_score_with_ob_confirm)
             and downTrend
             and htf_down
             and (not USE_CONFIRM or conf_short)
             and row['adx'] >= ADX_THRESHOLD['moderate']
             and vwap_short
             and slope_ok_short
             and mdi_now > pdi_now
             and spread_guard_short
             and (atr_ok and vol_ok)
             and (strong_breakout_dn or retest_dn or ha_trend_short)
             and ob_ok_short
        )
        
        # Check for consecutive signals if required
        if long_ok:
            long_ok = not CONSECUTIVE_SIGNALS or track_signal(sym, 'LONG', long_score)
        if short_ok:
            short_ok = not CONSECUTIVE_SIGNALS or track_signal(sym, 'SHORT', short_score)
        
        # Momentum divergence check to avoid counter-trend signals
        if long_ok or short_ok:
            # Calculate momentum divergence using RSI and price action
            recent_prices = d1['close'].iloc[-10:]
            recent_rsi = d1['rsi'].iloc[-10:]
            
            # Check for bearish divergence (price up, RSI down) for long signals
            if long_ok:
                price_trend = recent_prices.iloc[-1] > recent_prices.iloc[-5]
                rsi_trend = recent_rsi.iloc[-1] < recent_rsi.iloc[-5]
                if price_trend and rsi_trend:
                    long_ok = False  # Bearish divergence detected, avoid long signal
            
            # Check for bullish divergence (price down, RSI up) for short signals  
            if short_ok:
                price_trend = recent_prices.iloc[-1] < recent_prices.iloc[-5]
                rsi_trend = recent_rsi.iloc[-1] > recent_rsi.iloc[-5]
                if price_trend and rsi_trend:
                    short_ok = False  # Bullish divergence detected, avoid short signal
        
        # Store score for output
        if DEBUG_LOG and not (long_ok or short_ok):
            print(
                f"DBG {sym}: scoreL={long_score:.1f} scoreS={short_score:.1f} min={min_score_required} "
                f"adx={row['adx']:.1f} vol_ratio={vol_ratio:.2f} spread={row.get('ob_spread_bp', 999)} "
                f"vwap_long={vwap_long} vwap_short={vwap_short} trend_up={upTrend} trend_down={downTrend}"
            )
        signal_score = long_score if long_ok else short_score if short_ok else 0
    else:
        # Traditional binary signal logic
        long_ok = (atr_ok and vol_ok and conf_long and upTrend and 
                  ((strong_breakout_up or retest_up) and long_core))
                  
        short_ok = (atr_ok and vol_ok and conf_short and downTrend and 
                   ((strong_breakout_dn or retest_dn) and short_core))
                   
        signal_score = 0  # Not used in binary mode

    # Generate signal with optimized risk parameters
    if long_ok:
        # Dynamic TP sizing by regime/ADX
        adx_now = row['adx']
        if regime == 'strong_trend' or adx_now >= 30:
            tp1r, tp2r, tp3r = TP1_R*1.1, TP2_R*1.2, TP3_R*1.3
        elif regime == 'choppy' or adx_now < 20:
            tp1r, tp2r, tp3r = TP1_R*0.9, TP2_R*0.85, TP3_R*0.8
        else:
            tp1r, tp2r, tp3r = TP1_R, TP2_R, TP3_R
        # Calculate stop loss (use swing low or ATR-based, capped at MAX_SL_PCT)
        atr_sl = min(price - SL_ATR*atr_v, swing_low - 0.1*atr_v if swing_low is not None else price - SL_ATR*atr_v)
        pct_sl = price * (1 - MAX_SL_PCT / 100)
        sl = max(atr_sl, pct_sl) # Use the tighter (higher) of the two stop-losses
        r = price - sl
        tp1 = price + tp1r * r
        tp2 = price + tp2r * r
        tp3 = price + tp3r * r
        
        return {'symbol': sym, 'interval': PRIMARY_IV, 'side': 'LONG',
                'price': round(price,6), 'sl': round(sl,6), 
                'tp1': round(tp1,6), 'tp2': round(tp2,6), 'tp3': round(tp3,6),
                'atr_pct': round(atr_pct,3), 'rsi': round(row['rsi'],2),
                'vol_mult': round(row['volume']/row['vol_ma'],2),
                'adx': round(row['adx'],1), 'slope': round(row['ema_fast_slope'],3),
                'vwap_side': 'above' if vwap_long else 'below',
                'sr_high': round(swing_high,6) if swing_high is not None else None,
                'sr_low': round(swing_low,6) if swing_low is not None else None,
                'score': round(signal_score) if USE_SCORE_SYSTEM else 'N/A',
                'regime': regime,
                'rr': f"{round(tp1r,2)}/{round(tp2r,2)}/{round(tp3r,2)}"}
                
    if short_ok:
        # Dynamic TP sizing by regime/ADX
        adx_now = row['adx']
        if regime == 'strong_trend' or adx_now >= 30:
            tp1r, tp2r, tp3r = TP1_R*1.1, TP2_R*1.2, TP3_R*1.3
        elif regime == 'choppy' or adx_now < 20:
            tp1r, tp2r, tp3r = TP1_R*0.9, TP2_R*0.85, TP3_R*0.8
        else:
            tp1r, tp2r, tp3r = TP1_R, TP2_R, TP3_R
        # Calculate stop loss (use swing high or ATR-based, capped at MAX_SL_PCT)
        atr_sl = max(price + SL_ATR*atr_v, swing_high + 0.1*atr_v if swing_high is not None else price + SL_ATR*atr_v)
        pct_sl = price * (1 + MAX_SL_PCT / 100)
        sl = min(atr_sl, pct_sl) # Use the tighter (lower) of the two stop-losses
        r = sl - price
        tp1 = price - tp1r * r
        tp2 = price - tp2r * r
        tp3 = price - tp3r * r
        
        return {'symbol': sym, 'interval': PRIMARY_IV, 'side': 'SHORT',
                'price': round(price,6), 'sl': round(sl,6), 
                'tp1': round(tp1,6), 'tp2': round(tp2,6), 'tp3': round(tp3,6),
                'atr_pct': round(atr_pct,3), 'rsi': round(row['rsi'],2),
                'vol_mult': round(row['volume']/row['vol_ma'],2),
                'adx': round(row['adx'],1), 'slope': round(row['ema_fast_slope'],3),
                'vwap_side': 'below' if vwap_short else 'above',
                'sr_high': round(swing_high,6) if swing_high is not None else None,
                'sr_low': round(swing_low,6) if swing_low is not None else None,
                'score': round(signal_score) if USE_SCORE_SYSTEM else 'N/A',
                'regime': regime,
                'rr': f"{round(tp1r,2)}/{round(tp2r,2)}/{round(tp3r,2)}"}
                
    return None

def scan_once():
    global SCAN_CYCLE
    SCAN_CYCLE += 1
    include_extended = (SCAN_CYCLE % EXTENDED_SCAN_EVERY == 0)
    sigs = []
    for s in SYMBOLS:
        # Stagger extended symbols to keep cycle time tight
        if (s in EXTENDED_SYMBOLS) and not include_extended:
            continue
        try:
            if not cooldown_ok(s):
                continue
            sig = make_signal(s)
            if sig:
                sigs.append(sig)
        except Exception as e:
            if not QUIET_ERRORS:
                print(f'ERR {s}: {e}')
    return sigs

# ---------- Backtesting ----------
def backtest(symbol, start_date=None, end_date=None, plot=False):
    """
    Backtest the strategy on historical data
    
    Args:
        symbol: Trading pair to backtest
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        plot: Whether to plot results
        
    Returns:
        Dictionary with backtest results
    """
    print(f"Backtesting {symbol} from {start_date} to {end_date}...")
    
    # Get historical data
    if start_date and end_date:
        # Implementation would fetch historical data from start_date to end_date
        # This is a placeholder - in a real implementation, you'd fetch from an API
        print(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
    
    # For demonstration, we'll simulate with current data
    d1 = enrich(klines(symbol, PRIMARY_IV, 500))  # Get more data for backtest
    
    # Initialize results
    trades = []
    wins = 0
    losses = 0
    total_profit_pct = 0
    max_drawdown = 0
    current_drawdown = 0
    peak_equity = 1000  # Starting equity
    current_equity = 1000
    
    # Simulate trading through the data
    for i in range(100, len(d1)-1):  # Skip first 100 bars for indicators to stabilize
        # Create a subset of data up to current bar (to prevent lookahead bias)
        d1_subset = d1.iloc[:i+1].copy()
        d5_subset = enrich(klines(symbol, CONFIRM_IV, 100))  # In real backtest, this would be aligned
        d15_subset = enrich(klines(symbol, TREND_IV, 100))   # In real backtest, this would be aligned
        
        # Detect market regime
        regime = detect_market_regime(d5_subset)
        
        # Get current bar data
        row = d1_subset.iloc[-2]  # Use second-to-last bar as current
        
        # Skip if any NaN values in key indicators
        if any(math.isnan(x) for x in [row['atr'], row['rsi'], row['ema_fast'], row['ema_slow'], row['vol_ma'], row['adx']]):
            continue
            
        # Get trend direction
        upTrend, downTrend = trend_ok(d15_subset)
        
        # Get confirmation timeframe data
        conf = d5_subset.iloc[-2]
        
        # Skip if any NaN values in confirmation indicators
        if any(math.isnan(x) for x in [conf['ema_fast'], conf['ema_slow'], conf['rsi'], conf['adx']]):
            continue
            
        # Market structure analysis
        swing_high, swing_low, hi_idx, lo_idx = last_swing_levels(d1_subset)
        
        # Breakout detection
        breakout_up = (swing_high is not None) and (row['close'] > swing_high * (1 + BREAK_K))
        breakout_dn = (swing_low is not None) and (row['close'] < swing_low * (1 - BREAK_K))
        
        # Calculate signal scores
        long_score = calculate_signal_score(d1_subset, d5_subset, d15_subset, regime, row, conf, 
                                          swing_high, swing_low, breakout_up, False, False, False, 'long')
                                           
        short_score = calculate_signal_score(d1_subset, d5_subset, d15_subset, regime, row, conf, 
                                           swing_high, swing_low, False, breakout_dn, False, False, 'short')
        
        # Check for signals
        signal = None
        if long_score >= MIN_SCORE and upTrend:
            signal = 'LONG'
        elif short_score >= MIN_SCORE and downTrend:
            signal = 'SHORT'
            
        # If we have a signal, simulate the trade
        if signal:
            entry_price = row['close']
            atr_v = row['atr']
            
            # Calculate stop loss and take profit levels
            if signal == 'LONG':
                sl = min(entry_price - SL_ATR*atr_v, swing_low - 0.1*atr_v if swing_low is not None else entry_price - SL_ATR*atr_v)
                r = entry_price - sl
                tp1 = entry_price + TP1_R * r
                tp2 = entry_price + TP2_R * r
                tp3 = entry_price + TP3_R * r
            else:  # SHORT
                sl = max(entry_price + SL_ATR*atr_v, swing_high + 0.1*atr_v if swing_high is not None else entry_price + SL_ATR*atr_v)
                r = sl - entry_price
                tp1 = entry_price - TP1_R * r
                tp2 = entry_price - TP2_R * r
                tp3 = entry_price - TP3_R * r
                
            # Simulate trade outcome using next 20 bars
            outcome = 'open'
            exit_price = entry_price
            exit_bar = i
            profit_pct = 0
            
            for j in range(i+1, min(i+21, len(d1))):
                future_bar = d1.iloc[j]
                
                if signal == 'LONG':
                    # Check for stop loss hit
                    if future_bar['low'] <= sl:
                        outcome = 'loss'
                        exit_price = sl
                        exit_bar = j
                        profit_pct = (exit_price - entry_price) / entry_price * 100
                        break
                    # Check for take profit hits
                    elif future_bar['high'] >= tp1:
                        outcome = 'win'
                        exit_price = tp1  # Simplified - in reality would be partial exits
                        exit_bar = j
                        profit_pct = (exit_price - entry_price) / entry_price * 100
                        break
                else:  # SHORT
                    # Check for stop loss hit
                    if future_bar['high'] >= sl:
                        outcome = 'loss'
                        exit_price = sl
                        exit_bar = j
                        profit_pct = (entry_price - exit_price) / entry_price * 100
                        break
                    # Check for take profit hits
                    elif future_bar['low'] <= tp1:
                        outcome = 'win'
                        exit_price = tp1  # Simplified - in reality would be partial exits
                        exit_bar = j
                        profit_pct = (entry_price - exit_price) / entry_price * 100
                        break
            
            # If trade still open after 20 bars, close at current price
            if outcome == 'open':
                exit_price = d1.iloc[min(i+20, len(d1)-1)]['close']
                exit_bar = min(i+20, len(d1)-1)
                if signal == 'LONG':
                    profit_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - exit_price) / entry_price * 100
                outcome = 'win' if profit_pct > 0 else 'loss'
            
            # Update statistics
            if outcome == 'win':
                wins += 1
            else:
                losses += 1
                
            total_profit_pct += profit_pct
            
            # Update equity curve
            trade_profit_dollars = current_equity * (profit_pct / 100)
            current_equity += trade_profit_dollars
            
            # Update drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
                current_drawdown = 0
            else:
                current_drawdown = (peak_equity - current_equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, current_drawdown)
            
            # Record trade
            trades.append({
                'entry_bar': i,
                'exit_bar': exit_bar,
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'sl': sl,
                'tp1': tp1,
                'outcome': outcome,
                'profit_pct': profit_pct,
                'score': long_score if signal == 'LONG' else short_score,
                'regime': regime
            })
    
    # Calculate results
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_profit = total_profit_pct / total_trades if total_trades > 0 else 0
    profit_factor = sum(t['profit_pct'] for t in trades if t['profit_pct'] > 0) / abs(sum(t['profit_pct'] for t in trades if t['profit_pct'] <= 0)) if sum(t['profit_pct'] for t in trades if t['profit_pct'] <= 0) != 0 else float('inf')
    
    # Print results
    print(f"\nBacktest Results for {symbol}:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Profit: {avg_profit:.2f}%")
    print(f"Total Profit: {total_profit_pct:.2f}%")
    print(f"Final Equity: ${current_equity:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Return results
    return {
        'symbol': symbol,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'total_profit_pct': total_profit_pct,
        'final_equity': current_equity,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'trades': trades
    }

# ---------- Runner ----------
if __name__ == '__main__':
    print('Advanced Crypto Scalping Scanner v5 (High Accuracy). Not financial advice.')
    print(f'Monitoring {len(SYMBOLS)} symbols with {MIN_SCORE}+ signal score threshold.')
    print(f'Extended symbols scanned every {EXTENDED_SCAN_EVERY} cycles.')
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        # Run backtest mode
        if len(sys.argv) > 2:
            symbol = sys.argv[2]
            start_date = sys.argv[3] if len(sys.argv) > 3 else None
            end_date = sys.argv[4] if len(sys.argv) > 4 else None
            backtest(symbol, start_date, end_date)
        else:
            print("Usage: python scanner.py --backtest SYMBOL [START_DATE] [END_DATE]")
            print("Example: python scanner.py --backtest BTCUSDT 2023-01-01 2023-12-31")
            sys.exit(1)
    else:
        # Run live scanner mode
        print("Running in live scanning mode. Press Ctrl+C to exit.")
        try:
            while True:
                try:
                    emits = scan_once()
                    if emits:
                        for s in emits:
                            mark_emit(s['symbol'])
                            print(f"{s['symbol']} {s['interval']} {s['side']} @ {s['price']} | SL {s['sl']} TP1 {s['tp1']} TP2 {s['tp2']} TP3 {s['tp3']} "
                                  f"| ATR% {s['atr_pct']} RSI {s['rsi']} ADX {s['adx']} Slope% {s['slope']} VWAP {s['vwap_side']} "
                                  f"| Volx {s['vol_mult']} | S:{s['sr_low']} R:{s['sr_high']} | Score: {s['score']} | Regime: {s['regime']} | R:R {s['rr']}")
                    time.sleep(SLEEP_SEC)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    time.sleep(SLEEP_SEC)
        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
            sys.exit(0)
