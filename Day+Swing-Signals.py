# pip install requests pandas numpy ta-lib
import time, math, requests
import os, csv
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

LOG_FILE = 'signals.log'
PRINT_SIGNALS = True  # print concise signals to console when emitted

def log_signal(s, log_file=LOG_FILE):
    try:
        file_exists = os.path.exists(log_file)
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'ts','symbol','mode','interval','side','price','sl','tp1','tp2','tp3',
                    'atr_pct','rsi','adx','slope','vwap_side','vol_mult','sr_low','sr_high','score','regime','rr'
                ])
            writer.writerow([
                datetime.utcnow().isoformat(),
                s['symbol'], s.get('mode','day'), s['interval'], s['side'], s['price'], s['sl'], s['tp1'], s['tp2'], s['tp3'],
                s['atr_pct'], s['rsi'], s['adx'], s['slope'], s['vwap_side'], s['vol_mult'], s['sr_low'], s['sr_high'], s['score'], s['regime'], s['rr']
            ])
        if PRINT_SIGNALS:
            try:
                print(
                    f"SIGNAL {s['symbol']} {s['side']} @ {float(s['price']):.4f} "
                    f"SL {float(s['sl']):.4f} TP1 {float(s['tp1']):.4f} "
                    f"score {int(s['score'])} rr {s.get('rr','-')} interval {s['interval']}"
                )
            except Exception:
                pass
    except Exception as e:
        try:
            if DEBUG:
                print(f"[DBG] log write failed: {e}")
        except:
            pass

# ---------- Config (Advanced for high win rate) ----------
# Last updated: November 3, 2025
SYMBOLS = [
'BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','DOGEUSDT','XRPUSDT','ADAUSDT','LINKUSDT',
'AVAXUSDT','NEARUSDT','SEIUSDT','APTUSDT','ARBUSDT','OPUSDT','DOTUSDT','ATOMUSDT','SUIUSDT','ZENUSDT',
'MATICUSDT','LTCUSDT','INJUSDT','FETUSDT','RNDRUSDT','UNIUSDT','FTMUSDT','SANDUSDT','MANAUSDT','GMTUSDT',
'BCHUSDT','TRXUSDT','TONUSDT','SHIBUSDT','AAVEUSDT','IMXUSDT','DYDXUSDT','FILUSDT','XLMUSDT','STXUSDT','RUNEUSDT','TIAUSDT'
]
PRIMARY_IV = '15m'
CONFIRM_IV = '1h'
TREND_IV = '4h'
LOOKBACK = 500
SLEEP_SEC = 30

TRADING_MODE = 'swing'  # options: 'day', 'swing'

# Default settings (day trading)
PRIMARY_IV = '15m'
CONFIRM_IV = '1h'
TREND_IV = '4h'
SL_ATR = 1.60
TP1_R = 1.0
TP2_R = 2.5
TP3_R = 4.0
EMA_SLOPE_MIN = 0.018
COOLDOWN_MIN = 10
LOOKBACK = 500
MIN_SCORE = 80



# Advanced Indicators
USE_TREND = True         # require 15m trend alignment
USE_CONFIRM = True       # require 5m confirmation
USE_MARKET_REGIME = True # adapt to market conditions
USE_SCORE_SYSTEM = True  # use scoring instead of binary decisions
USE_MTF_CONFLUENCE = True # multi-timeframe confluence system
USE_SMART_MONEY = True   # smart money concepts
USE_VELOCITY_FILTERS = True # momentum velocity and acceleration

# Volatility Settings
ATR_LEN = 14
ATR_MIN_PCT = 0.06       # stricter volatility requirement for cleaner moves
ATR_FILTER_MULT = 1.2    # for dynamic ATR filtering
ATR_THRESHOLD = {
    'low': 0.04,         # low volatility threshold
    'moderate': 0.05,    # moderate volatility threshold
    'high': 0.08         # high volatility threshold
}

# Volume Settings
VOL_MA_LEN = 20
VOL_MIN_MULT = 1.2       # require at least 20% volume surge
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

# Trend Strength
ADX_LEN = 14
ADX_THRESHOLD = {
    'weak': 15,
    'moderate': 25,      # stricter for stronger trend requirement
    'strong': 35
}
EMA_SLOPE_MIN = 0.01    # slightly reduced momentum requirement

# Market Structure
SR_LOOKBACK = 60
SWING_WINDOW = 3
BREAK_K = 0.0018         # breakout threshold (0.18%)
RETEST_BARS = 6
LIQUIDITY_ZONES = True   # identify liquidity zones
LIQUIDITY_LOOKBACK = 50  # bars to look back for liquidity zones

# Risk Management
SL_ATR = 1.60            # much wider stop to avoid premature exits
TP1_R = 1.0              # balanced partials
TP2_R = 2.5              # wider targets
TP3_R = 4.0
PARTIAL_TP = True        # enable partial take profit
PARTIAL_EXIT_1 = 0.6     # exit 60% at TP1 (secure more profit early)
PARTIAL_EXIT_2 = 0.25    # exit 25% at TP2

# Signal Quality (Ultra High Accuracy)
MIN_SCORE = 80           # lowered for diagnostics
MIN_SCORE_VOLATILE = 85  # slightly lowered for volatile conditions
MIN_SCORE_CONFLUENCE = 80 # align with diagnostic threshold
COOLDOWN_MIN = 10        # shorter cooldown for more frequent checks
CONSECUTIVE_SIGNALS = 1  # single confirmation in diagnostics
HISTORY_WINDOW_MIN = 30  # 30min window for tracking signals
CONFLUENCE_WEIGHT = 1.5  # weight multiplier for confluence signals

# Momentum Filters (NEW)
MIN_MOMENTUM_CHANGE = 0.005  # relaxed for diagnostics
MIN_VOLUME_SPIKE = 1.15      # relaxed for diagnostics
RSI_MOMENTUM_MIN = 3         # relaxed for diagnostics
MACD_MOMENTUM_MIN = 0.06     # relaxed for diagnostics

# Market Hours (avoid low liquidity periods)
AVOID_LOW_LIQUIDITY = False
LOW_LIQUIDITY_HOURS = [
    (0, 2),              # UTC hours to avoid (low volume)
    (22, 24)
]

BASE = 'https://fapi.binance.com'

# Debugging
DEBUG = False

def dbg(sym, msg):
    if DEBUG:
        try:
            ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            ts = str(time.time())
        print(f"[DEBUG {ts}] {sym}: {msg}")

# ---------- Advanced Data Helpers ----------
def klines(symbol, interval, limit):
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

def validate_momentum(df):
    """Validate that there's sufficient momentum to avoid sideways entries"""
    close = df['close']
    volume = df['volume']
    vol_ma = df['vol_ma']
    
    # 1. Price momentum check - last 2 candles
    price_change_1 = abs(close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    price_change_2 = abs(close.iloc[-2] - close.iloc[-3]) / close.iloc[-3] * 100
    avg_price_change = (price_change_1 + price_change_2) / 2
    
    if avg_price_change < MIN_MOMENTUM_CHANGE:
        return False, f"Insufficient price momentum: {avg_price_change:.2f}% < {MIN_MOMENTUM_CHANGE}%"
    
    # 2. Volume spike check
    recent_vol_avg = volume.iloc[-3:].mean()
    vol_ma_recent = vol_ma.iloc[-1]
    vol_spike = recent_vol_avg / vol_ma_recent
    
    if vol_spike < MIN_VOLUME_SPIKE:
        return False, f"Insufficient volume spike: {vol_spike:.2f}x < {MIN_VOLUME_SPIKE}x"
    
    # 3. RSI momentum check
    rsi_val = df['rsi'].iloc[-1]
    rsi_prev = df['rsi'].iloc[-2]
    rsi_change = abs(rsi_val - 50)  # Distance from neutral
    rsi_momentum = abs(rsi_val - rsi_prev)
    
    if rsi_change < RSI_MOMENTUM_MIN:
        return False, f"RSI too neutral: {rsi_change:.1f} < {RSI_MOMENTUM_MIN}"
    
    # 4. MACD momentum check
    macd_hist = df['macd_hist'].iloc[-1]
    if abs(macd_hist) < MACD_MOMENTUM_MIN:
        return False, f"MACD momentum too weak: {abs(macd_hist):.3f} < {MACD_MOMENTUM_MIN}"
    
    return True, "Momentum validated"

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
    
    # Heikin-Ashi transformation
    ha = heikin_ashi(df)
    df['ha_open'] = ha['ha_open']
    df['ha_high'] = ha['ha_high']
    df['ha_low'] = ha['ha_low']
    df['ha_close'] = ha['ha_close']
    df['ha_ema_fast'] = ema(df['ha_close'], HAMA_FAST)
    df['ha_ema_slow'] = ema(df['ha_close'], HAMA_SLOW)
    
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
    
    # Momentum
    df['ema_fast_slope'] = (df['ema_fast'] - df['ema_fast'].shift(1)) / (df['close'] + 1e-9) * 100
    df['ema_mid_slope'] = (df['ema_mid'] - df['ema_mid'].shift(1)) / (df['close'] + 1e-9) * 100
    # Generic slope alias for stronger trend alignment checks
    df['ema_slope'] = df['ema_fast_slope']

    # MACD (12/26/9)
    macd_fast = ema(df['close'], 12)
    macd_slow = ema(df['close'], 26)
    df['macd'] = macd_fast - macd_slow
    df['macd_signal'] = ema(df['macd'], 9)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Price action
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = df['body_size'] / (df['candle_range'] + 1e-9)
    df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
    
    return df

def trend_ok(df15):
    """Advanced trend detection using multiple indicators"""
    try:
        # Check if we have enough data
        if len(df15) < 3:
            return False, False
            
        # Check if required columns exist
        required_cols = ['close', 'ema_fast', 'ema_mid']
        for col in required_cols:
            if col not in df15.columns:
                return False, False
        
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
    except (IndexError, KeyError, AttributeError) as e:
        # Return neutral trend if any error occurs
        return False, False

def is_low_liquidity_period():
    """Check if current time is in low liquidity period"""
    if not AVOID_LOW_LIQUIDITY:
        return False
        
    now_utc = datetime.utcnow().hour
    for start, end in LOW_LIQUIDITY_HOURS:
        if start <= now_utc < end:
            return True
    return False

# Structure helpers (global)

def detect_fvg(df, lookback=20):
    """Detect Fair Value Gap (FVG) using a simple 3-candle pattern"""
    try:
        if len(df) < 3:
            return False, False
        # Check if required columns exist
        required_cols = ['low', 'high']
        for col in required_cols:
            if col not in df.columns:
                return False, False
        i = len(df) - 2
        bull = df['low'].iloc[i+1] > df['high'].iloc[i-1]
        bear = df['high'].iloc[i+1] < df['low'].iloc[i-1]
        return bull, bear
    except (IndexError, KeyError, AttributeError):
        return False, False


def detect_choch(d1, swing_high, swing_low):
    """Detect Change of Character (ChoCh) relative to recent swing levels"""
    if swing_high is None or swing_low is None:
        return False, False
    try:
        row = d1.iloc[-2]
        # Check if required columns exist
        if 'close' not in row or 'ema_fast' not in row or 'ema_mid' not in row:
            return False, False
        bull_choch = (row['close'] > swing_high) and (row['ema_fast'] > row['ema_mid'])
        bear_choch = (row['close'] < swing_low) and (row['ema_fast'] < row['ema_mid'])
        return bull_choch, bear_choch
    except (IndexError, KeyError, AttributeError):
        return False, False

# ---------- Advanced Multi-Timeframe Confluence System ----------

def calculate_mtf_confluence(d1, d5, d15, d1h, d4h):
    """Calculate multi-timeframe confluence score for ultra-high accuracy"""
    confluence_score = 0
    max_score = 100
    
    # Get current data for each timeframe
    current_1m = d1.iloc[-1] if len(d1) > 0 else None
    current_5m = d5.iloc[-1] if len(d5) > 0 else None
    current_15m = d15.iloc[-1] if len(d15) > 0 else None
    current_1h = d1h.iloc[-1] if len(d1h) > 0 else None
    current_4h = d4h.iloc[-1] if len(d4h) > 0 else None
    
    if not all([current_1m is not None, current_5m is not None, current_15m is not None]):
        return 0, 0  # bull_confluence, bear_confluence
    
    # 1. Trend Alignment (30 points)
    trend_alignment_bull = 0
    trend_alignment_bear = 0
    
    timeframes = [
        (current_1m, 5),   # 1m gets 5 points
        (current_5m, 8),   # 5m gets 8 points  
        (current_15m, 10), # 15m gets 10 points
        (current_1h, 7) if current_1h is not None else (None, 0),   # 1h gets 7 points
    ]
    
    for tf_data, points in timeframes:
        if tf_data is None:
            continue
        if tf_data['ema_fast'] > tf_data['ema_slow'] and tf_data.get('ema_slope', 0) > 0:
            trend_alignment_bull += points
        if tf_data['ema_fast'] < tf_data['ema_slow'] and tf_data.get('ema_slope', 0) < 0:
            trend_alignment_bear += points
    
    # 2. Momentum Confluence (25 points)
    momentum_bull = 0
    momentum_bear = 0
    
    for tf_data, points in timeframes:
        if tf_data is None:
            continue
        rsi = tf_data.get('rsi', 50)
        macd_hist = tf_data.get('macd_hist', 0)
        
        # Bullish momentum
        if rsi > 55 and macd_hist > 0:
            momentum_bull += points * 0.8
        # Bearish momentum  
        if rsi < 45 and macd_hist < 0:
            momentum_bear += points * 0.8
    
    # 3. Volume Confluence (20 points)
    volume_bull = 0
    volume_bear = 0
    
    for tf_data, points in timeframes:
        if tf_data is None:
            continue
        vol_ratio = tf_data.get('volume', 0) / max(tf_data.get('vol_ma', 1), 1)
        if vol_ratio > 1.3:  # Above average volume
            if tf_data['close'] > tf_data['open']:  # Bullish candle
                volume_bull += points * 0.6
            else:  # Bearish candle
                volume_bear += points * 0.6
    
    # 4. Price Action Confluence (25 points)
    pa_bull = 0
    pa_bear = 0
    
    for tf_data, points in timeframes:
        if tf_data is None:
            continue
        
        # Bullish price action
        if (tf_data['close'] > tf_data['ema_fast'] and 
            tf_data['low'] > tf_data['ema_mid'] * 0.998):  # Staying above key level
            pa_bull += points * 0.8
            
        # Bearish price action
        if (tf_data['close'] < tf_data['ema_fast'] and 
            tf_data['high'] < tf_data['ema_mid'] * 1.002):  # Staying below key level
            pa_bear += points * 0.8
    
    # Calculate final confluence scores
    bull_confluence = min(100, trend_alignment_bull + momentum_bull + volume_bull + pa_bull)
    bear_confluence = min(100, trend_alignment_bear + momentum_bear + volume_bear + pa_bear)
    
    return bull_confluence, bear_confluence

def calculate_smart_money_score(d1, d5, d15):
    """Calculate smart money concepts score"""
    if len(d1) < 20 or len(d5) < 20:
        return 0, 0
    
    bull_sm = 0
    bear_sm = 0
    
    # 1. Liquidity Sweeps (40 points)
    recent_high = d1['high'].iloc[-10:].max()
    recent_low = d1['low'].iloc[-10:].min()
    current_price = d1['close'].iloc[-1]
    
    # Bullish liquidity sweep (sweep lows then reverse up)
    if (d1['low'].iloc[-3:].min() < recent_low * 0.999 and 
        current_price > recent_low * 1.002):
        bull_sm += 40
    
    # Bearish liquidity sweep (sweep highs then reverse down)
    if (d1['high'].iloc[-3:].max() > recent_high * 1.001 and 
        current_price < recent_high * 0.998):
        bear_sm += 40
    
    # 2. Order Block Formation (35 points)
    # Look for strong rejection candles
    for i in range(-5, -1):
        if i >= -len(d1):
            candle = d1.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            wick_size_up = candle['high'] - max(candle['open'], candle['close'])
            wick_size_down = min(candle['open'], candle['close']) - candle['low']
            
            # Bullish order block (strong rejection from low)
            if wick_size_down > body_size * 2 and candle['close'] > candle['open']:
                bull_sm += 20
                
            # Bearish order block (strong rejection from high)
            if wick_size_up > body_size * 2 and candle['close'] < candle['open']:
                bear_sm += 20
    
    # 3. Fair Value Gap (25 points)
    fvg_bull, fvg_bear = detect_fvg(d1)
    if fvg_bull:
        bull_sm += 25
    if fvg_bear:
        bear_sm += 25
    
    return min(100, bull_sm), min(100, bear_sm)

def calculate_velocity_score(d1, d5):
    """Calculate momentum velocity and acceleration score"""
    if len(d1) < 10 or len(d5) < 5:
        return 0, 0
    
    bull_vel = 0
    bear_vel = 0
    
    # 1. Price Velocity (50 points)
    price_change_1m = (d1['close'].iloc[-1] - d1['close'].iloc[-3]) / d1['close'].iloc[-3] * 100
    price_change_5m = (d5['close'].iloc[-1] - d5['close'].iloc[-2]) / d5['close'].iloc[-2] * 100
    
    if price_change_1m > 0.2 and price_change_5m > 0.1:  # Strong upward velocity
        bull_vel += 50
    if price_change_1m < -0.2 and price_change_5m < -0.1:  # Strong downward velocity
        bear_vel += 50
    
    # 2. Volume Acceleration (30 points)
    vol_current = d1['volume'].iloc[-3:].mean()
    vol_previous = d1['volume'].iloc[-10:-3].mean()
    vol_acceleration = vol_current / max(vol_previous, 1)
    
    if vol_acceleration > 1.5:
        if price_change_1m > 0:
            bull_vel += 30
        else:
            bear_vel += 30
    
    # 3. RSI Momentum (20 points)
    rsi_current = d1['rsi'].iloc[-1]
    rsi_previous = d1['rsi'].iloc[-3]
    rsi_change = rsi_current - rsi_previous
    
    if rsi_change > 8 and rsi_current < 75:  # Strong bullish momentum, not overbought
        bull_vel += 20
    if rsi_change < -8 and rsi_current > 25:  # Strong bearish momentum, not oversold
        bear_vel += 20
    
    return min(100, bull_vel), min(100, bear_vel)

def calculate_ml_signal_score(d1, d5, d15, current_score, side):
    """
    Machine Learning-inspired signal scoring based on pattern recognition
    Uses feature engineering and weighted scoring for signal quality assessment
    """
    try:
        ml_score = 0
        
        # Feature 1: Multi-timeframe EMA alignment strength (25 points)
        ema_alignment_score = 0
        timeframes = [(d1, 0.3), (d5, 0.4), (d15, 0.3)]
        
        for df, weight in timeframes:
            if df is None or len(df) < 20:
                continue
            ema_fast = df['ema_fast'].iloc[-1]
            ema_slow = df['ema_slow'].iloc[-1]
            ema_diff = abs(ema_fast - ema_slow) / ema_slow * 100
            
            if side == 'LONG' and ema_fast > ema_slow:
                ema_alignment_score += min(10, ema_diff * 2) * weight
            elif side == 'SHORT' and ema_fast < ema_slow:
                ema_alignment_score += min(10, ema_diff * 2) * weight
        
        ml_score += ema_alignment_score
        
        # Feature 2: RSI divergence and momentum (20 points)
        if len(d1) >= 10:
            rsi_values = d1['rsi'].iloc[-10:].values
            price_values = d1['close'].iloc[-10:].values
            
            # Calculate momentum consistency
            rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
            price_trend = np.polyfit(range(len(price_values)), price_values, 1)[0]
            
            if side == 'LONG' and rsi_trend > 0 and price_trend > 0:
                ml_score += 20
            elif side == 'SHORT' and rsi_trend < 0 and price_trend < 0:
                ml_score += 20
        
        # Feature 3: Volume pattern analysis (20 points)
        if len(d1) >= 5:
            recent_vol = d1['volume'].iloc[-3:].mean()
            avg_vol = d1['volume'].iloc[-20:].mean()
            vol_ratio = recent_vol / max(avg_vol, 1)
            
            # Volume confirmation
            if vol_ratio > 1.5:
                ml_score += 15
            elif vol_ratio > 1.2:
                ml_score += 10
            
            # Volume-price relationship
            recent_price_change = (d1['close'].iloc[-1] - d1['close'].iloc[-3]) / d1['close'].iloc[-3]
            if (side == 'LONG' and recent_price_change > 0 and vol_ratio > 1.3) or \
               (side == 'SHORT' and recent_price_change < 0 and vol_ratio > 1.3):
                ml_score += 5
        
        # Feature 4: Volatility-adjusted momentum (15 points)
        if len(d1) >= 14:
            atr = d1['atr'].iloc[-1]
            price = d1['close'].iloc[-1]
            atr_pct = atr / price * 100
            
            # Optimal volatility range for scalping
            if 0.5 <= atr_pct <= 2.0:
                ml_score += 15
            elif 0.3 <= atr_pct <= 3.0:
                ml_score += 10
        
        # Feature 5: Pattern consistency score (20 points)
        pattern_score = 0
        if len(d1) >= 20:
            # Check for consistent higher lows (LONG) or lower highs (SHORT)
            lows = d1['low'].iloc[-10:].values
            highs = d1['high'].iloc[-10:].values
            
            if side == 'LONG':
                # Count higher lows in recent candles
                higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
                pattern_score = min(20, higher_lows * 3)
            else:
                # Count lower highs in recent candles
                lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
                pattern_score = min(20, lower_highs * 3)
        
        ml_score += pattern_score
        
        # Normalize and combine with existing score
        ml_score = min(100, ml_score)
        
        # Weighted combination: 70% traditional score + 30% ML score
        final_score = current_score * 0.7 + ml_score * 0.3
        
        return min(100, final_score)
        
    except Exception as e:
        print(f"ML scoring error: {e}")
        return current_score  # Fallback to original score

def analyze_market_microstructure(d1, d5):
    """
    Analyze market microstructure for institutional activity and liquidity patterns
    Returns microstructure score (0-100) for both bull and bear scenarios
    """
    try:
        micro_bull = 0
        micro_bear = 0
        
        if len(d1) < 20 or len(d5) < 10:
            return 0, 0
        
        # 1. Order Flow Analysis (30 points)
        # Analyze candle body vs wick ratios for institutional footprints
        recent_candles = d1.iloc[-10:]
        
        for _, candle in recent_candles.iterrows():
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Strong bullish institutional candles
                if (candle['close'] > candle['open'] and body_ratio > 0.7 and 
                    lower_wick < body_size * 0.2):
                    micro_bull += 3
                
                # Strong bearish institutional candles
                if (candle['close'] < candle['open'] and body_ratio > 0.7 and 
                    upper_wick < body_size * 0.2):
                    micro_bear += 3
        
        # 2. Liquidity Grab Detection (25 points)
        # Look for stop hunt patterns followed by reversal
        highs = d1['high'].iloc[-15:].values
        lows = d1['low'].iloc[-15:].values
        closes = d1['close'].iloc[-15:].values
        
        # Bullish liquidity grab (sweep lows then rally)
        for i in range(5, len(lows)-2):
            if (lows[i] < min(lows[i-5:i]) and  # New low
                closes[i+1] > closes[i] and closes[i+2] > closes[i+1]):  # Recovery
                micro_bull += 12
        
        # Bearish liquidity grab (sweep highs then drop)
        for i in range(5, len(highs)-2):
            if (highs[i] > max(highs[i-5:i]) and  # New high
                closes[i+1] < closes[i] and closes[i+2] < closes[i+1]):  # Decline
                micro_bear += 12
        
        # 3. Volume Profile Analysis (25 points)
        # Analyze volume distribution for institutional accumulation/distribution
        vol_ma = d1['volume'].iloc[-20:].mean()
        recent_vol = d1['volume'].iloc[-5:].mean()
        vol_trend = recent_vol / max(vol_ma, 1)
        
        price_change_5min = (d1['close'].iloc[-1] - d1['close'].iloc[-5]) / d1['close'].iloc[-5]
        
        # Accumulation pattern (rising volume + rising price)
        if vol_trend > 1.3 and price_change_5min > 0.002:
            micro_bull += 25
        
        # Distribution pattern (rising volume + falling price)
        if vol_trend > 1.3 and price_change_5min < -0.002:
            micro_bear += 25
        
        # 4. Time and Sales Simulation (20 points)
        # Simulate aggressive buying/selling based on candle patterns
        for i in range(-5, 0):
            candle = d1.iloc[i]
            prev_candle = d1.iloc[i-1] if i > -len(d1) else None
            
            if prev_candle is not None:
                # Gap up with volume (institutional buying)
                if (candle['open'] > prev_candle['high'] and 
                    candle['volume'] > d1['volume'].iloc[-20:].mean() * 1.5):
                    micro_bull += 4
                
                # Gap down with volume (institutional selling)
                if (candle['open'] < prev_candle['low'] and 
                    candle['volume'] > d1['volume'].iloc[-20:].mean() * 1.5):
                    micro_bear += 4
        
        return min(100, micro_bull), min(100, micro_bear)
        
    except Exception as e:
        print(f"Microstructure analysis error: {e}")
        return 0, 0

# ---------- Signal logic ----------
EMIT_COOLDOWNS = defaultdict(lambda: defaultdict(float))
_signal_history = {}

def cooldown_ok(sym, mode):
    """Check if the cooldown period for a given symbol and mode has passed."""
    last_emit = EMIT_COOLDOWNS[sym][mode]
    return time.time() - last_emit > COOLDOWN_MIN * 60

def mark_emit(sym, mode):
    """Mark that a signal has been emitted for a given symbol and mode."""
    EMIT_COOLDOWNS[sym][mode] = time.time()
    
HISTORY_WINDOW_MIN = 30  # extend confirmation window for day trading

def track_signal(sym, side, score):
    """Track signal history for consecutive signal confirmation"""
    now_min = int(time.time() // 60)
    if sym not in _signal_history:
        _signal_history[sym] = []
    
    # Add new signal to history
    _signal_history[sym].append((now_min, side, score))
    
    # Clean up old signals (older than HISTORY_WINDOW_MIN)
    _signal_history[sym] = [s for s in _signal_history[sym] if now_min - s[0] <= HISTORY_WINDOW_MIN]
    
    # Check for consecutive signals
    if len(_signal_history[sym]) >= CONSECUTIVE_SIGNALS:
        recent_signals = _signal_history[sym][-CONSECUTIVE_SIGNALS:]
        sides = [s[1] for s in recent_signals]
        return all(s == side for s in sides)
    
    return False

def calculate_signal_score(d1, d5, d15, regime, row, conf, swing_high, swing_low, breakout_up, breakout_dn, retest_up, retest_dn):
    """Day-trade signal score (0-100) with strong HTF alignment and structure)"""
    score = 0

    # Higher timeframe alignment (4h) — up to 25 points
    trend = d15.iloc[-2]
    htf_bull = (trend['ema_fast'] > trend['ema_slow']) and (trend['close'] > trend['vwap'])
    htf_bear = (trend['ema_fast'] < trend['ema_slow']) and (trend['close'] < trend['vwap'])
    if htf_bull:
        score += 20
        score += 5 if trend['adx'] >= ADX_THRESHOLD['moderate'] else 0
    if htf_bear:
        score += 20
        score += 5 if trend['adx'] >= ADX_THRESHOLD['moderate'] else 0

    # Structure signals: FVG and ChoCh
    def _detect_fvg(df):
        try:
            a = df.iloc[-3]
            b = df.iloc[-2]
            c = df.iloc[-1]
            bull = (c['low'] > a['high']) and (b['low'] > a['high'])
            bear = (c['high'] < a['low']) and (b['high'] < a['low'])
            return bull, bear
        except Exception:
            return False, False

    def _detect_choch(df, swing_high, swing_low):
        try:
            close = df.iloc[-1]['close']
            prev_close = df.iloc[-2]['close']
            bull = (close > swing_high) and (prev_close <= swing_high)
            bear = (close < swing_low) and (prev_close >= swing_low)
            return bull, bear
        except Exception:
            return False, False

    fvg_bull, fvg_bear = _detect_fvg(d1)
    choch_bull, choch_bear = _detect_choch(d1, swing_high, swing_low)

    # Base score from primary timeframe (15m)
    if row['ema_fast'] > row['ema_slow']:  # Bullish
        # Trend alignment (20 points)
        score += 8 if row['ema_fast'] > row['ema_mid'] else 0
        score += 6 if row['ema_mid'] > row['ema_slow'] else 0
        score += 3 if row['close'] > row['ema_fast'] else 0
        score += 3 if row['close'] > row['vwap'] else 0

        # Oscillators (15 points) — less weight for day trades
        score += min(8, max(0, (row['rsi'] - 50) / 2))
        score += 4 if row['stoch_k'] > row['stoch_d'] else 0
        score += 3 if row['stoch_k'] > 50 else 0

        # Momentum (15 points)
        score += min(10, max(0, row['ema_fast_slope'] / 0.006))
        score += 5 if row['ha_close'] > row['ha_open'] else 0

        # Volume (10 points)
        score += min(10, max(0, (row['vol_ratio'] - 1) * 8))

        # Breakout/Retest (15 points)
        if breakout_up:
            score += 5
        if retest_up:
            score += 15
        
        # Structure boosts
        if fvg_bull:
            score += 8
        if choch_bull:
            score += 10
        
        # Exhaustion penalty
        if row['body_ratio'] > 0.7:
            score -= 5

        # Confirmation timeframe (1h) (20 points)
        if conf['ema_fast'] > conf['ema_slow']:
            score += 12
        if conf['rsi'] > 55:
            score += 8

    else:  # Bearish
        # Trend alignment (20 points)
        score += 8 if row['ema_fast'] < row['ema_mid'] else 0
        score += 6 if row['ema_mid'] < row['ema_slow'] else 0
        score += 3 if row['close'] < row['ema_fast'] else 0
        score += 3 if row['close'] < row['vwap'] else 0

        # Oscillators (15 points)
        score += min(8, max(0, (50 - row['rsi']) / 2))
        score += 4 if row['stoch_k'] < row['stoch_d'] else 0
        score += 3 if row['stoch_k'] < 50 else 0

        # Momentum (15 points)
        score += min(10, max(0, -row['ema_fast_slope'] / 0.006))
        score += 5 if row['ha_close'] < row['ha_open'] else 0

        # Volume (10 points)
        score += min(10, max(0, (row['vol_ratio'] - 1) * 8))

        # Breakout/Retest (15 points)
        if breakout_dn:
            score += 5
        if retest_dn:
            score += 15
        
        # Structure boosts
        if fvg_bear:
            score += 8
        if choch_bear:
            score += 10
        
        # Exhaustion penalty
        if row['body_ratio'] > 0.7:
            score -= 5

        # Confirmation timeframe (1h) (20 points)
        if conf['ema_fast'] < conf['ema_slow']:
            score += 12
        if conf['rsi'] < 45:
            score += 8

    # Market regime adjustment
    if regime == 'strong_trend':
        score *= 1.15
    elif regime == 'choppy':
        score *= 0.75

    return min(100, score)

def make_signal(sym, trading_mode):
    

    # Skip during low liquidity periods
    if is_low_liquidity_period():
        dbg(sym, 'Low-liquidity hours; skipping')
        return None
    
    # Get data for multiple timeframes (including higher timeframes for confluence)
    try:
        d1 = enrich(klines(sym, PRIMARY_IV, LOOKBACK))
        d5 = enrich(klines(sym, CONFIRM_IV, LOOKBACK))
        d15 = enrich(klines(sym, TREND_IV, LOOKBACK))  # Correctly fetch 15m data
    
    # Get higher timeframes for advanced confluence analysis
    except Exception as e:
        dbg(sym, f'Error fetching klines: {e}')
        return None
    # Initialize MTF variables so later checks don't raise UnboundLocalError
    d1h = None
    d4h = None
    if USE_MTF_CONFLUENCE:
        try:
            d1h = enrich(klines(sym, CONFIRM_IV, min(200, LOOKBACK)))
            d4h = enrich(klines(sym, TREND_IV, min(100, LOOKBACK)))
        except:
            pass  # Continue without higher timeframes if unavailable
    
    # Detect market regime
    regime = detect_market_regime(d5)
    
    # Get trend direction with stronger alignment requirement
    upTrend, downTrend = trend_ok(d15) # Use 15-minute data for trend
    
    # Additional trend strength validation - require alignment across all timeframes
    d1_trend_up = d1.iloc[-1]['ema_fast'] > d1.iloc[-1]['ema_slow']
    d5_trend_up = d5.iloc[-1]['ema_fast'] > d5.iloc[-1]['ema_slow']
    d15_trend_up = d15.iloc[-1]['ema_fast'] > d15.iloc[-1]['ema_slow']
    
    d1_trend_dn = d1.iloc[-1]['ema_fast'] < d1.iloc[-1]['ema_slow']
    d5_trend_dn = d5.iloc[-1]['ema_fast'] < d5.iloc[-1]['ema_slow']
    d15_trend_dn = d15.iloc[-1]['ema_fast'] < d15.iloc[-1]['ema_slow']
    
    # Require multi-timeframe alignment (less strict)
    strong_upTrend = upTrend and (d1_trend_up or d5_trend_up) and d15_trend_up
    strong_downTrend = downTrend and (d1_trend_dn or d5_trend_dn) and d15_trend_dn
    
    # Volatility spike detection (safe against NaN/zero)
    recent_atr = d1['atr'].iloc[-3:].dropna().mean()
    avg_atr = d1['atr'].iloc[-20:].dropna().mean()
    volatile_ratio = 0.0
    if (
        recent_atr is not None and avg_atr is not None and
        not math.isnan(recent_atr) and not math.isnan(avg_atr) and
        avg_atr > 0
    ):
        volatile_ratio = recent_atr / avg_atr
    # Require stronger spikes for day mode to avoid chop
    volatility_spike = volatile_ratio > (1.4 if trading_mode == 'day' else 1.3)

    # Get current candle data
    row = d1.iloc[-2]
    if any(math.isnan(x) for x in [row['atr'], row['rsi'], row['ema_fast'], row['ema_slow'], row['vol_ma'], row['adx']]):
        dbg(sym, 'NaN in key indicators on primary TF; skipping')
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
    # Day mode needs slightly higher ATR to avoid low-range noise
    if trading_mode == 'day':
        atr_min *= 1.05

    # Price action analysis
    candle_body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    body_ratio = candle_body / (candle_range + 1e-9)
    
    # Confirmation timeframe
    conf = d5.iloc[-2]
    if any(math.isnan(x) for x in [conf['ema_fast'], conf['ema_slow'], conf['rsi'], conf['adx']]):
        dbg(sym, 'NaN in confirmation TF indicators; skipping')
        return None
        
    # Confirmation conditions
    adx_th_conf = ADX_THRESHOLD['moderate'] + (2 if trading_mode == 'day' else 0)
    conf_long = (conf['ema_fast'] > conf['ema_slow'] and 
                conf['rsi'] > 55 and 
                conf['adx'] >= adx_th_conf)
                
    conf_short = (conf['ema_fast'] < conf['ema_slow'] and 
                 conf['rsi'] < 45 and 
                 conf['adx'] >= adx_th_conf)

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
    slope_ok_long = row['ema_fast_slope'] >= 0.01
    slope_ok_short = row['ema_fast_slope'] <= -0.01
    
    # VWAP conditions
    vwap_long = row['close'] >= row['vwap']
    vwap_short = row['close'] <= row['vwap']
    
    # Heikin-Ashi confirmation
    ha_trend_long = row['ha_ema_fast'] > row['ha_ema_slow'] and row['ha_close'] > row['ha_open']
    ha_trend_short = row['ha_ema_fast'] < row['ha_ema_slow'] and row['ha_close'] < row['ha_open']

    # Core signal conditions
    adx_th = ADX_THRESHOLD['moderate'] + (3 if trading_mode == 'day' else 0)
    long_core = (row['ema_fast'] > row['ema_mid'] and 
                row['ema_mid'] > row['ema_slow'] and 
                50 <= row['rsi'] <= 80 and
                row['stoch_k'] > row['stoch_d'] and
                slope_ok_long and 
                vwap_long and 
                row['adx'] >= adx_th and
                body_ratio > 0.4 and
                ha_trend_long)
                
    short_core = (row['ema_fast'] < row['ema_mid'] and 
                 row['ema_mid'] < row['ema_slow'] and 
                 20 <= row['rsi'] <= 50 and
                 row['stoch_k'] < row['stoch_d'] and
                 slope_ok_short and 
                 vwap_short and 
                 row['adx'] >= adx_th and
                 body_ratio > 0.4 and
                 ha_trend_short)

    # Volume condition
    vol_ok = row['volume'] > row['vol_ma']
    
    # ATR condition
    atr_ok = atr_pct >= atr_min

    # Signal conditions with advanced scoring system
    if USE_SCORE_SYSTEM:
        # Calculate structure filters for gating
        fvg_bull, fvg_bear = detect_fvg(d1)
        choch_bull, choch_bear = detect_choch(d1, swing_high, swing_low)
        structure_filter_short = (fvg_bear or choch_bear)
        structure_filter_long = (fvg_bull or choch_bull)
        
        # Calculate base signal scores
        long_score = calculate_signal_score(d1, d5, d15, regime, row, conf, swing_high, swing_low, 
                                           breakout_up, False, retest_up, False) if atr_ok and vol_ok else 0
                                           
        short_score = calculate_signal_score(d1, d5, d15, regime, row, conf, swing_high, swing_low, 
                                            False, breakout_dn, False, retest_dn) if atr_ok and vol_ok else 0
        
        # Advanced confluence analysis for ultra-high accuracy
        confluence_bull = 0
        confluence_bear = 0
        smart_money_bull = 0
        smart_money_bear = 0
        velocity_bull = 0
        velocity_bear = 0
        
        if USE_MTF_CONFLUENCE and d1h is not None:
            confluence_bull, confluence_bear = calculate_mtf_confluence(d1, d5, d15, d1h, d4h)
            
        if USE_SMART_MONEY:
            smart_money_bull, smart_money_bear = calculate_smart_money_score(d1, d5, d15)
            
        if USE_VELOCITY_FILTERS:
            velocity_bull, velocity_bear = calculate_velocity_score(d1, d5)
        
        # Enhanced scoring with confluence multipliers
        if confluence_bull >= MIN_SCORE_CONFLUENCE:
            long_score = long_score * CONFLUENCE_WEIGHT + confluence_bull * 0.3
        if confluence_bear >= MIN_SCORE_CONFLUENCE:
            short_score = short_score * CONFLUENCE_WEIGHT + confluence_bear * 0.3
            
        # Add smart money and velocity bonuses
        long_score += smart_money_bull * 0.2 + velocity_bull * 0.15
        short_score += smart_money_bear * 0.2 + velocity_bear * 0.15
        
        # Add microstructure analysis for institutional activity detection
        micro_bull, micro_bear = analyze_market_microstructure(d1, d5)
        long_score += micro_bull * 0.1  # 10% weight for microstructure
        short_score += micro_bear * 0.1
        
        # Apply ML-enhanced scoring for final signal quality assessment
        if long_score > 0:
            long_score = calculate_ml_signal_score(d1, d5, d15, long_score, 'LONG')
        if short_score > 0:
            short_score = calculate_ml_signal_score(d1, d5, d15, short_score, 'SHORT')
        
        # Cap scores at 100
        long_score = min(100, long_score)
        short_score = min(100, short_score)
        
        # Adaptive score threshold based on market conditions
        # Day mode slightly stricter to maintain accuracy in noisier intraday action
        min_score_required = MIN_SCORE + (2 if trading_mode == 'day' else 0)
        
        # Adjust threshold based on market regime
        # if regime == 'volatile':
        #     min_score_required = MIN_SCORE_VOLATILE
        # elif regime == 'strong_trend':
        #     min_score_required = max(85, MIN_SCORE - 5)  # Slightly lower for strong trends
        # elif regime == 'choppy':
        #     min_score_required = min(100, MIN_SCORE + 10)  # Much higher for choppy markets
            
        # Adjust based on volatility spike strength
        if volatile_ratio > 2.0:  # Very high volatility
            min_score_required = min(100, min_score_required + 5)
        
        # Require volatility spike for entry (unless in strong trend)
        volatility_ok = volatility_spike or regime == 'strong_trend'
        
        # Distance from nearest SR to avoid entries too close to reversal zones
        sr_dist_long_ok = (swing_high is None) or ((swing_high - price) >= 0.5 * atr_v)
        sr_dist_short_ok = (swing_low is None) or ((price - swing_low) >= 0.5 * atr_v)
        
        # Generate signals based on score threshold with structure + SR gating
        # Original breakout/retest path
        long_ok_breakout = (long_score >= min_score_required and strong_upTrend and structure_filter_long and 
                            (retest_up or (breakout_up and vwap_long)) and sr_dist_long_ok and volatility_ok)
        short_ok_breakout = (short_score >= min_score_required and strong_downTrend and structure_filter_short and 
                             (retest_dn or (breakout_dn and vwap_short)) and sr_dist_short_ok and volatility_ok)

        # New confluence pullback path (more signals, still high accuracy)
        pullback_long = (row['ema_fast'] > row['ema_mid'] and row['close'] > row['ema_mid'] and
                         row['rsi'] > 52 and row['macd_hist'] > 0 and ha_trend_long and vwap_long)
        pullback_short = (row['ema_fast'] < row['ema_mid'] and row['close'] < row['ema_mid'] and
                          row['rsi'] < 48 and row['macd_hist'] < 0 and ha_trend_short and vwap_short)

        if trading_mode == 'swing':
            # Swing: rely on HTF confluence instead of strict breakout gating
            long_ok = (long_score >= min_score_required and strong_upTrend and sr_dist_long_ok and volatility_ok and
                       (confluence_bull >= 80) and pullback_long)
            short_ok = (short_score >= min_score_required and strong_downTrend and sr_dist_short_ok and volatility_ok and
                        (confluence_bear >= 80) and pullback_short)
        else:
            # Day: allow either breakout/retest OR quality pullback entries with confluence
            long_ok = long_ok_breakout or (
                long_score >= (min_score_required - 3) and strong_upTrend and pullback_long and sr_dist_long_ok and volatility_ok
            )
            short_ok = short_ok_breakout or (
                short_score >= (min_score_required - 3) and strong_downTrend and pullback_short and sr_dist_short_ok and volatility_ok
            )
        
        # Check for consecutive signals if required
        if long_ok:
            long_ok = not CONSECUTIVE_SIGNALS or track_signal(sym, 'LONG', long_score)
        if short_ok:
            short_ok = not CONSECUTIVE_SIGNALS or track_signal(sym, 'SHORT', short_score)
        
        # Store score for output
        signal_score = long_score if long_ok else short_score if short_ok else 0
        if not long_ok and not short_ok:
            dbg(sym, f"No signal; long_score={round(long_score,1)}, short_score={round(short_score,1)}, min_req={min_score_required}, vol_ok={vol_ok}, atr_ok={atr_ok}, strong_up={strong_upTrend}, strong_down={strong_downTrend}, structure_long={structure_filter_long}, structure_short={structure_filter_short}, volatility_ok={volatility_ok}")
    else:
        # Traditional binary signal logic
        long_ok = (atr_ok and vol_ok and conf_long and upTrend and 
                  ((strong_breakout_up or retest_up) and long_core))
                  
        short_ok = (atr_ok and vol_ok and conf_short and downTrend and 
                   ((strong_breakout_dn or retest_dn) and short_core))
                   
        signal_score = 0  # Not used in binary mode

    # Validate momentum before generating signals
    momentum_ok, momentum_msg = validate_momentum(d1)
    if not momentum_ok:
        dbg(sym, f"Momentum invalid: {momentum_msg}")
        return None  # Skip signal if momentum is insufficient

    # Generate signal with dynamic risk management
    if long_ok:
        # Dynamic risk parameters based on market conditions and signal quality
        dynamic_sl_atr = SL_ATR
        dynamic_tp1_r = TP1_R
        dynamic_tp2_r = TP2_R
        dynamic_tp3_r = TP3_R
        
        # Adjust based on market regime
        if regime == 'volatile':
            dynamic_sl_atr *= 1.3  # Wider stops in volatile markets
            dynamic_tp1_r *= 0.8   # Closer first target
        elif regime == 'strong_trend':
            dynamic_sl_atr *= 0.9  # Tighter stops in strong trends
            dynamic_tp2_r *= 1.2   # Extended targets
            dynamic_tp3_r *= 1.3
        elif regime == 'choppy':
            dynamic_sl_atr *= 1.5  # Much wider stops in choppy markets
            dynamic_tp1_r *= 0.7   # Quick scalp targets
            
        # Adjust based on signal quality (higher scores = better risk/reward)
        if long_score >= 95:
            dynamic_tp2_r *= 1.4   # Premium signals get extended targets
            dynamic_tp3_r *= 1.5
        elif long_score >= 90:
            dynamic_tp2_r *= 1.2
            dynamic_tp3_r *= 1.3
            
        # Adjust based on confluence strength
        if confluence_bull >= 80:
            dynamic_sl_atr *= 0.85  # Tighter stops for high confluence
            dynamic_tp2_r *= 1.3    # Extended targets
            
        # Calculate stop loss (use swing low or dynamic ATR-based)
        sl = min(price - dynamic_sl_atr*atr_v, swing_low - 0.1*atr_v if swing_low is not None else price - dynamic_sl_atr*atr_v)
        r = price - sl
        tp1 = price + dynamic_tp1_r * r
        tp2 = price + dynamic_tp2_r * r
        tp3 = price + dynamic_tp3_r * r
        
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
                'mode': trading_mode,
                'rr': f"{TP1_R}/{TP2_R}/{TP3_R}"}
                
    if short_ok:
        # Dynamic risk parameters based on market conditions and signal quality
        dynamic_sl_atr = SL_ATR
        dynamic_tp1_r = TP1_R
        dynamic_tp2_r = TP2_R
        dynamic_tp3_r = TP3_R
        
        # Adjust based on market regime
        if regime == 'volatile':
            dynamic_sl_atr *= 1.3  # Wider stops in volatile markets
            dynamic_tp1_r *= 0.8   # Closer first target
        elif regime == 'strong_trend':
            dynamic_sl_atr *= 0.9  # Tighter stops in strong trends
            dynamic_tp2_r *= 1.2   # Extended targets
            dynamic_tp3_r *= 1.3
        elif regime == 'choppy':
            dynamic_sl_atr *= 1.5  # Much wider stops in choppy markets
            dynamic_tp1_r *= 0.7   # Quick scalp targets
            
        # Adjust based on signal quality (higher scores = better risk/reward)
        if short_score >= 95:
            dynamic_tp2_r *= 1.4   # Premium signals get extended targets
            dynamic_tp3_r *= 1.5
        elif short_score >= 90:
            dynamic_tp2_r *= 1.2
            dynamic_tp3_r *= 1.3
            
        # Adjust based on confluence strength
        if confluence_bear >= 80:
            dynamic_sl_atr *= 0.85  # Tighter stops for high confluence
            dynamic_tp2_r *= 1.3    # Extended targets
            
        # Calculate stop loss (use swing high or dynamic ATR-based)
        sl = max(price + dynamic_sl_atr*atr_v, swing_high + 0.1*atr_v if swing_high is not None else price + dynamic_sl_atr*atr_v)
        r = sl - price
        tp1 = price - dynamic_tp1_r * r
        tp2 = price - dynamic_tp2_r * r
        tp3 = price - dynamic_tp3_r * r
        
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
                'mode': trading_mode,
                'rr': f"{TP1_R}/{TP2_R}/{TP3_R}"}
                
    return None

def scan_once():
    """Scan all symbols for trading signals across both day and swing modes."""
    emits = []
    for mode in ['day', 'swing']:
        for s in SYMBOLS:
            try:
                if not cooldown_ok(s, mode):
                    dbg(s, f'{mode} cooldown active; skipping')
                    continue
                sig = make_signal(s, mode)
                if sig:
                    emits.append(sig)
            except Exception as e:
                print(f'ERR {s} [{mode}]: {e}')
    return emits

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
                                          swing_high, swing_low, breakout_up, False, False, False)
                                          
        short_score = calculate_signal_score(d1_subset, d5_subset, d15_subset, regime, row, conf, 
                                           swing_high, swing_low, False, breakout_dn, False, False)
        
        # Check for signals with momentum validation
        signal = None
        if long_score >= MIN_SCORE and upTrend:
            # Validate momentum before taking signal
            momentum_ok, momentum_msg = validate_momentum(d1_subset)
            if momentum_ok:
                signal = 'LONG'
        elif short_score >= MIN_SCORE and downTrend:
            # Validate momentum before taking signal
            momentum_ok, momentum_msg = validate_momentum(d1_subset)
            if momentum_ok:
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
    
    import argparse
    parser = argparse.ArgumentParser(description='Crypto Scanner: scalping and swing modes')
    parser.add_argument('--mode', choices=['day', 'swing'], default=TRADING_MODE, help='Trading mode')
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode')
    parser.add_argument('--symbol', help='Symbol for backtest, e.g., BTCUSDT')
    parser.add_argument('--start', help='Backtest start date YYYY-MM-DD')
    parser.add_argument('--end', help='Backtest end date YYYY-MM-DD')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Apply mode and debug
    TRADING_MODE = args.mode
    DEBUG = args.debug or DEBUG
    # Dual-mode: settings applied per-signal in make_signal

    print(f"Monitoring {len(SYMBOLS)} symbols with {MIN_SCORE}+ signal score threshold.")
    print(f"[Mode] TRADING_MODE={TRADING_MODE} PRIMARY_IV={PRIMARY_IV} CONFIRM_IV={CONFIRM_IV} TREND_IV={TREND_IV}")
    print(f"[Risk] SL_ATR={SL_ATR} TP1_R={TP1_R} TP2_R={TP2_R} TP3_R={TP3_R} COOLDOWN_MIN={COOLDOWN_MIN}")

    if args.backtest:
        if not args.symbol:
            print("Usage: python scalping_signal_bot.py --backtest --symbol BTCUSDT [--start YYYY-MM-DD] [--end YYYY-MM-DD]")
            exit(1)
        backtest(args.symbol, args.start, args.end)
    else:
        print("Running in live scanning mode. Press Ctrl+C to exit.")
        try:
            while True:
                emits = scan_once()
                if emits:
                    for s in emits:
                        mark_emit(s['symbol'], s.get('mode', 'day'))
                        print(f"[{s.get('mode','day').upper()}] {s['symbol']} {s['interval']} {s['side']} @ {s['price']} | SL {s['sl']} TP1 {s['tp1']} TP2 {s['tp2']} TP3 {s['tp3']} "
                               f"| ATR% {s['atr_pct']} RSI {s['rsi']} ADX {s['adx']} Slope% {s['slope']} VWAP {s['vwap_side']} "
                               f"| Volx {s['vol_mult']} | S:{s['sr_low']} R:{s['sr_high']} | Score: {s['score']} | Regime: {s['regime']} | R:R {s['rr']}")
                        try:
                            log_signal(s)
                            print('\a', end='')
                        except Exception:
                            pass
                time.sleep(SLEEP_SEC)
        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
            exit(0)

# New structure helpers for day trading

def detect_fvg(df, lookback=20):
    # Simple fair value gap (FVG) detection: three-candle pattern
    # Bullish FVG: low of candle n+1 > high of candle n-1
    # Bearish FVG: high of candle n+1 < low of candle n-1
    if len(df) < 3:
        return False, False
    i = len(df) - 2
    bull = df['low'].iloc[i+1] > df['high'].iloc[i-1]
    bear = df['high'].iloc[i+1] < df['low'].iloc[i-1]
    return bull, bear


def detect_choch(d1, swing_high, swing_low):
    # Change of Character (ChoCh): price breaks opposite swing then closes beyond
    if swing_high is None or swing_low is None:
        return False, False
    row = d1.iloc[-2]
    bull_choch = (row['close'] > swing_high) and (row['ema_fast'] > row['ema_mid'])
    bear_choch = (row['close'] < swing_low) and (row['ema_fast'] < row['ema_mid'])
    return bull_choch, bear_choch
