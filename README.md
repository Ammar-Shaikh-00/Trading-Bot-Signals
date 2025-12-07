# Trading-Bot-Signals
- Advanced crypto signal suite for high-accuracy scalping and disciplined day/swing trading on Binance USDT perpetuals.
- Two scripts:
  - Scalping-Signals.py : 1m primary with 5m confirm and 15m trend, optimized for scalps.
  - Day+Swing-Signals.py : 15m/1h/4h framework for cleaner, higher-RR swings.
Highlights (Scalping)

- 40 liquid symbols ( SYMBOLS ) with EXTENDED_SYMBOLS and staggered cadence via EXTENDED_SCAN_EVERY .
- Multi-timeframe alignment: PRIMARY_IV='1m' , CONFIRM_IV='5m' , TREND_IV='15m' .
- Score-based signal model, volume and microstructure guards, cooldown, and consecutive-signal checks.
- Indicators: EMA stack and slopes, RSI/Stoch/CCI, MACD, ADX, MFI, OBV, VWAP, Bollinger/Keltner, squeeze, ATR bands.
- Risk: ATR-derived stops capped by MAX_SL_PCT = 1.5% for scalping-friendly risk; take-profits via TP_RR and TP1_R/TP2_R/TP3_R .
Highlights (Day/Swing)

- Higher-timeframe confluence: PRIMARY_IV='15m' , CONFIRM_IV='1h' , TREND_IV='4h' .
- Multi-timeframe confluence scoring, Smart Money features (FVG, ChoCH), velocity and momentum filters.
- Stricter gates: MIN_SCORE=90+ , wider SL_ATR and partial exits enabled ( PARTIAL_TP , PARTIAL_EXIT_1/2 ).
- Logs signals to CSV ( signals.log ) with detailed context.
Quick Start

- Requirements: pip install requests pandas numpy (optional: ta-lib if you want to extend indicators).
- Run scalping: python Scalping-Signals.py
- Run day/swing: python "Day+Swing-Signals.py"
- Tweak config: edit SYMBOLS , EXTENDED_SYMBOLS , EXTENDED_SCAN_EVERY , MIN_SCORE , MAX_SL_PCT , and intervals to fit your preferences.
Notes

- Built for futures symbols on Binance ( BASE='https://fapi.binance.com' ).
- Not financial advice. Test thoroughly and manage risk.
