"""Технические индикаторы на pandas. На вход - DataFrame со столбцами open, high, low, close, volume."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # если убытков не было вовсе: 100 при росте, 50 при полном штиле
    no_loss = loss == 0
    out[no_loss & (gain > 0)] = 100.0
    out[no_loss & (gain == 0)] = 50.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n).std()
    return mid - k * std, mid, mid + k * std


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema20"] = ema(c, 20)
    df["ema50"] = ema(c, 50)
    df["sma200"] = sma(c, 200)
    df["rsi14"] = rsi(c, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["bb_low"], df["bb_mid"], df["bb_high"] = bollinger(c)
    df["atr14"] = atr(df, 14)
    df["ret_5"] = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    return df


def snapshot(df: pd.DataFrame) -> dict:
    """Последние значения индикаторов в виде словаря - удобно отдать агенту."""
    last = df.iloc[-1]
    keys = ["close", "ema20", "ema50", "sma200", "rsi14", "macd", "macd_signal", "macd_hist",
            "bb_low", "bb_mid", "bb_high", "atr14", "ret_5", "ret_20", "volume_z"]
    out = {}
    for k in keys:
        v = last.get(k)
        out[k] = None if v is None or pd.isna(v) else round(float(v), 4)
    return out
