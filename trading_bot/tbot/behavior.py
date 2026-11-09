"""Анализ поведения трейдеров: кто давит на цену, что делают крупные игроки, есть ли паника или жадность."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_flow(trades: pd.DataFrame) -> dict:
    """Метрики по ленте сделок (сторона B/S известна из MOEX ISS)."""
    if trades is None or trades.empty or trades["side"].isna().all():
        return {"available": False}
    buy = trades.loc[trades["side"] == "B", "value"].sum()
    sell = trades.loc[trades["side"] == "S", "value"].sum()
    total = buy + sell
    # «крупные» сделки - верхние 5% по объёму: так обычно торгуют фонды и крупные спекулянты
    threshold = trades["value"].quantile(0.95)
    big = trades[trades["value"] >= threshold]
    big_buy = big.loc[big["side"] == "B", "value"].sum()
    big_sell = big.loc[big["side"] == "S", "value"].sum()
    big_total = big_buy + big_sell
    return {
        "available": True,
        "trades": int(len(trades)),
        "buy_share": round(float(buy / total), 3) if total else 0.5,
        "imbalance": round(float((buy - sell) / total), 3) if total else 0.0,
        "big_trades_imbalance": round(float((big_buy - big_sell) / big_total), 3) if big_total else 0.0,
        "big_trades_share_of_value": round(float(big_total / total), 3) if total else 0.0,
    }


def candle_behavior(df: pd.DataFrame, n: int = 20) -> dict:
    """Метрики поведения толпы по свечам: объём на росте vs на падении, всплески объёма, «страх/жадность»."""
    tail = df.tail(n)
    up = tail[tail["close"] > tail["open"]]["volume"].sum()
    down = tail[tail["close"] < tail["open"]]["volume"].sum()
    total = up + down
    rets = df["close"].pct_change().dropna()
    recent_vol = rets.tail(n).std()
    long_vol = rets.tail(n * 5).std()
    vol_z = df["volume_z"].iloc[-1] if "volume_z" in df else np.nan
    rsi = df["rsi14"].iloc[-1] if "rsi14" in df else np.nan
    # простой индекс «страха и жадности» 0..100 из RSI, импульса и волатильности
    momentum = df["close"].iloc[-1] / df["close"].iloc[-n] - 1 if len(df) > n else 0.0
    fg = 50.0
    if not np.isnan(rsi):
        fg = 0.5 * rsi + 0.5 * (50 + np.clip(momentum * 1000, -50, 50))
    if long_vol and recent_vol > 1.5 * long_vol:
        fg -= 10  # резкий рост волатильности - обычно признак страха
    return {
        "up_volume_share": round(float(up / total), 3) if total else 0.5,
        "volume_spike_z": None if pd.isna(vol_z) else round(float(vol_z), 2),
        "volatility_ratio": round(float(recent_vol / long_vol), 2) if long_vol else None,
        "fear_greed": round(float(np.clip(fg, 0, 100)), 1),
    }
