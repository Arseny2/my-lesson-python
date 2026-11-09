"""Бумажный брокер: виртуальные деньги, реальные цены. Состояние сохраняется в JSON."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..models import Action, Fill, Position
from .base import Broker


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, cash: float = 100_000.0, commission_pct: float = 0.0005,
                 slippage_pct: float = 0.0005, state_file: str | Path | None = None):
        self.cash = cash
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.closed_trades: list[dict] = []   # для обучения: вход, выход, результат
        self.state_file = Path(state_file) if state_file else None
        if self.state_file and self.state_file.exists():
            self._load()

    # --- сохранение ---
    def _load(self) -> None:
        s = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.cash = s["cash"]
        self.positions = {k: Position(**v) for k, v in s["positions"].items()}
        self.fills = [Fill(**{**f, "side": Action(f["side"])}) for f in s.get("fills", [])]
        self.closed_trades = s.get("closed_trades", [])

    def save(self) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash": self.cash,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "fills": [{**asdict(f), "side": f.side.value} for f in self.fills[-500:]],
            "closed_trades": self.closed_trades[-500:],
        }
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_file)

    # --- интерфейс ---
    def position(self, ticker: str) -> Position | None:
        return self.positions.get(ticker)

    def portfolio(self, prices: dict[str, float]) -> dict:
        positions, exposure = {}, 0.0
        for t, p in self.positions.items():
            px = prices.get(t, p.avg_price)
            value = p.shares * px
            exposure += value
            positions[t] = {"lots": p.lots, "shares": p.shares, "avg_price": round(p.avg_price, 4),
                            "price": px, "value": round(value, 2),
                            "pnl_pct": round(px / p.avg_price - 1, 4),
                            "stop": p.stop_price, "take": p.take_price}
        return {"cash": round(self.cash, 2), "equity": round(self.cash + exposure, 2),
                "exposure": round(exposure, 2), "positions": positions}

    def buy(self, ticker, lots, lot_size, price, stop, take, time, reason="", decision_id=""):
        px = price * (1 + self.slippage_pct)
        per_lot = px * lot_size * (1 + self.commission_pct)
        lots = min(lots, int(self.cash // per_lot))
        if lots <= 0:
            return None
        gross = px * lot_size * lots
        commission = gross * self.commission_pct
        self.cash -= gross + commission
        old = self.positions.get(ticker)
        if old:
            total = old.lots + lots
            old.avg_price = (old.avg_price * old.lots + px * lots) / total
            old.lots = total
            old.stop_price, old.take_price = stop or old.stop_price, take or old.take_price
        else:
            self.positions[ticker] = Position(ticker, lots, lot_size, px, stop, take, time, decision_id)
        fill = Fill(ticker, Action.BUY, lots, px, round(commission, 2), time, reason)
        self.fills.append(fill)
        self.save()
        return fill

    def sell(self, ticker, lots, price, time, reason=""):
        pos = self.positions.get(ticker)
        if not pos:
            return None
        lots = min(lots, pos.lots)
        px = price * (1 - self.slippage_pct)
        gross = px * pos.lot_size * lots
        commission = gross * self.commission_pct
        self.cash += gross - commission
        cost_basis = pos.avg_price * pos.lot_size * lots
        self.closed_trades.append({
            "ticker": ticker, "opened_at": pos.opened_at, "closed_at": time,
            "entry": round(pos.avg_price, 4), "exit": round(px, 4), "lots": lots,
            "pnl": round(gross - commission - cost_basis * (1 + self.commission_pct), 2),
            "pnl_pct": round(px / pos.avg_price - 1 - 2 * self.commission_pct, 4),
            "reason": reason, "decision_id": pos.decision_id,
        })
        pos.lots -= lots
        if pos.lots <= 0:
            del self.positions[ticker]
        fill = Fill(ticker, Action.SELL, lots, px, round(commission, 2), time, reason)
        self.fills.append(fill)
        self.save()
        return fill
