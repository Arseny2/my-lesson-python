"""Брокер T-Invest API (Т-Банк, бывш. Тинькофф Инвестиции) через REST-шлюз.

Документация: https://developer.tbank.ru/invest/api
Токен создаётся в приложении Т-Инвестиций: Настройки -> Токены T-Invest API.
  * Для песочницы (sandbox) - любой токен, деньги виртуальные.
  * Для реальной торговли - токен с правом торговли, лучше ограниченный одним счётом.

Стоп-лосс и тейк-профит бот хранит у себя и проверяет на каждом цикле (закрывает позицию
рыночной заявкой). Пока бот выключен, стопы НЕ работают - для страховки ставьте
стоп-заявки в приложении брокера.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import requests

from ..models import Action, Fill, Position
from .base import Broker

log = logging.getLogger(__name__)

BASE_URL = "https://invest-public-api.tbank.ru/rest"  # старый адрес: invest-public-api.tinkoff.ru
PKG = "tinkoff.public.invest.api.contract.v1"


def q2f(q: dict | None) -> float:
    """Quotation / MoneyValue {'units': '123', 'nano': 450000000} -> 123.45"""
    if not q:
        return 0.0
    return int(q.get("units", 0) or 0) + int(q.get("nano", 0) or 0) / 1e9


def f2q(x: float) -> dict:
    units = int(x)
    return {"units": str(units), "nano": int(round((x - units) * 1e9))}


class TInvestError(RuntimeError):
    pass


class TInvestBroker(Broker):
    def __init__(self, token: str, sandbox: bool = True, account_id: str | None = None,
                 class_code: str = "TQBR", state_file: str | Path | None = None,
                 base_url: str = BASE_URL, sandbox_start_cash: float = 100_000.0):
        if not token:
            raise TInvestError("Не задан TINVEST_TOKEN в .env")
        self.sandbox = sandbox
        self.is_real_money = not sandbox
        self.name = "tinvest_sandbox" if sandbox else "tinvest_live"
        self.class_code = class_code
        self.base_url = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        self._instruments: dict[str, dict] = {}
        self.state_file = Path(state_file) if state_file else None
        self.meta = self._load_meta()  # стопы, тейки, решения - то, чего нет у брокера
        self.closed_trades: list[dict] = self.meta.setdefault("closed_trades", [])
        self.account_id = account_id or self._default_account(sandbox_start_cash)

    # --- HTTP ---
    def _call(self, service: str, method: str, body: dict) -> dict:
        url = f"{self.base_url}/{PKG}.{service}/{method}"
        r = self.s.post(url, data=json.dumps(body), timeout=20)
        if r.status_code != 200:
            raise TInvestError(f"{service}/{method}: HTTP {r.status_code} {r.text[:300]}")
        return r.json()

    def _svc(self, live_service: str) -> str:
        return "SandboxService" if self.sandbox else live_service

    # --- счёт ---
    def _default_account(self, start_cash: float) -> str:
        if self.sandbox:
            accs = self._call("SandboxService", "GetSandboxAccounts", {}).get("accounts", [])
            if accs:
                return accs[0]["id"]
            acc_id = self._call("SandboxService", "OpenSandboxAccount", {})["accountId"]
            self._call("SandboxService", "SandboxPayIn",
                       {"accountId": acc_id, "amount": {"currency": "rub", **f2q(start_cash)}})
            log.info("Открыт счёт в песочнице %s, зачислено %.0f ₽", acc_id, start_cash)
            return acc_id
        accs = self._call("UsersService", "GetAccounts", {}).get("accounts", [])
        open_accs = [a for a in accs if a.get("status") == "ACCOUNT_STATUS_OPEN"]
        if len(open_accs) != 1:
            raise TInvestError("Найдено несколько счетов (или ни одного). Укажите TINVEST_ACCOUNT_ID в .env: "
                               + ", ".join(f"{a['id']} ({a.get('name')})" for a in accs))
        return open_accs[0]["id"]

    # --- инструменты ---
    def instrument(self, ticker: str) -> dict:
        if ticker not in self._instruments:
            data = self._call("InstrumentsService", "ShareBy", {
                "idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": self.class_code, "id": ticker})
            ins = data["instrument"]
            self._instruments[ticker] = {"uid": ins["uid"], "figi": ins.get("figi"), "lot": int(ins.get("lot", 1))}
        return self._instruments[ticker]

    # --- локальные метаданные позиций ---
    def _load_meta(self) -> dict:
        if self.state_file and self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        return {"positions": {}}

    def _save_meta(self) -> None:
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- портфель ---
    def _raw_portfolio(self) -> dict:
        method = "GetSandboxPortfolio" if self.sandbox else "GetPortfolio"
        return self._call(self._svc("OperationsService"), method, {"accountId": self.account_id, "currency": "RUB"})

    def _positions(self) -> dict[str, Position]:
        raw = self._raw_portfolio()
        by_uid = {v["uid"]: t for t, v in self._instruments.items()}
        out = {}
        for p in raw.get("positions", []):
            if p.get("instrumentType") != "share":
                continue
            ticker = by_uid.get(p.get("instrumentUid")) or p.get("ticker")
            if not ticker:
                continue
            lot = self.instrument(ticker)["lot"]
            shares = q2f(p.get("quantity"))
            m = self.meta["positions"].get(ticker, {})
            out[ticker] = Position(ticker, int(shares // lot), lot, q2f(p.get("averagePositionPrice")),
                                   m.get("stop"), m.get("take"), m.get("opened_at", ""), m.get("decision_id", ""))
        return out

    def position(self, ticker: str) -> Position | None:
        self.instrument(ticker)
        p = self._positions().get(ticker)
        return p if p and p.lots > 0 else None

    def portfolio(self, prices: dict[str, float]) -> dict:
        for t in prices:
            self.instrument(t)
        raw = self._raw_portfolio()
        equity = q2f(raw.get("totalAmountPortfolio"))
        cash = sum(q2f(p.get("quantity")) for p in raw.get("positions", []) if p.get("instrumentType") == "currency")
        positions, exposure = {}, 0.0
        for t, p in self._positions().items():
            px = prices.get(t, p.avg_price)
            value = p.shares * px
            exposure += value
            positions[t] = {"lots": p.lots, "shares": p.shares, "avg_price": p.avg_price, "price": px,
                            "value": round(value, 2), "pnl_pct": round(px / p.avg_price - 1, 4) if p.avg_price else 0,
                            "stop": p.stop_price, "take": p.take_price}
        return {"cash": round(cash, 2), "equity": round(equity, 2), "exposure": round(exposure, 2),
                "positions": positions}

    # --- заявки ---
    def _order(self, ticker: str, lots: int, direction: str) -> dict:
        ins = self.instrument(ticker)
        method = "PostSandboxOrder" if self.sandbox else "PostOrder"
        return self._call(self._svc("OrdersService"), method, {
            "instrumentId": ins["uid"], "quantity": str(lots), "direction": direction,
            "accountId": self.account_id, "orderType": "ORDER_TYPE_MARKET",
            "orderId": str(uuid.uuid4()),  # ключ идемпотентности: повтор не создаст вторую заявку
        })

    @staticmethod
    def _exec_price(r: dict, fallback: float) -> float:
        """executedOrderPrice - средняя цена одной бумаги. Если ответ странный - берём рыночную цену."""
        px = q2f(r.get("executedOrderPrice"))
        if px <= 0 or abs(px / fallback - 1) > 0.2:
            return fallback
        return px

    def buy(self, ticker, lots, lot_size, price, stop, take, time, reason="", decision_id=""):
        if lots <= 0:
            return None
        r = self._order(ticker, lots, "ORDER_DIRECTION_BUY")
        done = int(r.get("lotsExecuted", 0) or 0)
        if done <= 0:
            log.warning("Заявка на покупку %s не исполнена: %s", ticker, r.get("executionReportStatus"))
            return None
        px = self._exec_price(r, price)
        self.meta["positions"][ticker] = {"stop": stop, "take": take, "opened_at": time,
                                          "decision_id": decision_id, "entry": px}
        self._save_meta()
        return Fill(ticker, Action.BUY, done, px, q2f(r.get("initialCommission")), time, reason)

    def sell(self, ticker, lots, price, time, reason=""):
        pos = self.position(ticker)
        if not pos:
            return None
        lots = min(lots, pos.lots)
        r = self._order(ticker, lots, "ORDER_DIRECTION_SELL")
        done = int(r.get("lotsExecuted", 0) or 0)
        if done <= 0:
            log.warning("Заявка на продажу %s не исполнена: %s", ticker, r.get("executionReportStatus"))
            return None
        px = self._exec_price(r, price)
        m = self.meta["positions"].get(ticker, {})
        self.closed_trades.append({
            "ticker": ticker, "opened_at": m.get("opened_at"), "closed_at": time, "entry": pos.avg_price,
            "exit": px, "lots": done, "pnl": round((px - pos.avg_price) * done * pos.lot_size, 2),
            "pnl_pct": round(px / pos.avg_price - 1, 4) if pos.avg_price else 0,
            "reason": reason, "decision_id": m.get("decision_id", ""),
        })
        if done >= pos.lots:
            self.meta["positions"].pop(ticker, None)
        self._save_meta()
        return Fill(ticker, Action.SELL, done, px, q2f(r.get("initialCommission")), time, reason)
