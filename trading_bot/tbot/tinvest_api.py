"""Низкоуровневый клиент T-Invest API (REST-шлюз). Общий для данных и для заявок.

Документация: https://developer.tbank.ru/invest/api
"""
from __future__ import annotations

import json
import time

import requests

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


class TInvestClient:
    def __init__(self, token: str | None, class_code: str = "TQBR", base_url: str = BASE_URL):
        if not token:
            raise TInvestError("Не задан TINVEST_TOKEN в .env")
        self.class_code = class_code
        self.base_url = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        self._instruments: dict[str, dict] = {}

    def call(self, service: str, method: str, body: dict) -> dict:
        url = f"{self.base_url}/{PKG}.{service}/{method}"
        for attempt in range(3):
            r = self.s.post(url, data=json.dumps(body), timeout=20)
            if r.status_code == 429 and attempt < 2:   # превышен лимит запросов - подождём
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                raise TInvestError(f"{service}/{method}: HTTP {r.status_code} {r.text[:300]}")
            return r.json()
        raise TInvestError(f"{service}/{method}: превышен лимит запросов")

    def instrument(self, ticker: str) -> dict:
        if ticker not in self._instruments:
            data = self.call("InstrumentsService", "ShareBy", {
                "idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": self.class_code, "id": ticker})
            ins = data["instrument"]
            self._instruments[ticker] = {
                "uid": ins["uid"], "figi": ins.get("figi"), "lot": int(ins.get("lot", 1)),
                "name": ins.get("name", ticker),
                "tradable": ins.get("apiTradeAvailableFlag", True) and ins.get("buyAvailableFlag", True),
            }
        return self._instruments[ticker]

    def ticker_by_uid(self, uid: str) -> str | None:
        for t, v in self._instruments.items():
            if v["uid"] == uid:
                return t
        return None
