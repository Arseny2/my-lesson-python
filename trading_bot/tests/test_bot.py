import json
import sys
from pathlib import Path

import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tbot.agents import MarketContext, RiskManager  # noqa: E402
from tbot.backtest import run_backtest  # noqa: E402
from tbot.brokers import PaperBroker  # noqa: E402
from tbot.brokers.tinvest import TInvestBroker, f2q, q2f  # noqa: E402
from tbot.config import Config, RiskConfig, load_config  # noqa: E402
from tbot.data.synthetic import make_candles, make_trades  # noqa: E402
from tbot.engine import TradingEngine  # noqa: E402
from tbot.indicators import add_all, snapshot  # noqa: E402
from tbot.llm import LLM  # noqa: E402
from tbot.memory import Memory  # noqa: E402
from tbot.models import Action, AgentReport, DebateResult, RiskVerdict, TradeDecision  # noqa: E402


def ctx(price=100.0, lot=10, equity=100_000, exposure=0, position=None, atr=2.0):
    df = add_all(make_candles(300))
    ind = snapshot(df)
    ind["atr14"] = atr
    return MarketContext(ticker="T", name="T", price=price, lot_size=lot, candles=df, indicators=ind,
                         behavior={}, flow={}, position=position,
                         portfolio={"cash": equity - exposure, "equity": equity, "exposure": exposure})


DEB = DebateResult(bull_case="", bear_case="", winner="bull", signal=0.6, confidence=0.7)


def buy(size=0.5, stop=0.001, take=0.001):
    return TradeDecision(action=Action.BUY, size_pct=size, stop_loss_pct=stop, take_profit_pct=take, reasoning="")


# ---------- индикаторы ----------
def test_indicators_bounds():
    df = add_all(make_candles(500))
    r = df["rsi14"].dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert (df["atr14"].dropna() > 0).all()


# ---------- риск-менеджер ----------
def test_risk_caps_size_and_widens_stop():
    rm = RiskManager(RiskConfig(max_position_pct=0.2))
    v = rm.review(ctx(), buy(0.5), DEB, day_pnl_pct=0, trades_today=0)
    assert v.approved
    assert v.size_pct == pytest.approx(0.2)
    assert v.stop_loss_pct >= 2 * 2.0 / 100        # не ближе 2 ATR
    assert v.take_profit_pct >= v.stop_loss_pct * 1.5


def test_risk_llm_cannot_loosen_limits():
    rm = RiskManager(RiskConfig(max_position_pct=0.1, max_stop_loss_pct=0.05))
    loose = RiskVerdict(approved=True, size_pct=0.9, stop_loss_pct=0.5, take_profit_pct=0.1, reason="ИИ")
    v = rm.enforce(ctx(), buy(0.9), DEB, loose, 0, 0)
    assert v.size_pct <= 0.1 and v.stop_loss_pct <= 0.05


def test_risk_blocks_after_daily_loss_and_low_confidence():
    rm = RiskManager(RiskConfig())
    assert not rm.review(ctx(), buy(), DEB, day_pnl_pct=-0.05, trades_today=0).approved
    weak = DEB.model_copy(update={"confidence": 0.1})
    assert not rm.review(ctx(), buy(), weak, 0, 0).approved


def test_risk_always_allows_exit_and_never_shorts():
    rm = RiskManager(RiskConfig())
    sell = TradeDecision(action=Action.SELL, size_pct=1, stop_loss_pct=0.02, take_profit_pct=0.04, reasoning="")
    assert rm.review(ctx(position={"value": 5000}), sell, DEB, day_pnl_pct=-0.5, trades_today=999).approved
    assert not rm.review(ctx(position=None), sell, DEB, 0, 0).approved


def test_risk_denies_when_one_lot_too_expensive():
    rm = RiskManager(RiskConfig(max_position_pct=0.01))
    assert not rm.review(ctx(price=500, lot=10, equity=100_000), buy(), DEB, 0, 0).approved


# ---------- бумажный брокер ----------
def test_paper_broker_roundtrip_and_stop(tmp_path):
    b = PaperBroker(100_000, commission_pct=0.001, slippage_pct=0, state_file=tmp_path / "s.json")
    f = b.buy("T", 10, 10, 100.0, stop=95.0, take=110.0, time="t1", decision_id="d1")
    assert f.lots == 10 and b.cash == pytest.approx(100_000 - 10_000 - 10)
    # состояние переживает перезапуск
    b2 = PaperBroker(commission_pct=0.001, slippage_pct=0, state_file=tmp_path / "s.json")
    assert b2.position("T").lots == 10
    exit_fill = b2.check_exits("T", {"open": 97, "high": 98, "low": 94, "close": 96}, "t2")
    assert exit_fill and exit_fill.price == 95.0 and exit_fill.reason == "стоп-лосс"
    assert b2.position("T") is None
    assert b2.closed_trades[0]["pnl"] < 0


def test_paper_broker_gap_down_fills_at_open():
    b = PaperBroker(100_000, 0, 0)
    b.buy("T", 1, 1, 100.0, stop=95.0, take=None, time="t")
    assert b.check_exits("T", {"open": 90, "high": 91, "low": 89, "close": 90}, "t").price == 90


def test_paper_broker_cannot_overspend():
    b = PaperBroker(1_000, 0, 0)
    f = b.buy("T", 100, 10, 30.0, None, None, "t")
    assert f.lots == 3 and b.cash >= 0


# ---------- обучение ----------
def test_memory_learns_accuracy():
    m = Memory(eval_after_cycles=1, min_samples=3)
    for i in range(5):
        m.record_signals("T", 100.0, {"technical": 0.8, "sentiment": -0.8}, f"t{i}")
        m.evaluate("T", 110.0)   # цена выросла: технический прав, сентимент нет
    w = m.weights()
    assert w["technical"] > 1.0 > w["sentiment"]


# ---------- бэктест / движок ----------
def test_backtest_runs_and_executes_on_next_open():
    cfg = Config(use_llm=False)
    candles = make_candles(400, seed=3)
    rep = run_backtest(cfg, "T", candles, lot_size=10)
    assert rep.bars > 0 and rep.trades >= 1
    assert rep.end_equity > 0


def test_engine_cycle_with_mocked_llm(monkeypatch):
    calls = []

    def fake_ask(self, role, data, schema):
        calls.append(schema.__name__)
        if schema is AgentReport:
            return AgentReport(signal=0.9, confidence=0.9, summary="рост", key_points=["x"])
        if schema is DebateResult:
            return DebateResult(bull_case="a", bear_case="b", winner="bull", signal=0.8, confidence=0.9)
        if schema is TradeDecision:
            return TradeDecision(action=Action.BUY, size_pct=0.5, stop_loss_pct=0.03, take_profit_pct=0.06,
                                 reasoning="покупаем")
        if schema is RiskVerdict:
            return RiskVerdict(approved=True, size_pct=0.5, stop_loss_pct=0.03, take_profit_pct=0.06, reason="ок")
        return None

    monkeypatch.setattr(LLM, "ask", fake_ask)
    llm = LLM.__new__(LLM)
    broker = PaperBroker(100_000, 0, 0)
    engine = TradingEngine(Config(), broker, Memory(), llm)
    candles = make_candles(300)
    r = engine.run_cycle("T", candles, lot_size=1, trades=make_trades(), news=[])
    # новостей нет -> новостной агент не тратит запрос на Claude
    assert calls.count("AgentReport") == 2
    assert {"DebateResult", "TradeDecision", "RiskVerdict"} <= set(calls)
    assert r.fill and r.fill.side == Action.BUY
    assert r.verdict.size_pct <= Config().risk.max_position_pct   # ИИ просил 50%, лимит 20%


# ---------- запрос к Claude ----------
def _llm_with_transport(handler):
    from anthropic import DefaultHttpxClient
    import anthropic
    llm = LLM.__new__(LLM)
    llm.client = anthropic.Anthropic(api_key="test", max_retries=0,
                                     http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    llm.model, llm.effort = "claude-opus-5-5", "medium"
    llm.input_tokens = llm.output_tokens = llm.calls = llm.errors = 0
    return llm


def test_llm_request_shape_and_parsing():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        out = {"signal": 0.4, "confidence": 0.6, "summary": "ок", "key_points": []}
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(out)}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20}})

    llm = _llm_with_transport(handler)
    rep = llm.ask("роль", "данные", AgentReport)
    assert rep.signal == 0.4
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in seen["beta"]
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body and "temperature" not in body
    assert llm.calls == 1 and llm.cost_usd > 0


def test_llm_errors_fall_back_to_none():
    llm = _llm_with_transport(lambda r: httpx2.Response(500, json={"type": "error", "error": {
        "type": "api_error", "message": "boom"}}))
    assert llm.ask("роль", "данные", AgentReport) is None
    assert llm.errors == 1


# ---------- T-Invest ----------
def test_quotation_conversion():
    assert q2f({"units": "123", "nano": 450000000}) == pytest.approx(123.45)
    assert q2f(f2q(98.7654)) == pytest.approx(98.7654)


def test_tinvest_sandbox_order_flow(monkeypatch):
    sent = []

    class Resp:
        def __init__(self, data):
            self.status_code, self._d, self.text = 200, data, ""

        def json(self):
            return self._d

    def fake_post(self, url, data=None, timeout=None):
        method = url.rsplit("/", 1)[-1]
        sent.append((method, json.loads(data)))
        replies = {
            "GetSandboxAccounts": {"accounts": [{"id": "acc1"}]},
            "ShareBy": {"instrument": {"uid": "uid-sber", "figi": "BBG004730N88", "lot": 10}},
            "PostSandboxOrder": {"lotsExecuted": "2", "executedOrderPrice": {"units": "250", "nano": 0},
                                 "initialCommission": {"units": "0", "nano": 500000000}},
        }
        return Resp(replies[method])

    import requests
    monkeypatch.setattr(requests.Session, "post", fake_post)
    b = TInvestBroker("token", sandbox=True)
    assert b.account_id == "acc1"
    f = b.buy("SBER", 2, 10, 251.0, stop=240.0, take=270.0, time="t")
    assert f.lots == 2 and f.price == 250.0
    order = [body for m, body in sent if m == "PostSandboxOrder"][0]
    assert order["instrumentId"] == "uid-sber" and order["quantity"] == "2"
    assert order["direction"] == "ORDER_DIRECTION_BUY" and order["orderType"] == "ORDER_TYPE_MARKET"
    assert order["accountId"] == "acc1" and order["orderId"]


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("tickerz: [SBER]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
