import json
import os
import sys
from pathlib import Path

import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")  # реальных запросов тесты не делают

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
    rm = RiskManager(RiskConfig(max_position_pct=0.2, risk_per_trade_pct=1.0))
    v = rm.review(ctx(), buy(0.5), DEB, day_pnl_pct=0, trades_today=0)
    assert v.approved
    assert v.size_pct == pytest.approx(0.2)
    assert v.stop_loss_pct >= 2 * 2.0 / 100        # не ближе 2 ATR
    assert v.take_profit_pct >= v.stop_loss_pct * 1.5


def test_risk_llm_cannot_loosen_limits():
    rm = RiskManager(RiskConfig(max_position_pct=0.1, max_stop_loss_pct=0.05, risk_per_trade_pct=1.0))
    loose = RiskVerdict(approved=True, size_pct=0.9, stop_loss_pct=0.5, take_profit_pct=0.1, reason="ИИ")
    v = rm.enforce(ctx(), buy(0.9), DEB, loose, 0, 0)
    assert v.size_pct <= 0.1 and v.stop_loss_pct <= 0.05


def test_risk_blocks_after_daily_loss_and_low_confidence():
    rm = RiskManager(RiskConfig())
    assert not rm.review(ctx(), buy(), DEB, day_pnl_pct=-0.10, trades_today=0).approved
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


def test_risk_sizes_by_risk_per_trade():
    # капитал 1000, риск на сделку 1% = 10 ₽; стоп 4% + комиссии 0.7% -> позиция ~21% капитала
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.01), round_trip_cost=0.007)
    v = rm.review(ctx(price=10, lot=1, equity=1000, atr=0.2), buy(1.0, stop=0.04, take=0.1), DEB, 0, 0)
    assert v.approved
    assert v.size_pct == pytest.approx(0.01 / (0.04 + 0.007), rel=1e-3)
    assert v.size_pct * 1000 * v.stop_loss_pct <= 10 + 1e-6


def test_risk_rejects_take_that_does_not_cover_commission():
    rm = RiskManager(RiskConfig(min_edge_over_costs=2.0), round_trip_cost=0.007)
    v = rm.review(ctx(price=100, lot=1, equity=1000, atr=0.05), buy(1.0, stop=0.002, take=0.004), DEB, 0, 0)
    assert not v.approved and "комисси" in v.reason


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
    cfg = Config(use_llm=False, candle_interval=60, start_cash=100_000, commission_pct=0.0005)
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
    llm = LLM()
    broker = PaperBroker(100_000, 0, 0)
    cfg = Config(start_cash=100_000, commission_pct=0.0005, llm_mode="always")
    cfg.risk.max_position_pct = 0.2
    engine = TradingEngine(cfg, broker, Memory(), llm)
    candles = make_candles(300)
    r = engine.run_cycle("T", candles, lot_size=1, trades=make_trades(), news=[])
    # новостей нет -> новостной агент не тратит запрос на Claude
    assert calls.count("AgentReport") == 2
    assert {"DebateResult", "TradeDecision", "RiskVerdict"} <= set(calls)
    assert r.fill and r.fill.side == Action.BUY
    assert r.used_llm
    assert r.verdict.size_pct <= 0.2   # ИИ просил 50%, лимит 20%


# ---------- запрос к Claude ----------
def _llm_with_transport(handler, budget=None):
    from anthropic import DefaultHttpxClient
    import anthropic
    llm = LLM("claude-opus-5-5", "medium", budget_usd=budget)
    llm.client = anthropic.Anthropic(api_key="test", max_retries=0,
                                     http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
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


# ---------- бюджет Claude ----------
def test_llm_budget_is_paced():
    llm = _llm_with_transport(lambda r: httpx2.Response(500), budget=1.0)
    llm.session_seconds = 3600
    assert llm.can_spend()
    llm.output_tokens = 10_000   # $0.20 - больше 10% бюджета в самом начале сессии
    assert not llm.can_spend()
    assert llm.ask("роль", "данные", AgentReport) is None and llm.skipped == 1
    llm.started -= 3600          # сессия прошла целиком - можно тратить весь бюджет
    assert llm.can_spend()


def test_escalate_mode_skips_llm_on_weak_signal(monkeypatch):
    calls = []
    monkeypatch.setattr(LLM, "ask", lambda self, *a: calls.append(1))
    cfg = Config(llm_mode="escalate", escalate_threshold=0.99)
    engine = TradingEngine(cfg, PaperBroker(1000, 0, 0), Memory(), LLM())
    r = engine.run_cycle("T", make_candles(300, start_price=100), lot_size=1)
    assert not r.used_llm and not calls


# ---------- подбор стратегии ----------
class FakeData:
    def __init__(self, lots):
        self.lots = lots

    def info(self, t):
        lot, price = self.lots[t]
        return {"ticker": t, "name": t, "lot_size": lot, "last": price, "tradable": True}

    def candles(self, t, interval, days):
        freq = {1: "min", 10: "10min", 60: "h"}[interval]
        return make_candles(900, start_price=self.lots[t][1], vol=0.002 * interval ** 0.5, freq=freq,
                            seed=interval)


def test_select_filters_unaffordable_and_picks_interval(tmp_path):
    from tbot.strategy_select import load_strategy, save_strategy, select_strategy
    data = FakeData({"CHEAP": (1, 100.0), "PRICEY": (10, 300.0)})   # лот 3000 ₽ не влезает в 1000 ₽
    cfg = Config(tickers=["CHEAP", "PRICEY"], test_intervals=[1, 60], commission_pct=0.0005)
    s = select_strategy(cfg, data, data, log_fn=lambda *a: None)
    assert {r["ticker"] for r in s["results"]} == {"CHEAP"}
    assert {r["interval"] for r in s["results"]} == {1, 60}
    if s["profitable"]:
        assert s["interval"] in (1, 60) and s["tickers"] == ["CHEAP"]
    save_strategy(tmp_path / "s.json", s)
    assert load_strategy(tmp_path / "s.json")["results"]


def test_book_pressure():
    from tbot.behavior import book_pressure
    b = book_pressure({"bids": [(99.9, 300), (99.8, 100)], "asks": [(100.1, 100), (100.2, 100)]})
    assert b["imbalance"] == pytest.approx(0.333, abs=1e-3) and b["spread_pct"] > 0


def test_tinvest_candles_parsing(monkeypatch):
    from tbot.data.tinvest_data import TInvestData
    from tbot.tinvest_api import TInvestClient

    def fake_call(self, service, method, body):
        if method == "ShareBy":
            return {"instrument": {"uid": "u1", "lot": 10, "name": "ВТБ"}}
        if method == "GetCandles":
            assert body["interval"] == "CANDLE_INTERVAL_1_MIN"
            q = lambda x: {"units": str(int(x)), "nano": int(round((x - int(x)) * 1e9))}
            return {"candles": [
                {"open": q(90.1), "high": q(90.5), "low": q(90.0), "close": q(90.4), "volume": "7",
                 "time": "2026-11-09T07:00:00Z", "isComplete": True},
                {"open": q(90.4), "high": q(90.6), "low": q(90.3), "close": q(90.5), "volume": "3",
                 "time": "2026-11-09T07:01:00Z", "isComplete": False},
            ]}
        raise AssertionError(method)

    monkeypatch.setattr(TInvestClient, "call", fake_call)
    d = TInvestData(TInvestClient("tok"))
    df = d.candles("VTBR", 1, 0.01)
    assert len(df) == 1                          # незакрытая свеча отброшена
    assert df["close"].iloc[0] == pytest.approx(90.4) and df["volume"].iloc[0] == 70   # лоты -> штуки
    assert str(df.index[0]) == "2026-11-09 10:00:00"   # UTC -> МСК


# ---------- улучшения относительно TradingAgents ----------
def test_similar_situations_memory_finds_pattern():
    """Память должна понять: при перепроданности (RSI низкий) цена потом росла."""
    import random
    from tbot.memory import situation
    m = Memory(eval_after_cycles=1)
    rnd = random.Random(0)
    for _ in range(400):
        rsi = rnd.uniform(10, 90)
        f = situation({"close": 100, "rsi14": rsi}, {})
        m.record_signals("T", 100.0, {}, "t", features=f, interval=1)
        m.evaluate("T", 101.0 if rsi < 50 else 99.0)
    low = m.similar(situation({"close": 100, "rsi14": 15}, {}), interval=1)
    high = m.similar(situation({"close": 100, "rsi14": 85}, {}), interval=1)
    assert low["up_share"] > 0.9 and high["up_share"] < 0.1
    assert not m.similar(situation({"close": 100, "rsi14": 15}, {}), interval=60)["available"]


def test_relevant_lessons_prefers_similar_situation():
    from tbot.memory import situation
    m = Memory()
    m.lessons = [{"text": "урок про перепроданность", "f": situation({"close": 1, "rsi14": 15}, {}), "ticker": "A"},
                 {"text": "урок про перекупленность", "f": situation({"close": 1, "rsi14": 85}, {}), "ticker": "B"}]
    assert m.relevant_lessons(situation({"close": 1, "rsi14": 20}, {}), "Z", k=1)[0] == "урок про перепроданность"


def test_multi_round_debate_calls_bull_bear_and_judge(monkeypatch):
    from tbot.agents.debate import Argument, BullBearDebate
    seen = []

    def fake_ask(self, role, data, schema):
        seen.append(schema.__name__)
        if schema is Argument:
            return Argument(argument="довод", strength=0.6)
        assert "Раунд 2" in data   # судья видит всю стенограмму
        return DebateResult(bull_case="a", bear_case="b", winner="bull", signal=0.5, confidence=0.6)

    monkeypatch.setattr(LLM, "ask", fake_ask)
    d = BullBearDebate(LLM(), rounds=2)
    rep = AgentReport(signal=0.5, confidence=0.5, summary="", key_points=[])
    out = d.run(ctx(), {"technical": rep}, {})
    assert seen == ["Argument"] * 4 + ["DebateResult"] and out.signal == 0.5


def test_backtest_with_llm_hides_ticker_and_dates(monkeypatch):
    seen = []
    monkeypatch.setattr(LLM, "ask", lambda self, role, data, schema: seen.append(data))
    cfg = Config(candle_interval=60, llm_mode="always", start_cash=100_000)
    run_backtest(cfg, "SBER", make_candles(80), lot_size=1, llm=LLM())
    assert seen and all("SBER" not in d and "2025-" not in d for d in seen)


def test_risk_vetoes_when_similar_situations_lost():
    rm = RiskManager(RiskConfig())
    c = ctx(price=100, lot=1, equity=100_000)
    c.similar = {"available": True, "base": 500, "n": 20, "up_share": 0.2, "avg_return": -0.004}
    v = rm.review(c, buy(0.2, stop=0.05, take=0.1), DEB, 0, 0)
    assert not v.approved and "Память" in v.reason


def test_risk_denies_when_expected_move_below_costs():
    rm = RiskManager(RiskConfig(), round_trip_cost=0.002)
    c = ctx(price=100, lot=1, equity=100_000)
    c.similar = {"available": True, "base": 500, "n": 20, "up_share": 0.7, "avg_return": 0.001}
    v = rm.review(c, buy(0.2, stop=0.05, take=0.1), DEB, 0, 0)
    assert not v.approved and "меньше расходов" in v.reason


def test_risk_stops_trading_without_edge():
    rm = RiskManager(RiskConfig())
    c = ctx(price=100, lot=1, equity=100_000)
    c.agent_stats = {"debate": {"total": 300, "hits": 150, "acc": 0.5}}
    assert "монетки" in rm.review(c, buy(0.2, stop=0.05, take=0.1), DEB, 0, 0).reason


def test_time_exit_closes_stale_position():
    cfg = Config(start_cash=100_000, entry_threshold=0.99)   # новых входов не будет
    cfg.risk.max_hold_cycles = 3
    broker = PaperBroker(100_000, 0, 0)
    engine = TradingEngine(cfg, broker, Memory())
    candles = make_candles(300, start_price=100, vol=0.0005)
    broker.buy("T", 1, 1, float(candles["close"].iloc[-1]), stop=1.0, take=1e9, time="t")
    reasons = []
    for i in range(4):
        r = engine.run_cycle("T", candles.iloc[: 250 + i], lot_size=1)
        if r.fill:
            reasons.append(r.fill.reason)
    assert reasons == ["выход по времени"] and broker.position("T") is None
