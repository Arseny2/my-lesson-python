"""Торговый бот с командой из 6 ИИ-агентов для акций Мосбиржи (T-Invest).

Порядок работы:
  python main.py demo         # проверка без интернета и ключей
  python main.py select       # выбрать акции и таймфрейм (минутки / 10 мин / час) по истории
  python main.py analyze      # что думают агенты прямо сейчас, без сделок
  python main.py run          # торговая сессия (6 часов по умолчанию)
  python main.py status       # портфель и чему бот научился
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from tbot.backtest import run_backtest
from tbot.brokers import PaperBroker, TInvestBroker
from tbot.config import Config, load_config
from tbot.data.moex import MoexClient, now_str
from tbot.data.news import NewsSource
from tbot.data.synthetic import make_candles, make_trades
from tbot.data.tinvest_data import TInvestData
from tbot.engine import TradingEngine
from tbot.llm import LLM
from tbot.memory import Memory
from tbot.strategy_select import load_strategy, save_strategy, select_strategy
from tbot.terminal import console, show_backtest, show_cycle, show_portfolio, show_selection
from tbot.tinvest_api import TInvestClient

MSK = ZoneInfo("Europe/Moscow")
log = logging.getLogger("tbot")


# ---------------- фабрики ----------------
def make_llm(cfg: Config, enabled: bool, budget: bool = True) -> LLM | None:
    if not enabled or not cfg.use_llm:
        return None
    if not cfg.anthropic_available:
        console.print("[yellow]ANTHROPIC_API_KEY не задан - агенты работают по алгоритмам без ИИ.[/yellow]")
        return None
    return LLM(cfg.llm_model, cfg.llm_effort,
               budget_usd=cfg.llm_session_budget_usd if budget else None, session_hours=cfg.session_hours)


def make_data(cfg: Config):
    """Источник текущих данных. T-Invest - без задержки, MOEX ISS - с задержкой ~15 минут."""
    if cfg.data_source == "tinvest":
        if not cfg.tinvest_token:
            console.print("[yellow]Нет TINVEST_TOKEN - беру данные MOEX ISS с задержкой ~15 минут. "
                          "Для минутной торговли это не годится.[/yellow]")
            return MoexClient(cfg.board)
        return TInvestData(TInvestClient(cfg.tinvest_token, cfg.board))
    return MoexClient(cfg.board)


def make_broker(cfg: Config, args):
    data = Path(cfg.data_dir)
    if cfg.broker == "paper":
        state = data / "paper_state.json"
        if getattr(args, "reset_paper", False) and state.exists():
            state.unlink()
        return PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct, state)
    if cfg.broker == "tinvest_sandbox":
        return TInvestBroker(cfg.tinvest_token, sandbox=True, account_id=cfg.tinvest_account_id,
                             class_code=cfg.board, state_file=data / "sandbox_meta.json",
                             sandbox_start_cash=cfg.start_cash)
    if cfg.broker == "tinvest_live":
        if not cfg.live_trading_enabled or not args.i_understand_real_money:
            sys.exit("Реальная торговля выключена. Нужно: live_trading_enabled: true в config.yaml "
                     "И флаг --i-understand-real-money при запуске.")
        console.print("[bold red]ВНИМАНИЕ: бот будет торговать РЕАЛЬНЫМИ деньгами.[/bold red]")
        if input('Напишите "ДА", чтобы продолжить: ').strip() != "ДА":
            sys.exit("Отменено.")
        return TInvestBroker(cfg.tinvest_token, sandbox=False, account_id=cfg.tinvest_account_id,
                             class_code=cfg.board, state_file=data / "live_meta.json")
    sys.exit(f"Неизвестный broker: {cfg.broker}")


def resolve_plan(cfg: Config, need_profitable: bool, force: bool) -> tuple[list[str], int, dict]:
    """Какие бумаги и какой таймфрейм использовать: из config.yaml или из результатов select."""
    auto = cfg.tickers == "auto" or cfg.candle_interval == "auto"
    if not auto:
        return list(cfg.tickers), int(cfg.candle_interval), {}
    strategy = load_strategy(Path(cfg.data_dir) / "strategy.json")
    if not strategy:
        sys.exit("Таймфрейм или бумаги стоят на auto - сначала запустите: python main.py select")
    if not strategy.get("profitable"):
        msg = f"select не нашёл прибыльной стратегии: {strategy.get('reason')}"
        if need_profitable and not force:
            sys.exit(msg + "\nТорговля реальными деньгами отменена (можно --force, но не советую).")
        console.print(f"[yellow]{msg}[/yellow]")
        if not strategy.get("tickers"):
            sys.exit("Нет бумаг для торговли. Укажите tickers и candle_interval в config.yaml вручную.")
    tickers = strategy["tickers"] if cfg.tickers == "auto" else list(cfg.tickers)
    interval = strategy["interval"] if cfg.candle_interval == "auto" else int(cfg.candle_interval)
    return tickers, interval, strategy.get("thresholds", {})


# ---------------- часы биржи ----------------
def session_end_today(now: datetime) -> datetime | None:
    """Когда закончится текущая сессия Мосбиржи (основная 10:00-18:40, вечерняя 19:05-23:50 МСК)."""
    if now.weekday() >= 5:
        return None
    day = now.replace(second=0, microsecond=0)
    for start, stop in (((10, 0), (18, 40)), ((19, 5), (23, 50))):
        s = day.replace(hour=start[0], minute=start[1])
        e = day.replace(hour=stop[0], minute=stop[1])
        if s <= now < e:
            return e
    return None


def sleep_until_next_candle(interval: int, end: datetime) -> None:
    """Ждём закрытия следующей свечи (+5 секунд, чтобы биржа успела её отдать)."""
    now = datetime.now(MSK)
    minutes = now.hour * 60 + now.minute
    step = min(interval, 60)
    nxt = now.replace(hour=0, minute=0, second=5, microsecond=0) + timedelta(minutes=(minutes // step + 1) * step)
    time.sleep(max(1.0, (min(nxt, end) - now).total_seconds()))


# ---------------- один цикл с живыми данными ----------------
class LiveFeed:
    def __init__(self, cfg: Config, data, news: NewsSource):
        self.cfg, self.data, self.news = cfg, data, news
        self.info: dict[str, dict] = {}

    def cycle(self, engine: TradingEngine, ticker: str, interval: int, trade: bool, threshold=None):
        if ticker not in self.info:
            self.info[ticker] = self.data.info(ticker)
        info = self.info[ticker]
        # агентам нужно ~300 свечей истории
        days = 400 if interval >= 1440 else max(1, math.ceil(300 * interval / (60 * 14)) + 2)
        candles = self.data.candles(ticker, interval, days)
        price = self.data.last_price(ticker) if hasattr(self.data, "last_price") else None
        price = price or float(candles["close"].iloc[-1])
        book = trades = None
        try:
            book = self.data.order_book(ticker)
            trades = self.data.trades(ticker)
        except Exception as e:
            log.warning("%s: стакан/лента недоступны: %s", ticker, e)
        return engine.run_cycle(ticker, candles, name=info["name"], lot_size=info["lot_size"], trades=trades,
                                book=book, news=self.news.for_ticker(ticker), market_news=self.news.market(),
                                exec_price=price, time=now_str(), trade=trade, entry_threshold=threshold)


# ---------------- команды ----------------
def cmd_demo(cfg: Config, args) -> None:
    console.rule(f"[bold]ДЕМО на синтетических данных (минутные свечи, капитал {cfg.start_cash:,.0f} ₽)")
    llm = make_llm(cfg, args.llm, budget=False)
    c = dataclasses.replace(cfg, candle_interval=1)
    candles = make_candles(3000, start_price=100, vol=0.0015, freq="min", seed=7)
    show_backtest(run_backtest(c, "DEMO", candles, lot_size=1))
    console.print("[yellow]Внимание: в синтетике искусственно сильные тренды - эти цифры показывают только, "
                  "что всё работает, а НЕ будущую доходность. Реальную картину даёт: python main.py select[/yellow]")
    console.rule("Один цикл анализа (как это выглядит вживую)")
    broker = PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct)
    engine = TradingEngine(c, broker, Memory(), llm)
    last = float(candles["close"].iloc[-1])
    book = {"bids": [(last * (1 - i * 1e-4), 50 + i) for i in range(1, 11)],
            "asks": [(last * (1 + i * 1e-4), 30 + i) for i in range(1, 11)]}
    r = engine.run_cycle("DEMO", candles, name="Демо-акция", lot_size=1,
                         trades=make_trades(price=last, buy_share=0.6), book=book)
    show_cycle(r)
    show_portfolio(broker.portfolio(engine.prices), broker.name)


def cmd_select(cfg: Config, args) -> None:
    data = make_data(cfg)
    history = MoexClient(cfg.board)   # глубокая бесплатная история
    console.rule(f"[bold]Подбор стратегии: капитал {cfg.start_cash:,.0f} ₽, таймфреймы {cfg.test_intervals} мин")
    strategy = select_strategy(cfg, data, history, log_fn=console.print, max_tickers=args.pool)
    save_strategy(Path(cfg.data_dir) / "strategy.json", strategy)
    show_selection(strategy)


def cmd_analyze(cfg: Config, args) -> None:
    llm = make_llm(cfg, not args.no_llm, budget=False)
    if args.tickers:
        tickers, interval, thresholds = args.tickers, args.interval, {}
    else:
        tickers, interval, thresholds = resolve_plan(cfg, need_profitable=False, force=True)
    c = dataclasses.replace(cfg, candle_interval=interval, llm_mode="always")  # хотим увидеть мнение Claude
    broker = PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct)
    engine = TradingEngine(c, broker, Memory(Path(cfg.data_dir) / "memory.json"), llm)
    feed = LiveFeed(cfg, make_data(cfg), NewsSource(cfg.news_feeds, cfg.company_keywords))
    for t in tickers:
        show_cycle(feed.cycle(engine, t, interval, trade=False, threshold=thresholds.get(t)))
    if llm:
        console.print(f"[dim]Claude: {llm.calls} запросов, ~${llm.cost_usd:.3f}[/dim]")


def cmd_backtest(cfg: Config, args) -> None:
    llm = make_llm(cfg, args.llm, budget=False)
    if llm:
        console.print("[yellow]Бэктест с Claude: несколько платных запросов на КАЖДУЮ свечу. Ограничьте --days.[/yellow]")
    tickers = args.tickers or ([] if cfg.tickers == "auto" else cfg.tickers)
    if not tickers:
        sys.exit("Укажите бумаги: python main.py backtest VTBR AFLT --interval 1")
    c = dataclasses.replace(cfg, candle_interval=args.interval)
    for t in tickers:
        if args.synthetic:
            candles = make_candles(3000, start_price=100, vol=0.0015, freq="min", seed=zlib.crc32(t.encode()))
            lot = 1
        else:
            moex = MoexClient(cfg.board)
            candles = moex.candles(t, args.interval, args.days)
            lot = moex.security_info(t)["lot_size"]
        console.print(f"{t}: {len(candles)} свечей по {args.interval} мин")
        show_backtest(run_backtest(c, t, candles, lot_size=lot, llm=llm))


def cmd_run(cfg: Config, args) -> None:
    if args.hours:
        cfg.session_hours = args.hours
    broker = make_broker(cfg, args)
    tickers, interval, thresholds = resolve_plan(cfg, need_profitable=broker.is_real_money, force=args.force)
    llm = make_llm(cfg, not args.no_llm)
    c = dataclasses.replace(cfg, candle_interval=interval)
    engine = TradingEngine(c, broker, Memory(Path(cfg.data_dir) / "memory.json"), llm)
    feed = LiveFeed(cfg, make_data(cfg), NewsSource(cfg.news_feeds, cfg.company_keywords))
    end = datetime.now(MSK) + timedelta(hours=cfg.session_hours)
    start_equity = broker.portfolio({})["equity"]
    trades_done = 0
    budget = f"бюджет Claude ${cfg.llm_session_budget_usd:.2f}" if llm else "без Claude"
    console.print(f"Брокер [bold]{broker.name}[/bold] · бумаги {', '.join(tickers)} · свечи {interval} мин · "
                  f"до {end:%H:%M} МСК · {budget} · Ctrl+C - остановка")
    try:
        while datetime.now(MSK) < end:
            now = datetime.now(MSK)
            sess_end = session_end_today(now)
            if not sess_end and not args.ignore_hours:
                console.print(f"[dim]{now:%H:%M} биржа закрыта, жду...[/dim]")
                if args.once:
                    break
                time.sleep(60)
                continue
            # перед закрытием биржи или концом сессии бота - выходим из позиций
            near_close = sess_end is not None and sess_end - now <= timedelta(minutes=max(2, 2 * min(interval, 30)))
            near_end = end - now <= timedelta(minutes=min(interval, 30))
            if (near_close or near_end) and cfg.flat_at_session_end:
                for f in engine.close_all(now_str(), reason="конец сессии"):
                    trades_done += 1
                    console.print(f"[red]SELL[/red] {f.ticker} {f.lots} лот(ов) по {f.price:,.2f} - конец сессии")
                if near_end:
                    break
            else:
                for t in tickers:
                    try:
                        r = feed.cycle(engine, t, interval, trade=True, threshold=thresholds.get(t))
                        trades_done += int(r.fill is not None)
                        if r.fill or r.used_llm or args.verbose_cycles:
                            show_cycle(r)
                        else:
                            console.print(f"[dim]{now:%H:%M} {t} {r.price:,.2f} сигнал {r.debate.signal:+.2f} "
                                          f"→ {r.decision.action.value}: {r.verdict.reason[:70]}[/dim]")
                    except Exception as e:  # ошибка по одной бумаге не должна останавливать бота
                        log.error("%s: ошибка цикла - %s", t, e)
                if llm:
                    console.print(f"[dim]Claude: {llm.calls} запросов, ~${llm.cost_usd:.2f} из "
                                  f"${llm.budget_usd:.2f}[/dim]")
            if args.once:
                break
            sleep_until_next_candle(interval, end)
    except KeyboardInterrupt:
        console.print("\nОстановка по Ctrl+C")
        if cfg.flat_at_session_end:
            trades_done += len(engine.close_all(now_str(), reason="ручная остановка"))
    p = broker.portfolio(engine.prices)
    show_portfolio(p, broker.name)
    pnl = p["equity"] - start_equity
    console.rule(f"Итог сессии: {pnl:+,.2f} ₽ ({pnl / start_equity:+.2%}) · сделок {trades_done}"
                 + (f" · Claude ${llm.cost_usd:.2f}" if llm else ""))


def cmd_status(cfg: Config, args) -> None:
    data = Path(cfg.data_dir)
    broker = PaperBroker(state_file=data / "paper_state.json")
    show_portfolio(broker.portfolio({}), "paper (цены на момент сделок)")
    strategy = load_strategy(data / "strategy.json")
    if strategy:
        show_selection(strategy)
    mem = Memory(data / "memory.json")
    for k, v in mem.stats.items():
        console.print(f"{k:10s} проверено {v['total']:4d}  точность {v['acc']:.0%}  вес {mem.weights()[k]}")
    for l in mem.lessons[-10:]:
        console.print(f"• {l}")


def main() -> None:
    p = argparse.ArgumentParser(description="Торговый бот: 6 ИИ-агентов для Мосбиржи")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="демо без интернета")
    s.add_argument("--llm", action="store_true", help="использовать Claude в цикле анализа")
    s = sub.add_parser("select", help="подобрать бумаги и таймфрейм по истории")
    s.add_argument("--pool", type=int, default=None, help="сколько бумаг проверять (по умолчанию 2 x max_tickers)")
    s = sub.add_parser("analyze", help="анализ без сделок")
    s.add_argument("tickers", nargs="*")
    s.add_argument("--interval", type=int, default=1)
    s.add_argument("--no-llm", action="store_true")
    s = sub.add_parser("backtest", help="проверка на истории")
    s.add_argument("tickers", nargs="*")
    s.add_argument("--interval", type=int, default=1)
    s.add_argument("--days", type=int, default=10)
    s.add_argument("--synthetic", action="store_true", help="синтетические данные вместо MOEX")
    s.add_argument("--llm", action="store_true", help="агенты на Claude (дорого!)")
    s = sub.add_parser("run", help="торговая сессия")
    s.add_argument("--hours", type=float, default=None, help="длительность сессии (по умолчанию session_hours)")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--once", action="store_true", help="один цикл и выход")
    s.add_argument("--ignore-hours", action="store_true", help="не проверять часы работы биржи")
    s.add_argument("--reset-paper", action="store_true", help="начать бумажный счёт заново с start_cash")
    s.add_argument("--verbose-cycles", action="store_true", help="подробная таблица каждый цикл")
    s.add_argument("--force", action="store_true", help="торговать, даже если select не нашёл прибыльной стратегии")
    s.add_argument("--i-understand-real-money", action="store_true")
    sub.add_parser("status", help="портфель и обучение")

    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    {"demo": cmd_demo, "select": cmd_select, "analyze": cmd_analyze, "backtest": cmd_backtest,
     "run": cmd_run, "status": cmd_status}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
