"""Торговый бот с командой из 6 ИИ-агентов для акций Мосбиржи.

Примеры:
  python main.py demo                       # всё на синтетических данных, без интернета и ключей
  python main.py analyze SBER               # один анализ бумаги, без сделок
  python main.py backtest SBER --days 180   # проверка стратегии на истории
  python main.py run                        # торговля (режим из config.yaml: paper / tinvest_sandbox / tinvest_live)
  python main.py status                     # портфель и статистика обучения
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tbot.backtest import run_backtest
from tbot.brokers import PaperBroker, TInvestBroker
from tbot.config import Config, load_config
from tbot.data.moex import MoexClient, MoexError, now_str
from tbot.data.news import NewsSource
from tbot.data.synthetic import make_candles, make_trades
from tbot.engine import TradingEngine
from tbot.llm import LLM
from tbot.memory import Memory
from tbot.terminal import console, show_backtest, show_cycle, show_portfolio

MSK = ZoneInfo("Europe/Moscow")
log = logging.getLogger("tbot")


def make_llm(cfg: Config, enabled: bool) -> LLM | None:
    if not enabled or not cfg.use_llm:
        return None
    if not cfg.anthropic_available:
        console.print("[yellow]ANTHROPIC_API_KEY не задан - агенты работают по алгоритмам без ИИ.[/yellow]")
        return None
    return LLM(cfg.llm_model, cfg.llm_effort)


def make_broker(cfg: Config, confirm_live: bool):
    data = Path(cfg.data_dir)
    if cfg.broker == "paper":
        return PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct, data / "paper_state.json")
    if cfg.broker == "tinvest_sandbox":
        return TInvestBroker(cfg.tinvest_token, sandbox=True, account_id=cfg.tinvest_account_id,
                             class_code=cfg.board, state_file=data / "sandbox_meta.json",
                             sandbox_start_cash=cfg.start_cash)
    if cfg.broker == "tinvest_live":
        if not cfg.live_trading_enabled or not confirm_live:
            sys.exit("Реальная торговля выключена. Нужно: live_trading_enabled: true в config.yaml "
                     "И флаг --i-understand-real-money при запуске.")
        console.print("[bold red]ВНИМАНИЕ: бот будет торговать РЕАЛЬНЫМИ деньгами.[/bold red]")
        if input('Напишите "ДА", чтобы продолжить: ').strip() != "ДА":
            sys.exit("Отменено.")
        return TInvestBroker(cfg.tinvest_token, sandbox=False, account_id=cfg.tinvest_account_id,
                             class_code=cfg.board, state_file=data / "live_meta.json")
    sys.exit(f"Неизвестный broker: {cfg.broker}")


def market_open(now: datetime | None = None) -> bool:
    """Основная (10:00-18:40) и вечерняя (19:05-23:50) сессии Мосбиржи, будни, время МСК."""
    now = now or datetime.now(MSK)
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return 10 * 60 <= hm <= 18 * 60 + 40 or 19 * 60 + 5 <= hm <= 23 * 60 + 50


def cycle_live(engine: TradingEngine, moex: MoexClient, news: NewsSource, cfg: Config, ticker: str, trade: bool):
    info = moex.security_info(ticker)
    candles = moex.candles(ticker, cfg.candle_interval, cfg.history_days)
    try:
        trades = moex.trades(ticker)
    except Exception as e:
        log.warning("Лента сделок %s недоступна: %s", ticker, e)
        trades = None
    price = info["last"] or float(candles["close"].iloc[-1])
    return engine.run_cycle(ticker, candles, name=info["name"], lot_size=info["lot_size"], trades=trades,
                            news=news.for_ticker(ticker), market_news=news.market(), exec_price=price,
                            time=now_str(), trade=trade)


def cmd_demo(cfg: Config, args) -> None:
    console.rule("[bold]ДЕМО на синтетических данных")
    llm = make_llm(cfg, args.llm)
    candles = make_candles(800, seed=7)
    rep = run_backtest(cfg, "DEMO", candles, lot_size=10, llm=None)
    show_backtest(rep)
    console.rule("Один цикл анализа (как это выглядит вживую)")
    broker = PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct)
    engine = TradingEngine(cfg, broker, Memory(), llm)
    r = engine.run_cycle("DEMO", candles, name="Демо-акция", lot_size=10,
                         trades=make_trades(price=float(candles["close"].iloc[-1]), buy_share=0.6))
    show_cycle(r)
    show_portfolio(broker.portfolio(engine.prices), broker.name)


def cmd_analyze(cfg: Config, args) -> None:
    llm = make_llm(cfg, not args.no_llm)
    broker = PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct)
    engine = TradingEngine(cfg, broker, Memory(Path(cfg.data_dir) / "memory.json"), llm)
    moex, news = MoexClient(cfg.board), NewsSource(cfg.news_feeds, cfg.company_keywords)
    for t in args.tickers or cfg.tickers:
        show_cycle(cycle_live(engine, moex, news, cfg, t, trade=False))
    if llm:
        console.print(f"[dim]Claude: {llm.calls} запросов, ~${llm.cost_usd:.3f}[/dim]")


def cmd_backtest(cfg: Config, args) -> None:
    llm = make_llm(cfg, args.llm)
    if llm:
        console.print("[yellow]Бэктест с Claude: 5+ платных запросов на КАЖДУЮ свечу. Ограничьте --days.[/yellow]")
    for t in args.tickers or cfg.tickers:
        if args.synthetic:
            candles, lot = make_candles(1000, seed=zlib.crc32(t.encode()) % 1000), 10
        else:
            moex = MoexClient(cfg.board)
            candles = moex.candles(t, cfg.candle_interval, args.days)
            lot = moex.security_info(t)["lot_size"]
        console.print(f"{t}: {len(candles)} свечей, загружено")
        show_backtest(run_backtest(cfg, t, candles, lot_size=lot, llm=llm))


def cmd_run(cfg: Config, args) -> None:
    broker = make_broker(cfg, args.i_understand_real_money)
    llm = make_llm(cfg, not args.no_llm)
    engine = TradingEngine(cfg, broker, Memory(Path(cfg.data_dir) / "memory.json"), llm)
    moex, news = MoexClient(cfg.board), NewsSource(cfg.news_feeds, cfg.company_keywords)
    console.print(f"Брокер: [bold]{broker.name}[/bold] · бумаги: {', '.join(cfg.tickers)} · "
                  f"цикл каждые {cfg.poll_seconds} c · Ctrl+C - остановка")
    while True:
        if not market_open() and not args.ignore_hours:
            console.print(f"[dim]{now_str()} биржа закрыта, жду...[/dim]")
        else:
            for t in cfg.tickers:
                try:
                    show_cycle(cycle_live(engine, moex, news, cfg, t, trade=True))
                except MoexError as e:
                    log.error("%s: нет данных с биржи - %s", t, e)
                except Exception as e:  # ошибка по одной бумаге не должна останавливать бота
                    log.exception("Ошибка цикла %s: %s", t, e)
            show_portfolio(broker.portfolio(engine.prices), broker.name)
            if llm:
                console.print(f"[dim]Claude: {llm.calls} запросов, ~${llm.cost_usd:.2f} за сессию[/dim]")
        if args.once:
            break
        time.sleep(cfg.poll_seconds)


def cmd_status(cfg: Config, args) -> None:
    data = Path(cfg.data_dir)
    broker = PaperBroker(state_file=data / "paper_state.json")
    show_portfolio(broker.portfolio({}), "paper (цены на момент сделок)")
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
    s = sub.add_parser("analyze", help="анализ без сделок")
    s.add_argument("tickers", nargs="*")
    s.add_argument("--no-llm", action="store_true")
    s = sub.add_parser("backtest", help="проверка на истории")
    s.add_argument("tickers", nargs="*")
    s.add_argument("--days", type=int, default=180)
    s.add_argument("--synthetic", action="store_true", help="синтетические данные вместо MOEX")
    s.add_argument("--llm", action="store_true", help="агенты на Claude (дорого!)")
    s = sub.add_parser("run", help="торговля")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--once", action="store_true", help="один цикл и выход")
    s.add_argument("--ignore-hours", action="store_true", help="не проверять часы работы биржи")
    s.add_argument("--i-understand-real-money", action="store_true")
    sub.add_parser("status", help="портфель и обучение")

    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx/anthropic слишком болтливы на INFO
    for noisy in ("httpx", "httpx2", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    {"demo": cmd_demo, "analyze": cmd_analyze, "backtest": cmd_backtest,
     "run": cmd_run, "status": cmd_status}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
