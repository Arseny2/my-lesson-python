"""Торговый терминал в консоли (библиотека rich)."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .backtest import BacktestReport
from .models import Action, CycleResult

console = Console()

AGENT_TITLES = {
    "technical": "1. Технический аналитик",
    "news": "2. Новостной аналитик",
    "sentiment": "3. Поведение трейдеров",
}


def _sig(x: float) -> str:
    color = "green" if x > 0.15 else "red" if x < -0.15 else "yellow"
    bar = "█" * int(abs(x) * 10)
    return f"[{color}]{x:+.2f} {bar}[/{color}]"


def show_cycle(r: CycleResult) -> None:
    t = Table(title=f"{r.ticker} · {r.price:,.2f} ₽ · {r.time}", show_lines=True, expand=True)
    t.add_column("Агент", style="bold", no_wrap=True)
    t.add_column("Сигнал", no_wrap=True)
    t.add_column("Увер.", no_wrap=True)
    t.add_column("Вывод")
    for key, rep in r.reports.items():
        points = "\n".join(f"• {p}" for p in rep.key_points[:3])
        t.add_row(AGENT_TITLES.get(key, key), _sig(rep.signal), f"{rep.confidence:.2f}",
                  rep.summary + (f"\n[dim]{points}[/dim]" if points else ""))
    if r.debate:
        d = r.debate
        winner = {"bull": "[green]Бык[/green]", "bear": "[red]Медведь[/red]", "draw": "ничья"}[d.winner]
        t.add_row("4. Бык и Медведь", _sig(d.signal), f"{d.confidence:.2f}",
                  f"Победил: {winner}\n[green]За:[/green] {d.bull_case[:300]}\n[red]Против:[/red] {d.bear_case[:300]}")
    if r.decision:
        dc = r.decision
        color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}[dc.action.value]
        t.add_row("5. Трейдер", f"[{color}]{dc.action.value}[/{color}]", f"{dc.size_pct:.0%}",
                  f"{dc.reasoning}\n[dim]стоп {dc.stop_loss_pct:.1%} · тейк {dc.take_profit_pct:.1%}[/dim]")
    if r.verdict:
        v = r.verdict
        mark = "[green]ОДОБРЕНО[/green]" if v.approved else "[red]ОТКЛОНЕНО[/red]"
        t.add_row("6. Риск-менеджер", mark, f"{v.size_pct:.0%}", v.reason)
    console.print(t)
    if r.fill:
        f = r.fill
        color = "green" if f.side == Action.BUY else "red"
        console.print(Panel(f"[{color}]{f.side.value}[/{color}] {f.ticker}: {f.lots} лот(ов) по {f.price:,.2f} ₽ "
                            f"(комиссия {f.commission:,.2f} ₽) - {f.reason}", title="Сделка"))


def show_portfolio(p: dict, broker_name: str) -> None:
    t = Table(title=f"Портфель ({broker_name})", expand=True)
    for col in ("Бумага", "Лоты", "Шт.", "Ср. цена", "Цена", "Стоимость", "P&L", "Стоп", "Тейк"):
        t.add_column(col)
    for tk, pos in p["positions"].items():
        pnl = pos["pnl_pct"]
        t.add_row(tk, str(pos["lots"]), str(pos["shares"]), f"{pos['avg_price']:,.2f}", f"{pos['price']:,.2f}",
                  f"{pos['value']:,.0f}", f"[{'green' if pnl >= 0 else 'red'}]{pnl:+.2%}[/]",
                  f"{pos['stop']:,.2f}" if pos.get("stop") else "-", f"{pos['take']:,.2f}" if pos.get("take") else "-")
    console.print(t)
    console.print(f"Деньги: [bold]{p['cash']:,.0f} ₽[/bold] · В акциях: {p['exposure']:,.0f} ₽ · "
                  f"Всего: [bold]{p['equity']:,.0f} ₽[/bold]")


def show_backtest(r: BacktestReport) -> None:
    good = lambda x: "green" if x >= 0 else "red"
    t = Table(title=f"Бэктест {r.ticker} · {r.bars} свечей")
    t.add_column("Метрика")
    t.add_column("Значение", justify="right")
    t.add_row("Начальный капитал", f"{r.start_equity:,.0f} ₽")
    t.add_row("Итоговый капитал", f"{r.end_equity:,.0f} ₽")
    t.add_row("Доходность бота", f"[{good(r.total_return)}]{r.total_return:+.2%}[/]")
    t.add_row("Купить и держать", f"[{good(r.buy_hold_return)}]{r.buy_hold_return:+.2%}[/]")
    t.add_row("Макс. просадка", f"[red]{r.max_drawdown:.2%}[/red]")
    t.add_row("Коэф. Шарпа", f"{r.sharpe:.2f}")
    t.add_row("Сделок", str(r.trades))
    t.add_row("Прибыльных", f"{r.win_rate:.0%}")
    console.print(t)
    a = Table(title="Чему научился бот: точность агентов")
    a.add_column("Агент")
    a.add_column("Проверено сигналов", justify="right")
    a.add_column("Угадал", justify="right")
    for k, v in r.agent_accuracy.items():
        a.add_row(k, str(v["total"]), f"{v['acc']:.0%}" if v["total"] else "-")
    console.print(a)
    if r.lessons:
        console.print(Panel("\n".join(f"• {l}" for l in r.lessons), title="Последние уроки"))


def show_selection(s: dict) -> None:
    """Итоги подбора стратегии (команда select)."""
    rows = s.get("results", [])
    if rows:
        t = Table(title="Сравнение таймфреймов (с комиссией)")
        for col in ("Бумага", "Свеча", "Порог", "Обучение", "Проверка", "Сделок", "Прибыльных", "Просадка",
                    "Купить и держать"):
            t.add_column(col, justify="right")
        c = lambda x: f"[{'green' if x > 0 else 'red'}]{x:+.2%}[/]"
        for r in sorted(rows, key=lambda r: r["test_return"], reverse=True):
            t.add_row(r["ticker"], f"{r['interval']} мин", f"{r['threshold']:.1f}", c(r["train_return"]),
                      c(r["test_return"]), str(r["test_trades"]), f"{r['test_win_rate']:.0%}",
                      f"{r['test_max_dd']:.1%}", c(r["test_buy_hold"]))
        console.print(t)
    if s.get("profitable"):
        console.print(Panel(f"Торгуем: [bold]{', '.join(s['tickers'])}[/bold] на свечах [bold]{s['interval']} мин[/bold]"
                            f"\nПороги входа: {s.get('thresholds')}", title="[green]Выбрана стратегия[/green]"))
    else:
        console.print(Panel(f"{s.get('reason')}\n\nРеальную торговлю бот запускать не будет (только --force). "
                            "Попробуйте тариф с меньшей комиссией, больше капитал или другие бумаги.",
                            title="[red]Прибыльной стратегии не найдено[/red]"))
