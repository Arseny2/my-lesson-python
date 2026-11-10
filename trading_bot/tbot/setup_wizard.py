"""Мастер настройки и проверка связи - чтобы запустить бота на своём компьютере без правки файлов руками."""
from __future__ import annotations

import getpass
import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def _write_env_value(env_path: Path, key: str, value: str) -> None:
    """Записать KEY=value в .env, не трогая остальные строки."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    out, done = [], False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def ensure_files(root: Path) -> None:
    for src, dst in (("config.example.yaml", "config.yaml"), (".env.example", ".env")):
        if not (root / dst).exists():
            shutil.copy(root / src, root / dst)
            console.print(f"Создан файл [bold]{dst}[/bold]")


def ask_token(root: Path) -> str | None:
    console.print(
        "\n[bold]Токен T-Invest API[/bold]\n"
        "Где взять: приложение Т-Инвестиций → Настройки → Токены T-Invest API → Создать токен.\n"
        "Для начала выберите тип [bold]«Только для чтения»[/bold]: этого хватит для данных и бумажной торговли,\n"
        "а реальными деньгами по такому токену торговать нельзя.\n"
        "[yellow]Никому не отправляйте токен (и в чат тоже): он даёт доступ к вашему счёту.[/yellow]\n"
        "Токен сохранится только на этом компьютере, в файле .env.\n")
    token = getpass.getpass("Вставьте токен и нажмите Enter (символы не отображаются; пусто - пропустить): ").strip()
    if not token:
        console.print("Токен не введён - бот будет брать данные Мосбиржи с задержкой ~15 минут.")
        return None
    _write_env_value(root / ".env", "TINVEST_TOKEN", token)
    console.print("[green]Токен сохранён в .env[/green]")
    return token


def check_connections(token: str | None, board: str = "TQBR") -> bool:
    """Проверить связь с Мосбиржей, T-Invest и новостями. Возвращает True, если T-Invest доступен."""
    from .data.moex import MoexClient
    from .data.news import fetch_feed

    console.rule("Проверка связи")
    try:
        info = MoexClient(board).info("SBER")
        console.print(f"[green]✓[/green] Мосбиржа (история): {info['name']}, лот {info['lot_size']}")
    except Exception as e:
        console.print(f"[red]✗[/red] Мосбиржа недоступна: {e}")

    for url in ("https://www.interfax.ru/rss.asp", "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"):
        try:
            console.print(f"[green]✓[/green] Новости: {len(fetch_feed(url))} шт. ({url.split('/')[2]})")
        except Exception as e:
            console.print(f"[yellow]![/yellow] Новости {url.split('/')[2]} недоступны: {e}")

    if not token:
        console.print("[yellow]![/yellow] T-Invest: токен не задан")
        return False
    from .data.tinvest_data import TInvestData
    from .tinvest_api import TInvestClient, TInvestError
    try:
        client = TInvestClient(token, board)
        accs = client.call("UsersService", "GetAccounts", {}).get("accounts", [])
        console.print(f"[green]✓[/green] T-Invest: токен работает, счетов: {len(accs)}")
        for a in accs:
            console.print(f"    {a.get('name') or 'счёт'} - {a.get('id')} ({a.get('status', '').replace('ACCOUNT_STATUS_', '')})")
        price = TInvestData(client).last_price("SBER")
        console.print(f"[green]✓[/green] T-Invest: цена SBER в реальном времени {price}")
        return True
    except TInvestError as e:
        msg = str(e)
        if "401" in msg or "UNAUTHENTICATED" in msg.upper():
            console.print("[red]✗[/red] T-Invest: токен неверный или отозван - создайте новый и запустите настройку снова")
        else:
            console.print(f"[red]✗[/red] T-Invest: {msg}")
    except Exception as e:
        console.print(f"[red]✗[/red] T-Invest недоступен: {e}")
    return False


def run_setup(root: Path) -> None:
    console.rule("[bold]Настройка торгового бота")
    ensure_files(root)
    from .config import load_config
    cfg = load_config(root / "config.yaml")
    token = cfg.tinvest_token
    if token:
        console.print("Токен T-Invest уже сохранён.")
        if input("Заменить его? (д/н): ").strip().lower() in ("д", "да", "y", "yes"):
            token = ask_token(root) or token
    else:
        token = ask_token(root)
    check_connections(token, cfg.board)
    console.print("\nГотово. Дальше: пункт меню «Подобрать акции и таймфрейм».")


MENU = [
    ("1", "Настройка и проверка связи (токен T-Invest)", ["setup"]),
    ("2", "Подобрать акции и таймфрейм по истории (select)", ["select"]),
    ("3", "Бумажная торговля 6 часов с нуля (1000 ₽ виртуально)", ["run", "--reset-paper"]),
    ("4", "Продолжить бумажную торговлю (без сброса счёта)", ["run"]),
    ("5", "Что думают агенты прямо сейчас (без сделок)", ["analyze"]),
    ("6", "Портфель, результаты и чему бот научился", ["status"]),
    ("7", "Демо без интернета", ["demo"]),
    ("0", "Выход", None),
]


def menu() -> list[str] | None:
    console.rule("[bold]Торговый бот · 6 агентов · Мосбиржа")
    for key, title, _ in MENU:
        console.print(f"  [bold]{key}[/bold]  {title}")
    choice = input("\nВыберите пункт: ").strip()
    for key, _, argv in MENU:
        if choice == key:
            return argv
    console.print("Нет такого пункта")
    return []
