"""Загрузка настроек из config.yaml и переменных окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20      # не больше 20% капитала в одной бумаге
    max_total_exposure_pct: float = 0.80  # не больше 80% капитала в акциях всего
    max_daily_loss_pct: float = 0.03    # при дневном убытке 3% торговля останавливается до завтра
    min_stop_loss_pct: float = 0.01
    max_stop_loss_pct: float = 0.08
    atr_stop_multiplier: float = 2.0    # стоп не ближе 2 ATR
    max_trades_per_day: int = 10
    min_confidence: float = 0.35        # слабые сигналы не торгуем


@dataclass
class Config:
    tickers: list[str] = field(default_factory=lambda: ["SBER", "GAZP", "LKOH"])
    board: str = "TQBR"
    candle_interval: int = 60           # минуты: 1, 10, 60, 24*60 (=1440 - день)
    history_days: int = 60
    poll_seconds: int = 3600            # как часто запускать цикл в режиме paper/live
    start_cash: float = 100_000.0
    commission_pct: float = 0.0005      # 0.05% - типичная брокерская комиссия
    slippage_pct: float = 0.0005
    broker: str = "paper"               # paper | tinvest_sandbox | tinvest_live
    live_trading_enabled: bool = False  # второй предохранитель для реальных денег
    use_llm: bool = True
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    news_feeds: list[str] = field(default_factory=lambda: [
        "https://www.interfax.ru/rss.asp",
        "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    ])
    # Ключевые слова для поиска новостей по тикеру
    company_keywords: dict[str, list[str]] = field(default_factory=lambda: {
        "SBER": ["Сбер", "Сбербанк", "SBER"],
        "GAZP": ["Газпром", "GAZP"],
        "LKOH": ["Лукойл", "ЛУКОЙЛ", "LKOH"],
    })
    data_dir: str = "data"
    risk: RiskConfig = field(default_factory=RiskConfig)

    # секреты - только из окружения, никогда не из config.yaml
    tinvest_token: str | None = None
    tinvest_account_id: str | None = None

    @property
    def anthropic_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _load_dotenv(path: Path) -> None:
    """Мини-загрузчик .env, чтобы не тянуть лишнюю зависимость."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    _load_dotenv(path.parent / ".env")
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    risk_raw = raw.pop("risk", {}) or {}
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Неизвестные параметры в {path}: {', '.join(sorted(unknown))}")
    risk_known = {f.name for f in fields(RiskConfig)}
    bad_risk = set(risk_raw) - risk_known
    if bad_risk:
        raise ValueError(f"Неизвестные параметры risk в {path}: {', '.join(sorted(bad_risk))}")

    cfg = Config(**raw, risk=RiskConfig(**risk_raw))
    cfg.tinvest_token = os.environ.get("TINVEST_TOKEN") or None
    cfg.tinvest_account_id = os.environ.get("TINVEST_ACCOUNT_ID") or None
    return cfg
