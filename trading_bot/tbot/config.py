"""Загрузка настроек из config.yaml и переменных окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 0.03    # дневной убыток 3% - покупки запрещены до конца дня
    risk_per_trade_pct: float = 0.01    # в одной сделке рискуем не больше 1% капитала (до стопа)
    max_position_pct: float = 1.0       # при маленьком капитале можно весь капитал в одну бумагу
    max_total_exposure_pct: float = 1.0
    min_stop_loss_pct: float = 0.002
    max_stop_loss_pct: float = 0.05
    atr_stop_multiplier: float = 2.0    # стоп не ближе 2 ATR
    min_edge_over_costs: float = 2.0    # тейк должен быть хотя бы в 2 раза больше комиссий за круг
    max_trades_per_day: int = 60
    min_confidence: float = 0.35        # слабые сигналы не торгуем


@dataclass
class Config:
    # "auto" - бот сам выберет ликвидные бумаги, лот которых по карману (команда select)
    tickers: list[str] | str = "auto"
    candidates: list[str] = field(default_factory=lambda: [
        "SBER", "SBERP", "GAZP", "LKOH", "ROSN", "VTBR", "NVTK", "GMKN", "YDEX", "T", "TATN", "SNGS",
        "SNGSP", "MOEX", "AFLT", "ALRS", "CHMF", "MAGN", "NLMK", "PLZL", "MTSS", "RUAL", "IRAO", "HYDR",
        "FEES", "PIKK", "SMLT", "OZON", "MGNT", "PHOR", "RNFT", "SGZH", "MTLR", "POSI", "ASTR", "CBOM",
        "FLOT", "BSPB", "UPRO", "SELG",
    ])
    max_tickers: int = 3
    board: str = "TQBR"
    # размер свечи в минутах или "auto" - взять лучший по итогам `python main.py select`
    candle_interval: int | str = "auto"
    test_intervals: list[int] = field(default_factory=lambda: [1, 10, 60])
    history_days: int = 5               # сколько дней истории смотрят агенты
    entry_threshold: float = 0.25       # с какой силы сигнала команды входить в сделку
    session_hours: float = 6.0          # сколько часов работает бот за запуск
    flat_at_session_end: bool = True    # в конце сессии/дня закрыть все позиции
    start_cash: float = 1_000.0
    # Комиссия зависит от тарифа Т-Инвестиций: «Инвестор» ~0.3%, «Трейдер» ~0.05%
    commission_pct: float = 0.003
    slippage_pct: float = 0.0005
    data_source: str = "tinvest"        # tinvest - без задержки (нужен токен) | moex - задержка ~15 мин
    broker: str = "paper"               # paper | tinvest_sandbox | tinvest_live
    live_trading_enabled: bool = False  # второй предохранитель для реальных денег
    use_llm: bool = True
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"             # на минутках важна скорость ответа
    # always - Claude в каждом цикле; escalate - алгоритмы каждую минуту, Claude только при сильном сигнале
    llm_mode: str = "escalate"
    escalate_threshold: float = 0.15
    llm_daily_budget_usd: float = 10.0  # дневной лимит расходов на Claude
    llm_budget_share: float = 0.5       # какую часть дневного лимита можно потратить за сессию
    news_feeds: list[str] = field(default_factory=lambda: [
        "https://www.interfax.ru/rss.asp",
        "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    ])
    # Ключевые слова для поиска новостей по тикеру
    company_keywords: dict[str, list[str]] = field(default_factory=lambda: {
        "SBER": ["Сбер"], "SBERP": ["Сбер"], "GAZP": ["Газпром"], "LKOH": ["Лукойл", "ЛУКОЙЛ"],
        "ROSN": ["Роснефт"], "VTBR": ["ВТБ"], "NVTK": ["Новатэк", "НОВАТЭК"], "GMKN": ["Норникел"],
        "YDEX": ["Яндекс"], "T": ["Т-Технологи", "Т-Банк"], "TATN": ["Татнефт"], "SNGS": ["Сургутнефтегаз"],
        "SNGSP": ["Сургутнефтегаз"], "MOEX": ["Мосбирж"], "AFLT": ["Аэрофлот"], "ALRS": ["АЛРОСА", "Алроса"],
        "CHMF": ["Северстал"], "MAGN": ["ММК"], "NLMK": ["НЛМК"], "PLZL": ["Полюс"], "MTSS": ["МТС"],
        "RUAL": ["Русал"], "IRAO": ["Интер РАО"], "HYDR": ["РусГидро"], "FEES": ["Россети", "ФСК"],
        "PIKK": ["ПИК"], "SMLT": ["Самолет"], "OZON": ["Ozon", "Озон"], "MGNT": ["Магнит"],
        "PHOR": ["ФосАгро"], "RNFT": ["РуссНефт"], "SGZH": ["Сегежа"], "MTLR": ["Мечел"],
        "POSI": ["Positive", "Позитив"], "ASTR": ["Астра"], "CBOM": ["МКБ"], "FLOT": ["Совкомфлот"],
        "BSPB": ["Банк Санкт-Петербург"], "UPRO": ["Юнипро"], "SELG": ["Селигдар"],
    })
    data_dir: str = "data"
    risk: RiskConfig = field(default_factory=RiskConfig)

    # секреты - только из окружения, никогда не из config.yaml
    tinvest_token: str | None = None
    tinvest_account_id: str | None = None

    @property
    def llm_session_budget_usd(self) -> float:
        return self.llm_daily_budget_usd * self.llm_budget_share

    def round_trip_cost_pct(self) -> float:
        """Сколько стоит вход + выход: комиссия и проскальзывание дважды."""
        return 2 * (self.commission_pct + self.slippage_pct)

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
