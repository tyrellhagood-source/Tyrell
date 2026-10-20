"""Central configuration for MoneyOps AI.

Every value can be overridden with an environment variable (a local `.env`
file is loaded automatically). Secrets such as API keys are only ever read from
the environment and are never written to logs, the database or the dashboard.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_VERSION = "ratings-blend-1.0"
APP_VERSION = "1.0"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _tuple(name: str, default: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in _env(name, default).split(",") if s.strip())


DEFAULT_SPORTS = "NFL,NCAAF,NBA,NCAAB,NCAAW,WNBA,MLB,NHL,EPL,MLS,UCL,LALIGA,BUNDESLIGA,SERIEA,LIGUE1,MMA"


@dataclass(frozen=True)
class Settings:
    # ---- data providers -------------------------------------------------
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com"
    odds_regions: tuple[str, ...] = ("us",)
    odds_markets: tuple[str, ...] = ("h2h", "spreads", "totals")
    odds_bookmakers: tuple[str, ...] = ()
    enabled_sports: tuple[str, ...] = tuple(DEFAULT_SPORTS.split(","))
    scan_tennis: bool = False
    scan_other_sports: bool = False
    include_player_props: bool = False
    prop_markets: tuple[str, ...] = ("player_points", "player_rebounds", "player_assists")
    max_prop_events_per_run: int = 3
    max_credits_per_run: int = 60
    min_credits_reserve: int = 25
    espn_enabled: bool = True
    espn_base_url: str = "https://site.api.espn.com/apis/site/v2/sports"
    use_odds_api_scores: bool = False
    http_timeout_seconds: float = 15.0
    http_max_retries: int = 3
    odds_cache_ttl_seconds: int = 300
    schedule_cache_ttl_seconds: int = 900
    recent_history_cache_ttl_seconds: int = 21600
    # ---- freshness ------------------------------------------------------
    stale_odds_minutes_pregame: float = 30.0
    stale_odds_minutes_live: float = 2.0
    max_start_time_diff_minutes: float = 120.0
    # ---- statistics / model --------------------------------------------
    history_lookback_days: int = 400
    history_half_life_days: float = 120.0
    min_games_per_team: int = 5
    min_sample: int = 15
    model_weight: float = 0.35
    max_model_market_gap: float = 0.10
    outlier_threshold: float = 0.04
    bootstrap_samples: int = 40
    random_seed: int = 7
    # ---- value / risk ---------------------------------------------------
    min_books: int = 3
    min_ev: float = 0.02
    min_ev_lower: float = -0.02
    starting_bankroll: float = 25.0
    max_stake_pct: float = 0.01
    parlay_max_stake_pct: float = 0.005
    max_open_exposure_pct: float = 0.05
    max_drawdown_halt_pct: float = 0.25
    kelly_fraction: float = 0.25
    min_stake: float = 0.10
    typical_sportsbook_min_bet: float = 1.00
    max_parlay_legs: int = 3
    parlay_min_ev: float = 0.03
    max_parlays: int = 10
    # ---- runtime --------------------------------------------------------
    timezone: str = "America/New_York"
    refresh_interval_minutes: int = 60
    database_url: str = f"sqlite:///{BASE_DIR / 'moneyops.db'}"
    log_dir: Path = field(default=BASE_DIR / "logs")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            odds_api_key=_env("ODDS_API_KEY", ""),
            odds_api_base_url=_env("ODDS_API_BASE_URL", d.odds_api_base_url),
            odds_regions=_tuple("ODDS_API_REGIONS", ",".join(d.odds_regions)),
            odds_markets=_tuple("ODDS_API_MARKETS", ",".join(d.odds_markets)),
            odds_bookmakers=_tuple("ODDS_API_BOOKMAKERS", ""),
            enabled_sports=_tuple("ENABLED_SPORTS", DEFAULT_SPORTS),
            scan_tennis=_bool("SCAN_TENNIS", d.scan_tennis),
            scan_other_sports=_bool("SCAN_OTHER_SPORTS", d.scan_other_sports),
            include_player_props=_bool("INCLUDE_PLAYER_PROPS", d.include_player_props),
            prop_markets=_tuple("PROP_MARKETS", ",".join(d.prop_markets)),
            max_prop_events_per_run=_int("MAX_PROP_EVENTS_PER_RUN", d.max_prop_events_per_run),
            max_credits_per_run=_int("MAX_CREDITS_PER_RUN", d.max_credits_per_run),
            min_credits_reserve=_int("MIN_CREDITS_RESERVE", d.min_credits_reserve),
            espn_enabled=_bool("ESPN_ENABLED", d.espn_enabled),
            espn_base_url=_env("ESPN_BASE_URL", d.espn_base_url),
            use_odds_api_scores=_bool("USE_ODDS_API_SCORES", d.use_odds_api_scores),
            http_timeout_seconds=_float("HTTP_TIMEOUT_SECONDS", d.http_timeout_seconds),
            http_max_retries=_int("HTTP_MAX_RETRIES", d.http_max_retries),
            odds_cache_ttl_seconds=_int("ODDS_CACHE_TTL_SECONDS", d.odds_cache_ttl_seconds),
            schedule_cache_ttl_seconds=_int("SCHEDULE_CACHE_TTL_SECONDS", d.schedule_cache_ttl_seconds),
            recent_history_cache_ttl_seconds=_int("RECENT_HISTORY_CACHE_TTL_SECONDS", d.recent_history_cache_ttl_seconds),
            stale_odds_minutes_pregame=_float("STALE_ODDS_MINUTES_PREGAME", d.stale_odds_minutes_pregame),
            stale_odds_minutes_live=_float("STALE_ODDS_MINUTES_LIVE", d.stale_odds_minutes_live),
            max_start_time_diff_minutes=_float("MAX_START_TIME_DIFF_MINUTES", d.max_start_time_diff_minutes),
            history_lookback_days=_int("HISTORY_LOOKBACK_DAYS", d.history_lookback_days),
            history_half_life_days=_float("HISTORY_HALF_LIFE_DAYS", d.history_half_life_days),
            min_games_per_team=_int("MIN_GAMES_PER_TEAM", d.min_games_per_team),
            min_sample=_int("MIN_SAMPLE", d.min_sample),
            model_weight=_float("MODEL_WEIGHT", d.model_weight),
            max_model_market_gap=_float("MAX_MODEL_MARKET_GAP", d.max_model_market_gap),
            outlier_threshold=_float("OUTLIER_THRESHOLD", d.outlier_threshold),
            bootstrap_samples=_int("BOOTSTRAP_SAMPLES", d.bootstrap_samples),
            random_seed=_int("RANDOM_SEED", d.random_seed),
            min_books=_int("MIN_BOOKS", d.min_books),
            min_ev=_float("MIN_EV", d.min_ev),
            min_ev_lower=_float("MIN_EV_LOWER", d.min_ev_lower),
            starting_bankroll=_float("STARTING_BANKROLL", d.starting_bankroll),
            max_stake_pct=_float("MAX_STAKE_PCT", d.max_stake_pct),
            parlay_max_stake_pct=_float("PARLAY_MAX_STAKE_PCT", d.parlay_max_stake_pct),
            max_open_exposure_pct=_float("MAX_OPEN_EXPOSURE_PCT", d.max_open_exposure_pct),
            max_drawdown_halt_pct=_float("MAX_DRAWDOWN_HALT_PCT", d.max_drawdown_halt_pct),
            kelly_fraction=_float("KELLY_FRACTION", d.kelly_fraction),
            min_stake=_float("MIN_STAKE", d.min_stake),
            typical_sportsbook_min_bet=_float("TYPICAL_SPORTSBOOK_MIN_BET", d.typical_sportsbook_min_bet),
            max_parlay_legs=_int("MAX_PARLAY_LEGS", d.max_parlay_legs),
            parlay_min_ev=_float("PARLAY_MIN_EV", d.parlay_min_ev),
            max_parlays=_int("MAX_PARLAYS", d.max_parlays),
            timezone=_env("TIMEZONE", d.timezone),
            refresh_interval_minutes=_int("REFRESH_INTERVAL_MINUTES", d.refresh_interval_minutes),
            database_url=_env("DATABASE_URL", d.database_url),
            log_dir=Path(_env("LOG_DIR", str(d.log_dir))),
        )

    def public_dict(self) -> dict:
        """Settings safe to display/store: secrets are masked."""
        out = {}
        for key, value in self.__dict__.items():
            if key == "odds_api_key":
                value = "configured" if value else "missing"
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            out[key] = value
        return out

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
