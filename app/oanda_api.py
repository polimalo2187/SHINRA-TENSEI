from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import pytz
import requests

logger = logging.getLogger(__name__)

DEFAULT_INSTRUMENTS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "USD_CAD",
    "AUD_USD",
    "NZD_USD",
    "EUR_GBP",
    "EUR_JPY",
    "GBP_JPY",
    "XAU_USD",
    "XAG_USD",
]

_GRANULARITY_CACHE_TTLS = {
    "M1": 15,
    "M5": 25,
    "M15": 45,
    "H1": 90,
    "H4": 180,
    "D": 600,
}

_CACHE: Dict[str, Tuple[float, Any]] = {}
_SESSION = requests.Session()
_NY_TZ = pytz.timezone("America/New_York")


def _env_mode() -> str:
    mode = str(os.getenv("OANDA_ENV", "practice")).strip().lower()
    if mode not in {"practice", "live"}:
        return "practice"
    return mode


def _api_base_url() -> str:
    custom = os.getenv("OANDA_API_BASE_URL", "").strip()
    if custom:
        return custom.rstrip("/")
    if _env_mode() == "live":
        return "https://api-fxtrade.oanda.com"
    return "https://api-fxpractice.oanda.com"


def _stream_base_url() -> str:
    custom = os.getenv("OANDA_STREAM_BASE_URL", "").strip()
    if custom:
        return custom.rstrip("/")
    if _env_mode() == "live":
        return "https://stream-fxtrade.oanda.com"
    return "https://stream-fxpractice.oanda.com"


def _token() -> str:
    return os.getenv("OANDA_API_TOKEN", "").strip()


def _account_id() -> str:
    return os.getenv("OANDA_ACCOUNT_ID", "").strip()


def _headers() -> Dict[str, str]:
    token = _token()
    headers = {
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _cache_get(key: str) -> Any | None:
    item = _CACHE.get(key)
    if not item:
        return None
    expires_at, value = item
    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    _CACHE[key] = (time.time() + max(1, ttl_seconds), value)


def _request_json(path: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Any:
    token = _token()
    if not token:
        logger.warning("OANDA_API_TOKEN no configurado; no se puede consultar mercado Forex")
        return None

    url = f"{_api_base_url()}{path}"
    try:
        response = _SESSION.get(url, headers=_headers(), params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("Error OANDA GET %s params=%s: %s", path, params, exc)
        return None


def is_market_open(now: Optional[datetime] = None) -> bool:
    now = now or datetime.utcnow().replace(tzinfo=pytz.UTC)
    ny = now.astimezone(_NY_TZ)
    weekday = ny.weekday()  # 0=lunes ... 6=domingo
    hhmm = ny.hour * 60 + ny.minute

    if weekday == 5:
        return False
    if weekday == 6 and hhmm < (17 * 60):
        return False
    if weekday == 4 and hhmm >= (17 * 60):
        return False
    return True


def normalize_instrument(symbol: str) -> Optional[str]:
    if not symbol:
        return None

    raw = str(symbol).upper().strip()
    raw = raw.replace(" ", "")
    raw = raw.replace("-", "_")
    raw = raw.replace("/", "_")

    if "_" in raw:
        parts = [p for p in raw.split("_") if p]
        if len(parts) == 2 and all(2 <= len(p) <= 6 for p in parts):
            return f"{parts[0]}_{parts[1]}"
        return None

    raw = "".join(ch for ch in raw if ch.isalnum())
    if len(raw) == 6:
        return f"{raw[:3]}_{raw[3:]}"

    if len(raw) == 7 and raw.startswith("XAU"):
        return f"{raw[:3]}_{raw[3:]}"

    if len(raw) == 7 and raw.startswith("XAG"):
        return f"{raw[:3]}_{raw[3:]}"

    if len(raw) == 8 and raw[:4] in {"US30", "NAS1", "SPX5"}:
        return raw

    return None


def normalize_many_instruments(raw: str) -> List[str]:
    if not raw:
        return []

    tokens: List[str] = []
    for chunk in raw.replace("\n", ",").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens.extend(part for part in chunk.split() if part.strip())

    out: List[str] = []
    seen: Set[str] = set()
    for token in tokens:
        instrument = normalize_instrument(token)
        if instrument and instrument not in seen:
            out.append(instrument)
            seen.add(instrument)
    return out


def _env_instruments(var_name: str, fallback: Iterable[str]) -> List[str]:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return list(fallback)

    parsed = normalize_many_instruments(raw)
    return parsed or list(fallback)


def get_configured_instruments() -> List[str]:
    return _env_instruments("OANDA_INSTRUMENTS", DEFAULT_INSTRUMENTS)


def get_scan_instruments() -> List[str]:
    return _env_instruments("OANDA_SCAN_INSTRUMENTS", get_configured_instruments())


def get_valid_instruments() -> Set[str]:
    return set(get_configured_instruments())


def get_price_precision(symbol: Optional[str] = None, price: Optional[float] = None) -> int:
    symbol = str(symbol or "").upper()
    if symbol.endswith("_JPY"):
        return 3
    if symbol.startswith("XAU_"):
        return 2
    if symbol.startswith("XAG_"):
        return 3
    if price is not None:
        p = abs(float(price))
        if p >= 1000:
            return 2
        if p >= 100:
            return 3
    return 5


def round_price(value: float, symbol: Optional[str] = None) -> float:
    precision = get_price_precision(symbol=symbol, price=value)
    return round(float(value), precision)


def format_price(value: float, symbol: Optional[str] = None) -> str:
    precision = get_price_precision(symbol=symbol, price=value)
    return f"{float(value):,.{precision}f}"


def get_candles_df(
    instrument: str,
    granularity: str,
    *,
    count: int = 250,
    price: str = "M",
    include_incomplete: bool = False,
) -> pd.DataFrame:
    instrument = normalize_instrument(instrument) or str(instrument).upper().strip()
    granularity = str(granularity).upper().strip()
    cache_key = f"candles:{instrument}:{granularity}:{count}:{price}:{int(include_incomplete)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    data = _request_json(
        f"/v3/instruments/{instrument}/candles",
        params={
            "granularity": granularity,
            "count": max(10, min(int(count), 5000)),
            "price": price,
            "alignmentTimezone": "America/New_York",
        },
    )
    if not data or "candles" not in data:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "complete"])

    rows: List[Dict[str, Any]] = []
    for candle in data.get("candles", []):
        complete = bool(candle.get("complete", False))
        if not include_incomplete and not complete:
            continue
        mid = candle.get("mid") or candle.get("bid") or candle.get("ask") or {}
        try:
            rows.append(
                {
                    "time": candle.get("time"),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(candle.get("volume", 0.0) or 0.0),
                    "complete": complete,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)

    _cache_set(cache_key, df.copy(), _GRANULARITY_CACHE_TTLS.get(granularity, 60))
    return df


def get_latest_price(instrument: str) -> float:
    df = get_candles_df(instrument, "M1", count=2)
    if df.empty:
        raise RuntimeError(f"No pude obtener precio para {instrument}")
    return float(df.iloc[-1]["close"])


def get_change_pct(instrument: str, granularity: str, lookback_candles: int = 1) -> Optional[float]:
    df = get_candles_df(instrument, granularity, count=max(lookback_candles + 2, 3))
    if len(df) < lookback_candles + 1:
        return None
    prev = float(df.iloc[-(lookback_candles + 1)]["close"])
    last = float(df.iloc[-1]["close"])
    if prev <= 0:
        return None
    return ((last - prev) / prev) * 100.0


def _safe_change(df: pd.DataFrame, lookback: int) -> float:
    if len(df) < lookback + 1:
        return 0.0
    prev = float(df.iloc[-(lookback + 1)]["close"])
    last = float(df.iloc[-1]["close"])
    if prev <= 0:
        return 0.0
    return ((last - prev) / prev) * 100.0


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 2:
        return 0.0

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    close_last = float(close.iloc[-1])
    if close_last <= 0:
        return 0.0
    return float(atr) / close_last


def get_instrument_snapshot(instrument: str) -> Dict[str, Any]:
    instrument = normalize_instrument(instrument) or instrument
    h1 = get_candles_df(instrument, "H1", count=30)
    h4 = get_candles_df(instrument, "H4", count=5)
    m5 = get_candles_df(instrument, "M5", count=120)

    if h1.empty or h4.empty or m5.empty:
        return {}

    price = float(m5.iloc[-1]["close"])
    chg_24h = _safe_change(h1, 24)
    chg_1h = _safe_change(h1, 1)
    chg_4h = _safe_change(h4, 1)
    vol_24h = float(h1.tail(24)["volume"].sum())
    atr_pct_1h = _atr_pct(h1)

    if chg_4h > 0.15 and chg_1h >= -0.05:
        trend = "Alcista"
    elif chg_4h < -0.15 and chg_1h <= 0.05:
        trend = "Bajista"
    else:
        trend = "Mixta"

    abs_mix = abs(chg_1h) + (abs(chg_24h) * 0.35) + (atr_pct_1h * 150)
    if abs_mix >= 2.6:
        momentum = "Fuerte"
    elif abs_mix >= 1.2:
        momentum = "Medio"
    else:
        momentum = "Débil"

    return {
        "symbol": instrument,
        "price": price,
        "chg_24h": chg_24h,
        "chg_1h": chg_1h,
        "chg_4h": chg_4h,
        "volume": vol_24h,
        "atr_pct": atr_pct_1h,
        "trend": trend,
        "momentum": momentum,
    }


def get_market_rows(limit_instruments: Optional[int] = None) -> List[Dict[str, Any]]:
    instruments = get_configured_instruments()
    if limit_instruments is not None:
        instruments = instruments[: max(1, int(limit_instruments))]

    rows: List[Dict[str, Any]] = []
    for instrument in instruments:
        snap = get_instrument_snapshot(instrument)
        if not snap:
            continue
        rows.append(
            {
                "symbol": instrument,
                "lastPrice": snap["price"],
                "priceChangePercent": snap["chg_24h"],
                "change_1h": snap["chg_1h"],
                "change_4h": snap["chg_4h"],
                "quoteVolume": snap["volume"],
                "atr_pct": snap["atr_pct"],
                "trend": snap["trend"],
                "momentum": snap["momentum"],
            }
        )
    return rows


def get_top_movers(limit: int = 10, *, kind: str = "gainers") -> List[Dict[str, Any]]:
    rows = get_market_rows()
    if kind == "losers":
        rows.sort(key=lambda x: float(x.get("priceChangePercent", 0.0)))
    elif kind == "absolute":
        rows.sort(key=lambda x: abs(float(x.get("priceChangePercent", 0.0))), reverse=True)
    else:
        rows.sort(key=lambda x: float(x.get("priceChangePercent", 0.0)), reverse=True)
    return rows[: max(1, int(limit))]


def get_radar_opportunities(limit: int = 8) -> List[Dict[str, Any]]:
    rows = get_market_rows()
    if not rows:
        return []

    abs_changes = [abs(float(r.get("priceChangePercent", 0.0))) for r in rows]
    volumes = [float(r.get("quoteVolume", 0.0)) for r in rows]
    atrs = [float(r.get("atr_pct", 0.0)) for r in rows]
    changes_1h = [abs(float(r.get("change_1h", 0.0))) for r in rows]

    def _rank(values: List[float]) -> Dict[int, float]:
        sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        denom = max(1, len(values) - 1)
        for rank, idx in enumerate(sorted_idx):
            out[idx] = rank / denom
        return {i: out[i] for i in range(len(values))}

    r_abs = _rank(abs_changes)
    r_vol = _rank(volumes)
    r_atr = _rank(atrs)
    r_1h = _rank(changes_1h)

    enriched: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score = int(round(100 * ((0.35 * r_abs[idx]) + (0.25 * r_vol[idx]) + (0.20 * r_atr[idx]) + (0.20 * r_1h[idx]))))
        change_pct = float(row.get("priceChangePercent", 0.0))
        enriched.append(
            {
                **row,
                "score": max(1, min(score, 100)),
                "direction": "LONG" if change_pct >= 0 else "SHORT",
                "change_pct": change_pct,
                "quote_volume": float(row.get("quoteVolume", 0.0)),
                "trades": float(row.get("quoteVolume", 0.0)),
            }
        )

    enriched.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),
            abs(float(x.get("change_pct", 0.0))),
            float(x.get("quote_volume", 0.0)),
        ),
        reverse=True,
    )
    return enriched[: max(1, int(limit))]


def get_watchlist_snapshot(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in list(symbols)[:10]:
        snap = get_instrument_snapshot(symbol)
        if snap:
            out[symbol] = snap
    return out


def get_latest_completed_candles_between(
    instrument: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    granularity: str = "M1",
) -> List[Dict[str, Any]]:
    instrument = normalize_instrument(instrument) or instrument
    params = {
        "granularity": granularity,
        "price": "M",
        "from": start_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "to": end_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "alignmentTimezone": "America/New_York",
    }
    data = _request_json(f"/v3/instruments/{instrument}/candles", params=params)
    if not data or "candles" not in data:
        return []

    rows: List[Dict[str, Any]] = []
    for candle in data.get("candles", []):
        if not candle.get("complete"):
            continue
        mid = candle.get("mid") or {}
        try:
            rows.append(
                {
                    "time": candle.get("time"),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(candle.get("volume", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    return rows


def get_pricing_metadata() -> Dict[str, str]:
    return {
        "mode": _env_mode(),
        "api_base_url": _api_base_url(),
        "stream_base_url": _stream_base_url(),
        "account_id": _account_id(),
    }
