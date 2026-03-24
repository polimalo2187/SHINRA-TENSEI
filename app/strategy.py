import math
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ta

from app.oanda_api import get_price_precision, round_price

STRATEGY_NAME = "HIBRIDO_S"

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
ADX_PERIOD = 14
ATR_PERIOD = 14
M15_LOOKBACK = 18
MAX_SCORE = 100.0

# Híbrido S = contexto MTF real + ejecución breakout/retest.
SHARED_PROFILE = {
    "name": "shared",
    "h1_adx_min": 13.5,
    "atr_pct_min": 0.0007,
    "atr_pct_max": 0.0076,
    "m15_breakout_body_min": 0.15,
    "m15_retest_tol_atr": 0.90,
    "m15_breakout_max_bars": 4,
    "m5_continuation_body_min": 0.18,
    "m5_entry_tol_atr": 0.75,
    "m5_min_expansion_atr": 0.45,
    "fresh_extension_atr_max": 1.35,
}

FREE_PROFILE = {
    "name": "free",
    "h1_adx_min": 11.5,
    "atr_pct_min": 0.0006,
    "atr_pct_max": 0.0088,
    "m15_breakout_body_min": 0.12,
    "m15_retest_tol_atr": 1.15,
    "m15_breakout_max_bars": 5,
    "m5_continuation_body_min": 0.14,
    "m5_entry_tol_atr": 0.95,
    "m5_min_expansion_atr": 0.35,
    "fresh_extension_atr_max": 1.60,
}

TRADING_PROFILES = {
    "conservador": {
        "label": "Riesgo bajo",
        "sl_buffer_atr": 0.18,
        "tp1_r": 1.35,
        "tp2_r": 2.20,
    },
    "moderado": {
        "label": "Riesgo medio",
        "sl_buffer_atr": 0.12,
        "tp1_r": 1.55,
        "tp2_r": 2.45,
    },
    "agresivo": {
        "label": "Riesgo alto",
        "sl_buffer_atr": 0.08,
        "tp1_r": 1.75,
        "tp2_r": 2.80,
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _valid_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    out["volume"] = out["volume"].fillna(0.0)
    out = out[(out["high"] >= out["low"]) & (out["close"] > 0)].copy()
    return out.reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_ohlcv(df)
    if len(df) < EMA_SLOW + 5:
        return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df["ema20"] = ta.trend.ema_indicator(df["close"], EMA_FAST)
        df["ema50"] = ta.trend.ema_indicator(df["close"], EMA_MID)
        df["ema200"] = ta.trend.ema_indicator(df["close"], EMA_SLOW)
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], ADX_PERIOD)
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], ATR_PERIOD
        )

    df["atr_pct"] = df["atr"] / df["close"].replace(0, pd.NA)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, 1e-9)
    df["body_ratio"] = df["body"] / df["range"]
    df["vol_ma"] = df["volume"].rolling(20, min_periods=5).mean()
    df["ema20_slope"] = df["ema20"].diff()
    df["ema50_slope"] = df["ema50"].diff()
    df["close_vs_ema20"] = (df["close"] - df["ema20"]) / df["close"].replace(0, pd.NA)
    return df.dropna().reset_index(drop=True)


def _trend_direction(last: pd.Series) -> Optional[str]:
    if (
        float(last["ema20"]) > float(last["ema50"]) > float(last["ema200"])
        and float(last["close"]) >= float(last["ema20"])
    ):
        return "LONG"
    if (
        float(last["ema20"]) < float(last["ema50"]) < float(last["ema200"])
        and float(last["close"]) <= float(last["ema20"])
    ):
        return "SHORT"
    return None


def _reference_level(df: pd.DataFrame, direction: str, lookback: int) -> Optional[float]:
    if len(df) < lookback + 4:
        return None
    ref = df.iloc[-(lookback + 3):-3]
    if ref.empty:
        return None
    if direction == "LONG":
        return float(ref["high"].max())
    return float(ref["low"].min())


def _htf_bias(df_1h: pd.DataFrame, profile: Dict) -> Optional[Tuple[str, Dict[str, float]]]:
    if len(df_1h) < 60:
        return None
    last = df_1h.iloc[-1]
    prev = df_1h.iloc[-2]
    recent = df_1h.iloc[-4:-1]
    direction = _trend_direction(last)
    if not direction:
        return None

    adx_value = float(last["adx"])
    if adx_value < float(profile["h1_adx_min"]):
        return None

    atr_pct = float(last["atr_pct"])
    if not (float(profile["atr_pct_min"]) <= atr_pct <= float(profile["atr_pct_max"])):
        return None

    if direction == "LONG":
        structure_ok = (
            float(last["close"]) >= float(prev["close"]) or
            float(last["high"]) >= float(recent["high"].max())
        )
        slope_ok = float(last["ema20_slope"]) >= 0 and float(last["ema50_slope"]) >= 0
        location_ok = float(last["close"]) >= float(last["ema50"])
    else:
        structure_ok = (
            float(last["close"]) <= float(prev["close"]) or
            float(last["low"]) <= float(recent["low"].min())
        )
        slope_ok = float(last["ema20_slope"]) <= 0 and float(last["ema50_slope"]) <= 0
        location_ok = float(last["close"]) <= float(last["ema50"])

    if not (structure_ok and slope_ok and location_ok):
        return None

    quality = {
        "h1_adx": adx_value,
        "h1_atr_pct": atr_pct,
        "h1_trend_sep": (
            abs(float(last["ema20"]) - float(last["ema50"]))
            + abs(float(last["ema50"]) - float(last["ema200"]))
        ) / max(float(last["close"]), 1e-9),
    }
    return direction, quality


def _confirm_m15_breakout_retest(
    df_15m: pd.DataFrame,
    direction: str,
    profile: Dict,
) -> Optional[Dict[str, float]]:
    if len(df_15m) < M15_LOOKBACK + 8:
        return None

    last = df_15m.iloc[-1]
    breakout_bars = int(profile.get("m15_breakout_max_bars", 3))
    candidates = df_15m.iloc[-(breakout_bars + 1):-1]
    if candidates.empty:
        return None
    prev = candidates.iloc[-1]
    prev2 = candidates.iloc[-2] if len(candidates) >= 2 else candidates.iloc[-1]
    level = _reference_level(df_15m, direction, M15_LOOKBACK)
    atr = float(last["atr"])
    if not _valid_number(level) or atr <= 0:
        return None

    tol = atr * float(profile["m15_retest_tol_atr"])
    min_body = float(profile["m15_breakout_body_min"])

    if direction == "LONG":
        breakout_seen = any(
            float(row["close"]) > level
            and float(row["high"]) > level
            and float(row["body_ratio"]) >= min_body
            for _, row in candidates.iterrows()
        )
        retest_ok = float(last["low"]) <= level + tol and float(last["close"]) >= level
        zone_side_ok = float(last["close"]) >= float(last["ema20"]) and float(last["ema20"]) >= float(last["ema50"])
        overshoot = max(0.0, max(float(x) - level for x in candidates["close"]))
        invalidation = min(float(last["low"]), float(candidates["low"].min()))
        retest_distance = max(0.0, float(last["close"]) - level)
    else:
        breakout_seen = any(
            float(row["close"]) < level
            and float(row["low"]) < level
            and float(row["body_ratio"]) >= min_body
            for _, row in candidates.iterrows()
        )
        retest_ok = float(last["high"]) >= level - tol and float(last["close"]) <= level
        zone_side_ok = float(last["close"]) <= float(last["ema20"]) and float(last["ema20"]) <= float(last["ema50"])
        overshoot = max(0.0, max(level - float(x) for x in candidates["close"]))
        invalidation = max(float(last["high"]), float(candidates["high"].max()))
        retest_distance = max(0.0, level - float(last["close"]))

    if not (breakout_seen and retest_ok and zone_side_ok):
        return None

    return {
        "m15_level": float(level),
        "m15_atr": atr,
        "m15_overshoot_atr": overshoot / atr,
        "m15_retest_distance_atr": retest_distance / atr,
        "m15_invalidation": invalidation,
    }


def _confirm_m5_trigger(
    df_5m: pd.DataFrame,
    direction: str,
    anchor_level: float,
    profile: Dict,
) -> Optional[Dict[str, float]]:
    if len(df_5m) < 30:
        return None

    last = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]
    atr = float(last["atr"])
    if atr <= 0:
        return None

    body_min = float(profile["m5_continuation_body_min"])
    entry_tol = float(profile["m5_entry_tol_atr"])
    fresh_max = float(profile["fresh_extension_atr_max"])

    if direction == "LONG":
        side_ok = float(last["close"]) >= anchor_level and float(last["close"]) > float(last["open"])
        continuation_ok = float(last["close"]) >= (float(prev["close"]) - atr * 0.15)
        touch_ok = float(last["low"]) <= anchor_level + (atr * entry_tol)
        invalidation = min(float(last["low"]), float(prev["low"]))
    else:
        side_ok = float(last["close"]) <= anchor_level and float(last["close"]) < float(last["open"])
        continuation_ok = float(last["close"]) <= (float(prev["close"]) + atr * 0.15)
        touch_ok = float(last["high"]) >= anchor_level - (atr * entry_tol)
        invalidation = max(float(last["high"]), float(prev["high"]))

    extension_atr = abs(float(last["close"]) - anchor_level) / atr
    expansion_atr = float(last["range"]) / atr
    if not (side_ok and continuation_ok and touch_ok):
        return None
    if float(last["body_ratio"]) < body_min:
        return None
    if extension_atr > fresh_max:
        return None
    if expansion_atr < float(profile.get("m5_min_expansion_atr", 0.45)):
        return None

    return {
        "m5_atr": atr,
        "m5_extension_atr": extension_atr,
        "m5_expansion_atr": expansion_atr,
        "m5_body_ratio": float(last["body_ratio"]),
        "m5_invalidation": invalidation,
    }


def _safe_volume_score(last: pd.Series) -> float:
    vol_ma = float(last.get("vol_ma", 0.0) or 0.0)
    volume = float(last.get("volume", 0.0) or 0.0)
    if vol_ma <= 0:
        return 0.0
    ratio = volume / vol_ma
    if ratio >= 1.8:
        return 6.0
    if ratio >= 1.5:
        return 4.5
    if ratio >= 1.2:
        return 3.0
    if ratio >= 1.0:
        return 1.5
    return 0.0


def _score_h1_bias(quality: Dict[str, float], profile: Dict) -> Tuple[float, List[Tuple[str, float]]]:
    trend_sep = quality["h1_trend_sep"]
    trend_points = _clamp((trend_sep / 0.010) * 18.0, 0.0, 18.0)
    adx_points = _clamp(((quality["h1_adx"] - float(profile["h1_adx_min"])) / 16.0) * 12.0, 0.0, 12.0)
    atr_points = _clamp(
        (1.0 - abs(quality["h1_atr_pct"] - ((float(profile["atr_pct_min"]) + float(profile["atr_pct_max"])) / 2.0))
         / max((float(profile["atr_pct_max"]) - float(profile["atr_pct_min"])) / 2.0, 1e-9)) * 10.0,
        0.0,
        10.0,
    )
    return trend_points + adx_points + atr_points, [
        ("h1_trend_bias", round(trend_points, 2)),
        ("h1_adx_strength", round(adx_points, 2)),
        ("h1_volatility_fit", round(atr_points, 2)),
    ]


def _score_m15_context(quality: Dict[str, float], profile: Dict) -> Tuple[float, List[Tuple[str, float]]]:
    overshoot = quality["m15_overshoot_atr"]
    if overshoot < 0.05:
        breakout_quality = overshoot / 0.05
    elif overshoot <= 0.90:
        breakout_quality = 1.0
    else:
        breakout_quality = _clamp(1.0 - ((overshoot - 0.90) / 1.20), 0.0, 1.0)

    retest_quality = _clamp(
        1.0 - (quality["m15_retest_distance_atr"] / max(float(profile["m15_retest_tol_atr"]), 1e-9)),
        0.0,
        1.0,
    )

    breakout_points = breakout_quality * 16.0
    retest_points = retest_quality * 12.0
    return breakout_points + retest_points, [
        ("m15_breakout_quality", round(breakout_points, 2)),
        ("m15_retest_quality", round(retest_points, 2)),
    ]


def _score_m5_trigger(last_5m: pd.Series, quality: Dict[str, float]) -> Tuple[float, List[Tuple[str, float]]]:
    body_points = _clamp((quality["m5_body_ratio"] / 0.75) * 12.0, 0.0, 12.0)
    expansion_points = _clamp(((quality["m5_expansion_atr"] - 0.75) / 0.90) * 12.0, 0.0, 12.0)
    freshness_points = _clamp((1.0 - (quality["m5_extension_atr"] / 1.0)) * 10.0, 0.0, 10.0)
    volume_points = _safe_volume_score(last_5m)
    return body_points + expansion_points + freshness_points + volume_points, [
        ("m5_body_quality", round(body_points, 2)),
        ("m5_expansion_quality", round(expansion_points, 2)),
        ("entry_freshness", round(freshness_points, 2)),
        ("activity_quality", round(volume_points, 2)),
    ]


def _build_trade_profiles(
    entry_price: float,
    direction: str,
    atr: float,
    invalidation: float,
    symbol: Optional[str] = None,
) -> Dict[str, Dict]:
    profiles: Dict[str, Dict] = {}
    atr = max(float(atr), 1e-9)
    precision = get_price_precision(symbol=symbol, price=entry_price)

    for name, cfg in TRADING_PROFILES.items():
        buffer_dist = atr * float(cfg["sl_buffer_atr"])
        if direction == "LONG":
            stop_loss = min(invalidation - buffer_dist, entry_price - (atr * 0.35))
            risk = max(entry_price - stop_loss, atr * 0.35)
            tp1 = entry_price + (risk * float(cfg["tp1_r"]))
            tp2 = entry_price + (risk * float(cfg["tp2_r"]))
        else:
            stop_loss = max(invalidation + buffer_dist, entry_price + (atr * 0.35))
            risk = max(stop_loss - entry_price, atr * 0.35)
            tp1 = entry_price - (risk * float(cfg["tp1_r"]))
            tp2 = entry_price - (risk * float(cfg["tp2_r"]))

        profiles[name] = {
            "stop_loss": round_price(stop_loss, symbol),
            "take_profits": [round_price(tp1, symbol), round_price(tp2, symbol)],
            "profile_label": cfg["label"],
            "risk_distance": round(risk, precision),
            "sl_buffer": round(buffer_dist, precision),
            "tp1_r": float(cfg["tp1_r"]),
            "tp2_r": float(cfg["tp2_r"]),
        }
    return profiles


def _evaluate_profile(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    profile: Dict,
    symbol: Optional[str] = None,
) -> Optional[Tuple[str, float, float, List[Tuple[str, float]], Dict[str, Dict], str, float]]:
    bias_result = _htf_bias(df_1h, profile)
    if not bias_result:
        return None
    direction, h1_quality = bias_result

    m15_quality = _confirm_m15_breakout_retest(df_15m, direction, profile)
    if not m15_quality:
        return None

    m5_quality = _confirm_m5_trigger(df_5m, direction, float(m15_quality["m15_level"]), profile)
    if not m5_quality:
        return None

    last_5m = df_5m.iloc[-1]
    entry_price = float(m15_quality["m15_level"]) + ((float(last_5m["close"]) - float(m15_quality["m15_level"])) * 0.25)
    entry_price = round_price(entry_price, symbol)

    invalidation = (
        min(float(m15_quality["m15_invalidation"]), float(m5_quality["m5_invalidation"]))
        if direction == "LONG"
        else max(float(m15_quality["m15_invalidation"]), float(m5_quality["m5_invalidation"]))
    )
    atr_value = max(float(m5_quality["m5_atr"]), float(m15_quality["m15_atr"]) * 0.65)
    trade_profiles = _build_trade_profiles(entry_price, direction, atr_value, invalidation, symbol=symbol)

    h1_points, h1_components = _score_h1_bias(h1_quality, profile)
    m15_points, m15_components = _score_m15_context(m15_quality, profile)
    m5_points, m5_components = _score_m5_trigger(last_5m, m5_quality)

    components = h1_components + m15_components + m5_components + [
        ("strategy_hybrid_bonus", 6.0),
    ]
    raw_score = round(_clamp(h1_points + m15_points + m5_points + 6.0, 0.0, MAX_SCORE), 2)

    return (
        direction,
        entry_price,
        raw_score,
        components,
        trade_profiles,
        str(profile["name"]),
        float(df_5m.iloc[-1]["atr_pct"]),
    )


def mtf_strategy(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
) -> Optional[Dict]:
    df_1h = add_indicators(df_1h)
    df_15m = add_indicators(df_15m)
    df_5m = add_indicators(df_5m)

    if len(df_1h) < 50 or len(df_15m) < M15_LOOKBACK + 6 or len(df_5m) < 30:
        return None

    for profile in (SHARED_PROFILE, FREE_PROFILE):
        result = _evaluate_profile(df_1h, df_15m, df_5m, profile, symbol=symbol)
        if result:
            direction, entry_price, raw_score, components, trade_profiles, setup_group, atr_pct = result
            return {
                "strategy_name": STRATEGY_NAME,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": trade_profiles["conservador"]["stop_loss"],
                "take_profits": list(trade_profiles["conservador"]["take_profits"]),
                "profiles": trade_profiles,
                "score": raw_score,
                "raw_score": raw_score,
                "components": components,
                "timeframes": ["1H", "15M", "5M"],
                "setup_group": setup_group,
                "atr_pct": atr_pct,
            }
    return None
