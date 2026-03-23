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
    "h1_adx_min": 16.5,
    "atr_pct_min": 0.0008,
    "atr_pct_max": 0.0070,
    "m15_breakout_body_min": 0.22,
    "m15_retest_tol_atr": 0.58,
    "m5_continuation_body_min": 0.28,
    "m5_entry_tol_atr": 0.45,
    "fresh_extension_atr_max": 0.85,
}

FREE_PROFILE = {
    "name": "free",
    "h1_adx_min": 14.5,
    "atr_pct_min": 0.0007,
    "atr_pct_max": 0.0082,
    "m15_breakout_body_min": 0.16,
    "m15_retest_tol_atr": 0.72,
    "m5_continuation_body_min": 0.20,
    "m5_entry_tol_atr": 0.60,
    "fresh_extension_atr_max": 1.00,
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
        structure_ok = float(last["high"]) > float(prev["high"]) and float(last["low"]) >= float(prev["low"])
        slope_ok = float(last["ema20_slope"]) > 0 and float(last["ema50_slope"]) > 0
        location_ok = float(last["close"]) > float(last["ema200"])
    else:
        structure_ok = float(last["low"]) < float(prev["low"]) and float(last["high"]) <= float(prev["high"])
        slope_ok = float(last["ema20_slope"]) < 0 and float(last["ema50_slope"]) < 0
        location_ok = float(last["close"]) < float(last["ema200"])

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
    prev = df_15m.iloc[-2]
    prev2 = df_15m.iloc[-3]
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
            for _, row in pd.DataFrame([prev2, prev]).iterrows()
        )
        retest_ok = float(last["low"]) <= level + tol and float(last["close"]) >= level
        zone_side_ok = float(last["close"]) >= float(last["ema20"]) and float(last["ema20"]) >= float(last["ema50"])
        overshoot = max(0.0, float(prev["close"]) - level, float(prev2["close"]) - level)
        invalidation = min(float(last["low"]), float(prev["low"]), float(prev2["low"]))
        retest_distance = max(0.0, float(last["close"]) - level)
    else:
        breakout_seen = any(
            float(row["close"]) < level
            and float(row["low"]) < level
            and float(row["body_ratio"]) >= min_body
            for _, row in pd.DataFrame([prev2, prev]).iterrows()
        )
        retest_ok = float(last["high"]) >= level - tol and float(last["close"]) <= level
        zone_side_ok = float(last["close"]) <= float(last["ema20"]) and float(last["ema20"]) <= float(last["ema50"])
        overshoot = max(0.0, level - float(prev["close"]), level - float(prev2["close"]))
        invalidation = max(float(last["high"]), float(prev["high"]), float(prev2["high"]))
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
        continuation_ok = float(last["close"]) >= float(prev["close"])
        touch_ok = float(last["low"]) <= anchor_level + (atr * entry_tol)
        invalidation = min(float(last["low"]), float(prev["low"]))
    else:
        side_ok = float(last["close"]) <= anchor_level and float(last["close"]) < float(last["open"])
        continuation_ok = float(last["close"]) <= float(prev["close"])
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
    if expansion_atr < 0.75:
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



def _instrument_bucket(symbol: Optional[str]) -> str:
    s = (symbol or "").upper().replace("-", "_").replace("/", "_")

    majors = {
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
        "USD_CAD", "AUD_USD", "NZD_USD",
    }
    crosses = {
        "EUR_JPY", "GBP_JPY", "EUR_GBP", "AUD_JPY",
        "CAD_JPY", "CHF_JPY", "GBP_CHF", "EUR_AUD",
    }
    metals = {"XAU_USD", "XAG_USD"}

    if s in metals:
        return "metal"
    if s in crosses:
        return "cross"
    if s in majors:
        return "major"
    return "major"



def _bucket_risk_limits(bucket: str) -> Dict[str, float]:
    if bucket == "metal":
        return {
            "min_risk_pct": 0.0012,
            "max_risk_pct": 0.0090,
            "spread_weight": 1.20,
            "min_rr": 1.70,
        }
    if bucket == "cross":
        return {
            "min_risk_pct": 0.0009,
            "max_risk_pct": 0.0060,
            "spread_weight": 1.00,
            "min_rr": 1.60,
        }
    return {
        "min_risk_pct": 0.0006,
        "max_risk_pct": 0.0045,
        "spread_weight": 0.85,
        "min_rr": 1.50,
    }



def _safe_rr(entry_price: float, stop_loss: float, take_profit: float, direction: str) -> float:
    if direction == "LONG":
        risk = max(entry_price - stop_loss, 1e-9)
        reward = max(take_profit - entry_price, 0.0)
    else:
        risk = max(stop_loss - entry_price, 1e-9)
        reward = max(entry_price - take_profit, 0.0)
    return reward / risk



def _nearest_directional_levels(
    values: List[float],
    entry_price: float,
    direction: str,
) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        if not _valid_number(value):
            continue
        v = float(value)
        if direction == "LONG" and v > entry_price:
            cleaned.append(v)
        elif direction == "SHORT" and v < entry_price:
            cleaned.append(v)

    if not cleaned:
        return []

    unique = sorted(set(round(v, 10) for v in cleaned))
    if direction == "LONG":
        return unique
    return list(reversed(unique))



def _derive_structural_targets(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    direction: str,
    entry_price: float,
) -> Tuple[Optional[float], Optional[float]]:
    candidates: List[float] = []

    if direction == "LONG":
        if len(df_15m) >= 24:
            candidates.extend(df_15m.iloc[-24:-1]["high"].tolist())
        if len(df_1h) >= 12:
            candidates.extend(df_1h.iloc[-12:-1]["high"].tolist())
    else:
        if len(df_15m) >= 24:
            candidates.extend(df_15m.iloc[-24:-1]["low"].tolist())
        if len(df_1h) >= 12:
            candidates.extend(df_1h.iloc[-12:-1]["low"].tolist())

    directional = _nearest_directional_levels(candidates, entry_price, direction)
    if not directional:
        return None, None

    tp1 = directional[0]
    tp2 = directional[1] if len(directional) > 1 else None
    return tp1, tp2



def _build_trade_profiles(
    entry_price: float,
    direction: str,
    atr: float,
    invalidation: float,
    symbol: Optional[str] = None,
    structural_tp1: Optional[float] = None,
    structural_tp2: Optional[float] = None,
    spread: Optional[float] = None,
) -> Dict[str, Dict]:
    profiles: Dict[str, Dict] = {}

    atr = max(float(atr), 1e-9)
    entry_price = float(entry_price)
    invalidation = float(invalidation)
    spread = max(float(spread or 0.0), 0.0)

    precision = get_price_precision(symbol=symbol, price=entry_price)
    bucket = _instrument_bucket(symbol)
    limits = _bucket_risk_limits(bucket)

    for name, cfg in TRADING_PROFILES.items():
        sl_buffer_atr = float(cfg["sl_buffer_atr"])
        tp1_r = float(cfg["tp1_r"])
        tp2_r = float(cfg["tp2_r"])

        atr_buffer = atr * sl_buffer_atr
        spread_cushion = spread * limits["spread_weight"]

        if direction == "LONG":
            raw_stop = invalidation - atr_buffer - spread_cushion
            min_stop = entry_price - (atr * 0.30)
            stop_loss = min(raw_stop, min_stop)

            risk = max(entry_price - stop_loss, atr * 0.30)
            risk_pct = risk / max(entry_price, 1e-9)
            if risk_pct < limits["min_risk_pct"] or risk_pct > limits["max_risk_pct"]:
                continue

            fallback_tp1 = entry_price + (risk * tp1_r)
            fallback_tp2 = entry_price + (risk * tp2_r)
            tp1 = max(float(structural_tp1), fallback_tp1) if structural_tp1 else fallback_tp1
            tp2 = max(float(structural_tp2), fallback_tp2) if structural_tp2 else fallback_tp2
        else:
            raw_stop = invalidation + atr_buffer + spread_cushion
            min_stop = entry_price + (atr * 0.30)
            stop_loss = max(raw_stop, min_stop)

            risk = max(stop_loss - entry_price, atr * 0.30)
            risk_pct = risk / max(entry_price, 1e-9)
            if risk_pct < limits["min_risk_pct"] or risk_pct > limits["max_risk_pct"]:
                continue

            fallback_tp1 = entry_price - (risk * tp1_r)
            fallback_tp2 = entry_price - (risk * tp2_r)
            tp1 = min(float(structural_tp1), fallback_tp1) if structural_tp1 else fallback_tp1
            tp2 = min(float(structural_tp2), fallback_tp2) if structural_tp2 else fallback_tp2

        rr1 = _safe_rr(entry_price, stop_loss, tp1, direction)
        rr2 = _safe_rr(entry_price, stop_loss, tp2, direction)
        if rr1 < limits["min_rr"]:
            continue

        profiles[name] = {
            "stop_loss": round_price(stop_loss, symbol),
            "take_profits": [round_price(tp1, symbol), round_price(tp2, symbol)],
            "profile_label": cfg["label"],
            "risk_distance": round(risk, precision),
            "risk_pct": round(risk_pct * 100.0, 4),
            "sl_buffer": round(atr_buffer + spread_cushion, precision),
            "spread_cushion": round(spread_cushion, precision),
            "tp1_r": round(rr1, 2),
            "tp2_r": round(rr2, 2),
            "bucket": bucket,
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
    spread = float(last_5m.get("spread", 0.0) or 0.0)
    structural_tp1, structural_tp2 = _derive_structural_targets(df_15m, df_1h, direction, float(entry_price))
    trade_profiles = _build_trade_profiles(
        entry_price=entry_price,
        direction=direction,
        atr=atr_value,
        invalidation=invalidation,
        symbol=symbol,
        structural_tp1=structural_tp1,
        structural_tp2=structural_tp2,
        spread=spread,
    )
    if not trade_profiles or "conservador" not in trade_profiles:
        return None

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
