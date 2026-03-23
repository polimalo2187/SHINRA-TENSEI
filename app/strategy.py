import pandas as pd
from typing import Optional, Dict, Tuple, List
import ta

from app.oanda_api import get_price_precision, round_price

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

ADX_PERIOD = 14
ATR_PERIOD = 14
BREAKOUT_LOOKBACK = 24
MAX_SCORE = 100.0

SHARED_PROFILE = {
    "name": "shared",
    "adx_min": 17.0,
    "atr_pct_min": 0.0009,
    "atr_pct_max": 0.0065,
    "retest_tol_atr": 0.48,
    "min_body_ratio_breakout": 0.30,
    "min_body_ratio_continuation": 0.22,
}

FREE_PROFILE = {
    "name": "free",
    "adx_min": 15.0,
    "atr_pct_min": 0.0007,
    "atr_pct_max": 0.0080,
    "retest_tol_atr": 0.62,
    "min_body_ratio_breakout": 0.22,
    "min_body_ratio_continuation": 0.16,
}

# Perfiles de riesgo pensados para Forex/Metals.
# Evito apalancamiento hardcodeado porque eso depende del broker y de la cuenta del usuario.
TRADING_PROFILES = {
    "conservador": {
        "label": "Riesgo bajo",
        "sl_atr": 1.80,
        "tp1_atr": 1.60,
        "tp2_atr": 2.80,
    },
    "moderado": {
        "label": "Riesgo medio",
        "sl_atr": 1.45,
        "tp1_atr": 1.45,
        "tp2_atr": 2.35,
    },
    "agresivo": {
        "label": "Riesgo alto",
        "sl_atr": 1.15,
        "tp1_atr": 1.20,
        "tp2_atr": 2.00,
    },
}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ta.trend.ema_indicator(df["close"], EMA_FAST)
    df["ema50"] = ta.trend.ema_indicator(df["close"], EMA_MID)
    df["ema200"] = ta.trend.ema_indicator(df["close"], EMA_SLOW)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], ADX_PERIOD)

    atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["atr"] = atr
    df["atr_pct"] = df["atr"] / df["close"]

    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, 1e-9)
    df["body_ratio"] = df["body"] / df["range"]
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def breakout_level(df: pd.DataFrame, direction: str) -> float:
    ref = df.iloc[-(BREAKOUT_LOOKBACK + 2):-2]
    if direction == "LONG":
        return float(ref["high"].max())
    return float(ref["low"].min())


def _trend_direction(last: pd.Series) -> Optional[str]:
    if float(last["ema20"]) > float(last["ema50"]) > float(last["ema200"]):
        return "LONG"
    if float(last["ema20"]) < float(last["ema50"]) < float(last["ema200"]):
        return "SHORT"
    return None


def _trend_strength_score(last: pd.Series) -> float:
    close = max(float(last["close"]), 1e-9)
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    sep_fast = abs(ema20 - ema50) / close
    sep_slow = abs(ema50 - ema200) / close
    total_sep = sep_fast + sep_slow
    return _clamp((total_sep / 0.012) * 18.0, 0.0, 18.0)


def _adx_score(adx_value: float, adx_min: float) -> float:
    return _clamp(((adx_value - adx_min) / 18.0) * 16.0, 0.0, 16.0)


def _atr_score(atr_pct: float, profile: Dict) -> float:
    lo = float(profile["atr_pct_min"])
    hi = float(profile["atr_pct_max"])
    mid = (lo + hi) / 2.0
    half = max((hi - lo) / 2.0, 1e-9)
    distance = abs(atr_pct - mid) / half
    return _clamp((1.0 - distance) * 12.0, 0.0, 12.0)


def _volume_score(last: pd.Series) -> float:
    vol_ma = float(last.get("vol_ma", 0.0) or 0.0)
    volume = float(last.get("volume", 0.0) or 0.0)
    if vol_ma <= 0:
        return 0.0

    ratio = volume / vol_ma
    if ratio >= 2.0:
        return 10.0
    if ratio >= 1.7:
        return 8.5
    if ratio >= 1.4:
        return 6.5
    if ratio >= 1.2:
        return 4.5
    if ratio >= 1.0:
        return 2.5
    return 0.0


def _confirm_breakout_retest(df: pd.DataFrame, direction: str, profile: Dict) -> Tuple[bool, Dict[str, float]]:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    level = breakout_level(df, direction)
    atr = float(last["atr"])
    tol_atr = float(profile["retest_tol_atr"])

    if atr <= 0:
        return False, {}

    if direction == "LONG":
        breakout_ok = (
            float(prev["close"]) > level
            and float(prev["high"]) > level
            and float(prev["body_ratio"]) >= float(profile["min_body_ratio_breakout"])
        )
        retest_distance = max(0.0, float(last["low"]) - level)
        retest_ok = float(last["low"]) <= level + (atr * tol_atr) and float(last["close"]) >= level
        overshoot = max(0.0, float(prev["close"]) - level)
    else:
        breakout_ok = (
            float(prev["close"]) < level
            and float(prev["low"]) < level
            and float(prev["body_ratio"]) >= float(profile["min_body_ratio_breakout"])
        )
        retest_distance = max(0.0, level - float(last["high"]))
        retest_ok = float(last["high"]) >= level - (atr * tol_atr) and float(last["close"]) <= level
        overshoot = max(0.0, level - float(prev["close"]))

    if not breakout_ok or not retest_ok:
        return False, {}

    overshoot_atr = overshoot / atr if atr > 0 else 0.0
    retest_distance_atr = abs(retest_distance) / atr if atr > 0 else 0.0
    quality = {
        "level": float(level),
        "breakout_body_ratio": float(prev["body_ratio"]),
        "continuation_body_ratio": float(last["body_ratio"]),
        "overshoot_atr": float(overshoot_atr),
        "retest_distance_atr": float(retest_distance_atr),
    }
    return True, quality


def _continuation_ok(last: pd.Series, direction: str, profile: Dict) -> bool:
    if direction == "LONG" and float(last["close"]) <= float(last["open"]):
        return False
    if direction == "SHORT" and float(last["close"]) >= float(last["open"]):
        return False
    return float(last["body_ratio"]) >= float(profile["min_body_ratio_continuation"])


def _breakout_score(quality: Dict[str, float], profile: Dict) -> float:
    body = quality["breakout_body_ratio"]
    min_body = float(profile["min_body_ratio_breakout"])
    body_quality = _clamp((body - min_body) / max(0.40, 1e-9), 0.0, 1.0)
    overshoot_atr = quality["overshoot_atr"]

    if overshoot_atr < 0.08:
        overshoot_quality = overshoot_atr / 0.08
    elif overshoot_atr <= 0.70:
        overshoot_quality = 1.0
    else:
        overshoot_quality = _clamp(1.0 - ((overshoot_atr - 0.70) / 1.20), 0.0, 1.0)

    return _clamp(((body_quality * 0.6) + (overshoot_quality * 0.4)) * 18.0, 0.0, 18.0)


def _retest_score(quality: Dict[str, float], profile: Dict) -> float:
    retest_dist = quality["retest_distance_atr"]
    tol = float(profile["retest_tol_atr"])
    retest_quality = _clamp(1.0 - (retest_dist / max(tol, 1e-9)), 0.0, 1.0)
    return retest_quality * 16.0


def _continuation_score(last: pd.Series, profile: Dict) -> float:
    body = float(last["body_ratio"])
    min_body = float(profile["min_body_ratio_continuation"])
    body_quality = _clamp((body - min_body) / max(0.35, 1e-9), 0.0, 1.0)
    return body_quality * 10.0


def _entry_freshness_score(level: float, close_price: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    extension_atr = abs(close_price - level) / atr
    if extension_atr <= 0.25:
        quality = 1.0
    elif extension_atr <= 0.90:
        quality = 1.0 - ((extension_atr - 0.25) / 0.65)
    else:
        quality = 0.0
    return _clamp(quality * 10.0, 0.0, 10.0)


def _build_trade_profiles(entry_price: float, direction: str, atr: float, symbol: Optional[str] = None) -> Dict[str, Dict]:
    profiles: Dict[str, Dict] = {}
    atr = max(float(atr), 1e-9)

    for name, cfg in TRADING_PROFILES.items():
        sl_distance = atr * float(cfg["sl_atr"])
        tp1_distance = atr * float(cfg["tp1_atr"])
        tp2_distance = atr * float(cfg["tp2_atr"])

        if direction == "LONG":
            stop_loss = entry_price - sl_distance
            tp1 = entry_price + tp1_distance
            tp2 = entry_price + tp2_distance
        else:
            stop_loss = entry_price + sl_distance
            tp1 = entry_price - tp1_distance
            tp2 = entry_price - tp2_distance

        profiles[name] = {
            "stop_loss": round_price(stop_loss, symbol),
            "take_profits": [round_price(tp1, symbol), round_price(tp2, symbol)],
            "profile_label": cfg["label"],
            "sl_distance": round(sl_distance, get_price_precision(symbol=symbol, price=entry_price)),
            "tp1_distance": round(tp1_distance, get_price_precision(symbol=symbol, price=entry_price)),
            "tp2_distance": round(tp2_distance, get_price_precision(symbol=symbol, price=entry_price)),
        }
    return profiles


def _compute_raw_score(df: pd.DataFrame, direction: str, profile: Dict, quality: Dict[str, float]) -> Tuple[float, List[Tuple[str, float]]]:
    last = df.iloc[-1]
    trend_points = _trend_strength_score(last)
    adx_points = _adx_score(float(last["adx"]), float(profile["adx_min"]))
    atr_points = _atr_score(float(last["atr_pct"]), profile)
    breakout_points = _breakout_score(quality, profile)
    retest_points = _retest_score(quality, profile)
    continuation_points = _continuation_score(last, profile)
    volume_points = _volume_score(last)
    entry_points = _entry_freshness_score(quality["level"], float(last["close"]), float(last["atr"]))

    components = [
        ("trend_structure", round(trend_points, 2)),
        ("adx_strength", round(adx_points, 2)),
        ("atr_quality", round(atr_points, 2)),
        ("breakout_quality", round(breakout_points, 2)),
        ("retest_quality", round(retest_points, 2)),
        ("continuation_quality", round(continuation_points, 2)),
        ("volume_quality", round(volume_points, 2)),
        ("entry_freshness", round(entry_points, 2)),
    ]

    raw_score = round(sum(points for _, points in components), 2)
    return round(_clamp(raw_score, 0.0, MAX_SCORE), 2), components


def _evaluate_profile(
    df: pd.DataFrame,
    profile: Dict,
    symbol: Optional[str] = None,
) -> Optional[Tuple[str, float, float, List[Tuple[str, float]], Dict[str, Dict], str, float]]:
    last = df.iloc[-1]
    direction = _trend_direction(last)
    if not direction:
        return None

    adx_value = float(last["adx"])
    if adx_value < float(profile["adx_min"]):
        return None

    atr_pct = float(last["atr_pct"])
    if not (float(profile["atr_pct_min"]) <= atr_pct <= float(profile["atr_pct_max"])):
        return None

    breakout_ok, quality = _confirm_breakout_retest(df, direction, profile)
    if not breakout_ok or not _continuation_ok(last, direction, profile):
        return None

    level = float(quality["level"])
    close_price = float(last["close"])
    atr_value = float(last["atr"])
    entry_price = round_price(level + ((close_price - level) * 0.25), symbol)
    trade_profiles = _build_trade_profiles(entry_price, direction, atr_value, symbol=symbol)
    raw_score, components = _compute_raw_score(df, direction, profile, quality)

    return (
        direction,
        entry_price,
        raw_score,
        components,
        trade_profiles,
        str(profile["name"]),
        atr_pct,
    )


def mtf_strategy(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
) -> Optional[Dict]:
    if len(df_5m) < BREAKOUT_LOOKBACK + 30:
        return None

    df = add_indicators(df_5m)
    if len(df) < BREAKOUT_LOOKBACK + 30:
        return None

    shared_result = _evaluate_profile(df, SHARED_PROFILE, symbol=symbol)
    if shared_result:
        direction, entry_price, raw_score, components, trade_profiles, setup_group, atr_pct = shared_result
        return {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": trade_profiles["conservador"]["stop_loss"],
            "take_profits": list(trade_profiles["conservador"]["take_profits"]),
            "profiles": trade_profiles,
            "score": raw_score,
            "raw_score": raw_score,
            "components": components,
            "timeframes": ["5M"],
            "setup_group": setup_group,
            "atr_pct": atr_pct,
        }

    free_result = _evaluate_profile(df, FREE_PROFILE, symbol=symbol)
    if free_result:
        direction, entry_price, raw_score, components, trade_profiles, setup_group, atr_pct = free_result
        return {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": trade_profiles["conservador"]["stop_loss"],
            "take_profits": list(trade_profiles["conservador"]["take_profits"]),
            "profiles": trade_profiles,
            "score": raw_score,
            "raw_score": raw_score,
            "components": components,
            "timeframes": ["5M"],
            "setup_group": setup_group,
            "atr_pct": atr_pct,
        }

    return None
