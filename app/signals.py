# app/signals.py

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytz

from app.models import new_signal
from app.plans import PLAN_FREE, PLAN_PREMIUM
from app.config import is_admin
from app.database import (
    signal_results_collection,
    signals_collection,
    user_signals_collection,
    users_collection,
)
from app.oanda_api import (
    format_price,
    get_latest_completed_candles_between,
    get_latest_price,
    round_price,
)

logger = logging.getLogger(__name__)

MAX_SIGNALS_PER_QUERY = int(os.getenv("MAX_SIGNALS_PER_QUERY", "10"))
USER_TIMEZONE = os.getenv("USER_TIMEZONE", "America/Havana")

RISK_PROFILE_LABELS = {
    "conservador": "Riesgo bajo",
    "moderado": "Riesgo medio",
    "agresivo": "Riesgo alto",
}

TIMEFRAME_TO_MINUTES = {
    "5M": 5,
    "15M": 15,
    "1H": 60,
}

DEDUP_MINUTES = int(os.getenv("DEDUP_MINUTES", "10"))
TELEGRAM_SIGNAL_COOLDOWN_MINUTES = 15
MIN_SIGNAL_VALIDITY_MINUTES = int(os.getenv("MIN_SIGNAL_VALIDITY_MINUTES", "15"))
MAX_SIGNAL_VALIDITY_MINUTES = int(os.getenv("MAX_SIGNAL_VALIDITY_MINUTES", "45"))


def _base_validity_by_plan(visibility: str) -> int:
    plan = str(visibility or "").lower()
    if plan == "premium":
        return 25
    if plan == "plus":
        return 18
    return 12


def calculate_signal_validity(
    timeframes: List[str],
    *,
    visibility: str = "",
    score: Optional[float] = None,
    entry_price: Optional[float] = None,
    current_price: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> int:
    validity = float(_base_validity_by_plan(visibility))

    minutes = [TIMEFRAME_TO_MINUTES.get(tf.upper(), 0) for tf in timeframes]
    tf_hint = max(minutes) if minutes else 5
    if tf_hint > 5:
        validity += min(10.0, (tf_hint - 5) * 0.15)

    if entry_price and current_price and float(entry_price) > 0 and float(current_price) > 0:
        distance_pct = abs(float(current_price) - float(entry_price)) / float(current_price)
        if distance_pct >= 0.006:
            validity += 8
        elif distance_pct >= 0.004:
            validity += 6
        elif distance_pct >= 0.0025:
            validity += 4
        elif distance_pct >= 0.0015:
            validity += 2

    if score is not None:
        try:
            score_val = float(score)
            if score_val >= 95:
                validity += 6
            elif score_val >= 90:
                validity += 5
            elif score_val >= 82:
                validity += 3
            elif score_val >= 76:
                validity += 2
        except Exception:
            pass

    if atr_pct is not None:
        try:
            atr_pct_val = float(atr_pct)
            if atr_pct_val >= 0.010:
                validity -= 4
            elif atr_pct_val >= 0.008:
                validity -= 3
            elif atr_pct_val <= 0.003:
                validity += 3
            elif atr_pct_val <= 0.004:
                validity += 2
        except Exception:
            pass

    return max(MIN_SIGNAL_VALIDITY_MINUTES, min(MAX_SIGNAL_VALIDITY_MINUTES, int(round(validity))))


def calculate_entry_zone(entry: float, pct: Optional[float] = None, atr_pct: Optional[float] = None, symbol: Optional[str] = None):
    if pct is None:
        dyn_pct = 0.0007
        if atr_pct is not None:
            dyn_pct = max(0.0003, min(0.0012, float(atr_pct) * 0.35))
        pct = dyn_pct

    low = round_price(entry * (1 - pct), symbol)
    high = round_price(entry * (1 + pct), symbol)
    return low, high


def get_current_price(symbol: str) -> float:
    return float(get_latest_price(symbol))


def estimate_minutes_to_entry(symbol: str, entry_zone: Dict[str, float], timeframes: List[str]) -> Dict[str, int]:
    try:
        current_price = get_current_price(symbol)
        zone_mid = (entry_zone["low"] + entry_zone["high"]) / 2

        if entry_zone["low"] <= current_price <= entry_zone["high"]:
            return {"min": 1, "max": 5}

        distance_pct = abs(current_price - zone_mid) / max(current_price, 1e-9)
        if "5M" in timeframes:
            speed = 0.0012
            base_tf = 5
        elif "15M" in timeframes:
            speed = 0.0009
            base_tf = 15
        else:
            speed = 0.0007
            base_tf = calculate_signal_validity(timeframes)

        candles_needed = max(1, distance_pct / speed)
        minutes_estimated = candles_needed * base_tf
        return {
            "min": max(1, int(minutes_estimated * 0.6)),
            "max": int(minutes_estimated * 1.4),
        }
    except Exception as exc:
        logger.warning("Fallback estimate_minutes_to_entry: %s", exc)
        base = calculate_signal_validity(timeframes)
        return {"min": max(1, int(base * 0.5)), "max": int(base * 1.5)}


def recent_duplicate_exists(symbol: str, direction: str, visibility: str) -> bool:
    since = datetime.utcnow() - timedelta(minutes=DEDUP_MINUTES)
    return signals_collection().find_one(
        {
            "symbol": symbol,
            "direction": direction,
            "visibility": visibility,
            "created_at": {"$gte": since},
        }
    ) is not None


def telegram_signal_blocked(symbol: Optional[str] = None) -> bool:
    now = datetime.utcnow()
    query = {"telegram_valid_until": {"$gt": now}}
    if symbol:
        query["symbol"] = symbol
    return signals_collection().find_one(query, sort=[("telegram_valid_until", -1)]) is not None


def generate_user_signal_for_plan(base_signal: Dict):
    visibility = base_signal.get("visibility", PLAN_FREE)
    now = datetime.utcnow()

    for user in users_collection().find({}):
        user_id = user.get("user_id")
        user_plan = user.get("plan", PLAN_FREE)
        plan_end = user.get("plan_end")
        admin = is_admin(user_id)

        if plan_end and plan_end < now:
            continue

        if admin or user_plan == visibility:
            existing = user_signals_collection().find_one(
                {
                    "user_id": user_id,
                    "symbol": base_signal["symbol"],
                    "telegram_valid_until": {"$gt": now},
                }
            )
            if existing:
                continue
            generate_user_signal(base_signal, user_id)


def create_base_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profits: List[float],
    timeframes: List[str],
    visibility: str,
    score: Optional[float] = None,
    components: Optional[List[str]] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    atr_pct: Optional[float] = None,
) -> Dict:
    if telegram_signal_blocked(symbol):
        logger.info("⏳ Bloqueo activo para %s, no se crea nueva señal", symbol)
        return {}

    zone_low, zone_high = calculate_entry_zone(entry_price, atr_pct=atr_pct, symbol=symbol)
    estimated_minutes = estimate_minutes_to_entry(symbol, {"low": zone_low, "high": zone_high}, timeframes)

    try:
        current_price = get_current_price(symbol)
    except Exception as exc:
        logger.warning("Fallback current_price en create_base_signal: %s", exc)
        current_price = entry_price

    signal = new_signal(
        symbol=symbol,
        direction=direction,
        entry_price=round_price(entry_price, symbol),
        stop_loss=round_price(stop_loss, symbol),
        take_profits=[round_price(tp, symbol) for tp in take_profits],
        timeframes=timeframes,
        visibility=visibility,
        leverage=RISK_PROFILE_LABELS,
        components=components,
        score=score,
    )

    if profiles:
        signal["profiles"] = profiles
    signal["risk_profiles"] = RISK_PROFILE_LABELS

    now = datetime.utcnow()
    validity_minutes = calculate_signal_validity(
        timeframes,
        visibility=visibility,
        score=score,
        entry_price=entry_price,
        current_price=current_price,
        atr_pct=atr_pct,
    )
    valid_until = now + timedelta(minutes=validity_minutes)
    telegram_valid_until = now + timedelta(minutes=TELEGRAM_SIGNAL_COOLDOWN_MINUTES)

    inserted_id = signals_collection().insert_one(signal).inserted_id
    signals_collection().update_one(
        {"_id": inserted_id},
        {
            "$set": {
                "created_at": now,
                "valid_until": valid_until,
                "telegram_valid_until": telegram_valid_until,
                "entry_zone": {"low": zone_low, "high": zone_high},
                "estimated_entry_minutes": estimated_minutes,
                "profiles": profiles if profiles else signal.get("profiles"),
                "risk_profiles": RISK_PROFILE_LABELS,
                "validity_minutes": validity_minutes,
                "signal_market_price": round_price(current_price, symbol),
                "signal_atr_pct": atr_pct,
                "evaluated": False,
            }
        },
    )

    signal.update(
        {
            "evaluated": False,
            "_id": inserted_id,
            "created_at": now,
            "valid_until": valid_until,
            "telegram_valid_until": telegram_valid_until,
            "entry_zone": {"low": zone_low, "high": zone_high},
            "estimated_entry_minutes": estimated_minutes,
            "validity_minutes": validity_minutes,
            "signal_market_price": round_price(current_price, symbol),
            "signal_atr_pct": atr_pct,
            "risk_profiles": RISK_PROFILE_LABELS,
        }
    )

    generate_user_signal_for_plan(signal)
    return signal


def _fallback_profiles(direction: str, entry: float, symbol: Optional[str] = None) -> Dict[str, Dict]:
    if direction == "LONG":
        return {
            "conservador": {
                "stop_loss": round_price(entry * 0.9975, symbol),
                "take_profits": [round_price(entry * 1.0020, symbol), round_price(entry * 1.0040, symbol)],
                "profile_label": RISK_PROFILE_LABELS["conservador"],
            },
            "moderado": {
                "stop_loss": round_price(entry * 0.9980, symbol),
                "take_profits": [round_price(entry * 1.0018, symbol), round_price(entry * 1.0032, symbol)],
                "profile_label": RISK_PROFILE_LABELS["moderado"],
            },
            "agresivo": {
                "stop_loss": round_price(entry * 0.9985, symbol),
                "take_profits": [round_price(entry * 1.0015, symbol), round_price(entry * 1.0027, symbol)],
                "profile_label": RISK_PROFILE_LABELS["agresivo"],
            },
        }

    return {
        "conservador": {
            "stop_loss": round_price(entry * 1.0025, symbol),
            "take_profits": [round_price(entry * 0.9980, symbol), round_price(entry * 0.9960, symbol)],
            "profile_label": RISK_PROFILE_LABELS["conservador"],
        },
        "moderado": {
            "stop_loss": round_price(entry * 1.0020, symbol),
            "take_profits": [round_price(entry * 0.9982, symbol), round_price(entry * 0.9968, symbol)],
            "profile_label": RISK_PROFILE_LABELS["moderado"],
        },
        "agresivo": {
            "stop_loss": round_price(entry * 1.0015, symbol),
            "take_profits": [round_price(entry * 0.9985, symbol), round_price(entry * 0.9973, symbol)],
            "profile_label": RISK_PROFILE_LABELS["agresivo"],
        },
    }


def generate_user_signal(base_signal: Dict, user_id: int) -> Dict:
    now = datetime.utcnow()
    existing = user_signals_collection().find_one(
        {
            "user_id": user_id,
            "symbol": base_signal["symbol"],
            "telegram_valid_until": {"$gt": now},
        }
    )
    if existing:
        return existing

    direction = str(base_signal["direction"]).upper()
    entry = float(base_signal["entry_price"])
    symbol = base_signal["symbol"]
    profiles = base_signal.get("profiles") or _fallback_profiles(direction, entry, symbol=symbol)

    normalized_profiles = {}
    for profile_name in ["conservador", "moderado", "agresivo"]:
        src = profiles.get(profile_name, {})
        tps = list(src.get("take_profits", [entry, entry]))
        while len(tps) < 2:
            tps.append(entry)
        normalized_profiles[profile_name] = {
            "stop_loss": round_price(float(src.get("stop_loss", entry)), symbol),
            "take_profits": [round_price(float(tps[0]), symbol), round_price(float(tps[1]), symbol)],
            "profile_label": str(src.get("profile_label") or RISK_PROFILE_LABELS[profile_name]),
        }

    user_signal = {
        "user_id": user_id,
        "signal_id": str(base_signal["_id"]),
        "symbol": symbol,
        "direction": direction,
        "entry_price": round_price(entry, symbol),
        "entry_zone": dict(zip(["low", "high"], calculate_entry_zone(entry, atr_pct=base_signal.get("signal_atr_pct"), symbol=symbol))),
        "profiles": normalized_profiles,
        "risk_profiles": RISK_PROFILE_LABELS,
        "timeframes": base_signal["timeframes"],
        "created_at": now,
        "valid_until": base_signal["valid_until"],
        "telegram_valid_until": base_signal["telegram_valid_until"],
        "fingerprint": secrets.token_hex(4),
        "visibility": base_signal["visibility"],
        "score": base_signal.get("score"),
        "evaluated": False,
    }

    user_signals_collection().insert_one(user_signal)
    return user_signal


def format_user_signal(user_signal: Dict) -> str:
    tz = pytz.timezone(USER_TIMEZONE)
    start = user_signal["created_at"].astimezone(tz).strftime("%H:%M")
    end = user_signal["telegram_valid_until"].astimezone(tz).strftime("%H:%M")
    symbol = user_signal["symbol"]

    lines = [
        "📊 NUEVA SEÑAL – FOREX",
        "",
        f"🏷️ PLAN: {user_signal['visibility'].upper()}",
        "",
        f"Par: {symbol}",
        f"Dirección: {user_signal['direction']}",
        f"Entrada base: {format_price(user_signal['entry_price'], symbol)}",
        f"Zona de entrada: {format_price(user_signal['entry_zone']['low'], symbol)} - {format_price(user_signal['entry_zone']['high'], symbol)}",
        f"Timeframes: {' / '.join(user_signal['timeframes'])}",
        "",
    ]

    for profile in ["conservador", "moderado", "agresivo"]:
        p = user_signal["profiles"][profile]
        lines.extend(
            [
                "━━━━━━━━━━━━━━━━━━",
                f"{profile.upper()} · {p['profile_label']}",
                f"SL: {format_price(p['stop_loss'], symbol)}",
                f"TP1: {format_price(p['take_profits'][0], symbol)}",
                f"TP2: {format_price(p['take_profits'][1], symbol)}",
                "",
            ]
        )

    lines.append(f"⏳ Activa: {start} → {end}")
    lines.append(f"🔐 ID: {user_signal['fingerprint']}")
    return "\n".join(lines)


def get_latest_base_signal_for_plan(user_id: int, user_plan: Optional[str] = None):
    visibility = PLAN_PREMIUM if is_admin(user_id) else (user_plan or PLAN_FREE)
    now = datetime.utcnow()
    return list(
        user_signals_collection()
        .find(
            {
                "user_id": user_id,
                "visibility": visibility,
                "telegram_valid_until": {"$gt": now},
            }
        )
        .sort("created_at", -1)
        .limit(MAX_SIGNALS_PER_QUERY)
    )


def _evaluate_signal_result(signal_doc: Dict) -> str:
    direction = str(signal_doc.get("direction", "")).upper()
    symbol = signal_doc.get("symbol")
    stop_loss = signal_doc.get("stop_loss")
    take_profits = signal_doc.get("take_profits", [])

    if (stop_loss is None or not take_profits) and signal_doc.get("profiles"):
        conservador = signal_doc.get("profiles", {}).get("conservador", {})
        stop_loss = conservador.get("stop_loss")
        take_profits = conservador.get("take_profits", [])

    tp1 = take_profits[0] if take_profits else None
    created_at = signal_doc.get("created_at")
    valid_until = signal_doc.get("valid_until")

    if not symbol or not direction or stop_loss is None or tp1 is None or not created_at or not valid_until:
        return "expired"

    try:
        stop_loss = float(stop_loss)
        tp1 = float(tp1)
    except Exception:
        return "expired"

    try:
        candles = get_latest_completed_candles_between(symbol, created_at, valid_until, granularity="M1")
    except Exception as exc:
        logger.error("❌ Error descargando velas para evaluar %s: %s", symbol, exc)
        return "expired"

    for row in candles:
        high = float(row.get("high", 0.0))
        low = float(row.get("low", 0.0))
        if direction == "LONG":
            if low <= stop_loss and high >= tp1:
                return "lost"
            if high >= tp1:
                return "won"
            if low <= stop_loss:
                return "lost"
        elif direction == "SHORT":
            if high >= stop_loss and low <= tp1:
                return "lost"
            if low <= tp1:
                return "won"
            if high >= stop_loss:
                return "lost"

    return "expired"


def evaluate_expired_signals(limit: int = 100) -> int:
    now = datetime.utcnow()
    pending = list(
        signals_collection()
        .find({"valid_until": {"$lte": now}, "evaluated": {"$ne": True}})
        .sort("valid_until", 1)
        .limit(limit)
    )

    processed = 0
    for signal in pending:
        try:
            result = _evaluate_signal_result(signal)
            evaluated_at = datetime.utcnow()
            result_doc = {
                "base_signal_id": str(signal.get("_id")),
                "signal_id": str(signal.get("_id")),
                "user_id": None,
                "symbol": signal.get("symbol"),
                "direction": signal.get("direction"),
                "visibility": signal.get("visibility"),
                "plan": signal.get("visibility"),
                "score": signal.get("score"),
                "result": result,
                "evaluated_at": evaluated_at,
                "evaluated_profile": "conservador",
                "evaluation_scope": "base",
                "tp_used": (signal.get("take_profits") or [None])[0],
                "sl_used": signal.get("stop_loss"),
                "signal_created_at": signal.get("created_at"),
                "signal_valid_until": signal.get("valid_until"),
            }
            signal_results_collection().insert_one(result_doc)
            signals_collection().update_one(
                {"_id": signal["_id"]},
                {
                    "$set": {
                        "evaluated": True,
                        "result": result,
                        "evaluated_at": evaluated_at,
                        "evaluated_profile": "conservador",
                    }
                },
            )
            processed += 1
        except Exception as exc:
            logger.error("❌ Error evaluando señal base %s: %s", signal.get("symbol"), exc, exc_info=True)

    if processed:
        logger.info("✅ Señales base evaluadas automáticamente: %s", processed)
    return processed
