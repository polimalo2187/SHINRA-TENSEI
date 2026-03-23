from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from app.database import get_db
from app.oanda_api import get_valid_instruments, normalize_instrument, normalize_many_instruments

collection = get_db()["watchlists"]


def _now():
    return datetime.now(timezone.utc)


def get_valid_symbols():
    return get_valid_instruments()


def normalize_symbol(symbol: str) -> Optional[str]:
    return normalize_instrument(symbol)


def normalize_many(raw: str) -> List[str]:
    return normalize_many_instruments(raw)


def get_symbols(user_id: int) -> List[str]:
    doc = collection.find_one({"user_id": int(user_id)}, {"symbols": 1})
    if not doc:
        return []
    return list(doc.get("symbols", []))


def get_watchlist(user_id: int) -> List[str]:
    return get_symbols(user_id)


def _ensure_doc(user_id: int):
    collection.update_one(
        {"user_id": int(user_id)},
        {
            "$setOnInsert": {
                "user_id": int(user_id),
                "symbols": [],
                "created_at": _now(),
            },
            "$set": {"updated_at": _now()},
        },
        upsert=True,
    )


def _plan_limit(plan: str):
    p = (plan or "FREE").upper().strip()
    if p == "PREMIUM":
        return None
    if p == "PLUS":
        return 10
    return 2


def add_symbol(user_id: int, symbol: str, plan: str = "FREE"):
    sym = normalize_symbol(symbol)
    if not sym:
        return False, "❌ Símbolo inválido. Ejemplos válidos: EURUSD, GBP_USD, XAUUSD"

    valid = get_valid_symbols()
    if valid and sym not in valid:
        return False, f"❌ {sym} no está habilitado en este bot Forex."

    current = get_symbols(int(user_id))
    if sym in current:
        return True, f"✅ {sym} ya está en tu Watchlist."

    limit = _plan_limit(plan)
    if limit is not None and len(current) >= limit:
        return False, f"🔒 Tu plan permite hasta {limit} símbolos en Watchlist."

    _ensure_doc(int(user_id))
    collection.update_one(
        {"user_id": int(user_id)},
        {
            "$addToSet": {"symbols": sym},
            "$set": {"updated_at": _now()},
        },
        upsert=True,
    )
    return True, f"✅ {sym} añadido a tu Watchlist."


def set_symbols(user_id: int, symbols: Iterable[str]):
    normalized = []
    seen = set()
    valid = get_valid_symbols()

    for symbol in symbols:
        sym = normalize_symbol(symbol)
        if not sym:
            continue
        if valid and sym not in valid:
            continue
        if sym not in seen:
            normalized.append(sym)
            seen.add(sym)

    collection.update_one(
        {"user_id": int(user_id)},
        {
            "$set": {
                "symbols": normalized,
                "updated_at": _now(),
            },
            "$setOnInsert": {"created_at": _now()},
        },
        upsert=True,
    )
    return True, "✅ Watchlist actualizada."


def remove_symbol(user_id: int, symbol: str):
    sym = normalize_symbol(symbol)
    if not sym:
        return False, "❌ Símbolo inválido."

    _ensure_doc(int(user_id))
    collection.update_one(
        {"user_id": int(user_id)},
        {
            "$pull": {"symbols": sym},
            "$set": {"updated_at": _now()},
        },
        upsert=True,
    )
    return True, f"✅ {sym} eliminado de tu Watchlist."


def clear(user_id: int):
    _ensure_doc(int(user_id))
    collection.update_one(
        {"user_id": int(user_id)},
        {
            "$set": {
                "symbols": [],
                "updated_at": _now(),
            }
        },
        upsert=True,
    )
    return True, "✅ Watchlist limpiada."


def clear_watchlist(user_id: int):
    return clear(user_id)


def format_watchlist(symbols: List[str]) -> str:
    if not symbols:
        return (
            "⭐ Watchlist vacía.\n\n"
            "Escribe un símbolo para añadir.\n"
            "Ejemplos válidos: EURUSD, GBP_USD, XAUUSD"
        )

    lines = ["⭐ WATCHLIST\n"]
    for i, symbol in enumerate(symbols, 1):
        lines.append(f"{i}) {symbol}")
    lines.append("\nTip: puedes escribir varios separados por coma. Ej: EURUSD, GBPUSD, XAUUSD")
    return "\n".join(lines)
