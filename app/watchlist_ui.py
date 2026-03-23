from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from app.oanda_api import format_price, get_watchlist_snapshot
from app.watchlist import format_watchlist

CB_WL_REFRESH = "wl_refresh"
CB_WL_CLEAR = "wl_clear"
CB_WL_REMOVE_PREFIX = "wl_rm:"
CB_BACK_MENU = "back_menu"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def watchlist_keyboard(symbols):
    rows = [
        [
            InlineKeyboardButton("🔄 Actualizar", callback_data=CB_WL_REFRESH),
            InlineKeyboardButton("🧹 Limpiar", callback_data=CB_WL_CLEAR),
        ]
    ]

    for symbol in symbols[:6]:
        rows.append([InlineKeyboardButton(f"❌ Quitar {symbol}", callback_data=f"{CB_WL_REMOVE_PREFIX}{symbol}")])

    rows.append([InlineKeyboardButton("⬅️ Volver", callback_data=CB_BACK_MENU)])
    return InlineKeyboardMarkup(rows)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.2f}%"


def _fmt_activity(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def render_watchlist_view(symbols):
    base = format_watchlist(symbols)
    snapshot: Dict[str, dict] = get_watchlist_snapshot(symbols)
    lines = [base]

    if snapshot:
        lines.append("\n📊 Watchlist PRO (Forex):")
        for symbol in symbols[:10]:
            if symbol not in snapshot:
                continue

            data = snapshot[symbol]
            lines.append(
                f"\n{symbol}\n"
                f"Precio: {format_price(data['price'], symbol)}\n"
                f"24h: {_fmt_pct(data['chg_24h'])} | 1h: {_fmt_pct(data['chg_1h'])} | 4h: {_fmt_pct(data['chg_4h'])}\n"
                f"Actividad 24h: {_fmt_activity(float(data['volume']))}\n"
                f"ATR 1h: {float(data['atr_pct']) * 100:.3f}%\n"
                f"Tendencia: {data['trend']} | Momentum: {data['momentum']}"
            )
    elif symbols:
        lines.append("\nℹ️ No pude cargar datos de mercado ahora mismo.")

    lines.append(f"\n🕒 Actualizado: {_now()}")
    kb = watchlist_keyboard(symbols)
    return "\n".join(lines), kb
