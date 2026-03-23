# app/menus.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

MENU_TEXT = "🏠 MENÚ PRINCIPAL — Selecciona una opción abajo"
ADMIN_TEXT = "🛠 PANEL ADMIN — Selecciona una opción abajo"


def main_menu(is_admin: bool = False) -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("🚨 Señales en vivo", callback_data="view_signals")],
        [
            InlineKeyboardButton("📡 Radar Forex", callback_data="radar"),
            InlineKeyboardButton("🎯 Rendimiento", callback_data="performance"),
        ],
        [
            InlineKeyboardButton("🔥 Movers Forex", callback_data="movers"),
            InlineKeyboardButton("📊 Mercado", callback_data="market"),
        ],
        [
            InlineKeyboardButton("⭐ Watchlist", callback_data="watchlist"),
            InlineKeyboardButton("🔔 Alertas", callback_data="alerts"),
        ],
        [InlineKeyboardButton("🧾 Historial", callback_data="history")],
        [
            InlineKeyboardButton("💼 Planes", callback_data="plans"),
            InlineKeyboardButton("👥 Referidos", callback_data="referrals"),
        ],
        [
            InlineKeyboardButton("👤 Mi cuenta", callback_data="my_account"),
            InlineKeyboardButton("📩 Soporte", callback_data="support"),
        ],
    ]

    if is_admin:
        keyboard.insert(1, [InlineKeyboardButton("🛠 Panel Admin", callback_data="admin_panel")])

    return InlineKeyboardMarkup(keyboard)


def back_to_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Volver al menú", callback_data="back_menu")]])


def admin_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Activar plan PLUS", callback_data="admin_activate_plus")],
        [InlineKeyboardButton("👑 Activar plan PREMIUM", callback_data="admin_activate_premium")],
        [InlineKeyboardButton("⏳ Extender plan actual", callback_data="admin_extend_plan")],
        [InlineKeyboardButton("📊 Estadísticas", callback_data="admin_stats")],
        [InlineKeyboardButton("⬅️ Volver", callback_data="back_menu")],
    ])
