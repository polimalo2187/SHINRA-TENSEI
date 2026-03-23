from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from app.market import get_market_snapshot
from app.oanda_api import format_price

CB_MARKET_REFRESH = "market_refresh"
CB_BACK_MENU = "back_menu"


def render_market():
    snap = get_market_snapshot()
    if not snap:
        return (
            "❌ No pude cargar datos del mercado ahora mismo.",
            InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Volver", callback_data=CB_BACK_MENU)]]),
        )

    text = "📊 MERCADO FOREX\n\n"
    if snap["eurusd"]:
        text += f"EUR_USD: {float(snap['eurusd']['priceChangePercent']):+.2f}%\n"
    if snap["gbpusd"]:
        text += f"GBP_USD: {float(snap['gbpusd']['priceChangePercent']):+.2f}%\n"

    text += "\n🔥 Top Gainers:\n"
    for row in snap["gainers"]:
        text += f"{row['symbol']} {float(row['priceChangePercent']):+.2f}%\n"

    text += "\n📉 Top Losers:\n"
    for row in snap["losers"]:
        text += f"{row['symbol']} {float(row['priceChangePercent']):+.2f}%\n"

    text += f"\n🕒 Actualizado: {snap['time']}"
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Actualizar", callback_data=CB_MARKET_REFRESH)],
        [InlineKeyboardButton("⬅️ Volver", callback_data=CB_BACK_MENU)],
    ])
    return text, keyboard
