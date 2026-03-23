# app/binance_api.py
# Compat layer legado: el bot ya no usa Binance.
# Mantengo este módulo para no romper imports antiguos.

from app.oanda_api import get_top_movers as _get_top_movers
from app.oanda_api import get_radar_opportunities, get_market_rows


def get_futures_24h_tickers():
    return get_market_rows()


def get_top_movers_usdtm(limit: int = 10, *, kind: str = "gainers"):
    return _get_top_movers(limit=limit, kind=kind)


def get_premium_index(symbol: str):
    return {}


def get_open_interest(symbol: str):
    return {}
