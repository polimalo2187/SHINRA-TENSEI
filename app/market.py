from datetime import datetime, timezone

from app.oanda_api import get_market_rows


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_market_snapshot():
    rows = get_market_rows()
    if not rows:
        return None

    gainers = sorted(rows, key=lambda x: float(x.get("priceChangePercent", 0.0)), reverse=True)[:5]
    losers = sorted(rows, key=lambda x: float(x.get("priceChangePercent", 0.0)))[:5]

    anchors = {row["symbol"]: row for row in rows}
    return {
        "eurusd": anchors.get("EUR_USD"),
        "gbpusd": anchors.get("GBP_USD"),
        "gainers": gainers,
        "losers": losers,
        "time": _now(),
    }
