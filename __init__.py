"""
Forex Signal Bot
================

Bot de Telegram para escaneo de Forex y metales con señales multi-timeframe.
"""

import logging
import sys


def setup_logging():
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )

    log_level = logging.INFO
    if "--debug" in sys.argv or "-d" in sys.argv:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("✅ Logging configurado")
    return logger


__version__ = "1.1.0"
__author__ = "Forex Signal Bot"
__description__ = "Bot de escaneo Forex/Metals con señales MTF"


def initialize_app():
    logger = logging.getLogger(__name__)
    try:
        logger.info("=" * 60)
        logger.info("Iniciando Forex Signal Bot v%s", __version__)
        logger.info("=" * 60)

        import os
        critical_vars = ["BOT_TOKEN", "MONGODB_URI", "DATABASE_NAME"]
        missing_vars = [var for var in critical_vars if not os.getenv(var)]
        if missing_vars:
            logger.error("❌ Variables de entorno faltantes: %s", missing_vars)
            return False

        logger.info("✅ Variables de entorno críticas verificadas")
        logger.info("✅ Aplicación inicializada correctamente")
        return True
    except Exception as e:
        logger.error("❌ Error durante la inicialización: %s", e)
        return False


if __name__ == "__main__":
    print(f"{__description__} v{__version__}")
    print(f"Por: {__author__}")
    print()
    print("Este archivo es parte del paquete 'app'.")
    print("Para iniciar el bot, ejecuta: python main.py")
