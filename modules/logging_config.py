import logging
import os

_sd_webui_log_level = None

_sd_webui_rich_log = None

try:
    from tqdm import tqdm

    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, fallback_handler: logging.Handler):
            super().__init__()
            self.fallback_handler = fallback_handler

        def emit(self, record):
            try:
                # If there are active tqdm progress bars,
                # attempt to not interfere with them.
                if tqdm._instances:
                    tqdm.write(self.format(record))
                else:
                    self.fallback_handler.emit(record)
            except Exception:
                self.fallback_handler.emit(record)

except ImportError:
    TqdmLoggingHandler = None


def setup_logging(loglevel):
    if loglevel is None:
        global _sd_webui_log_level
        if _sd_webui_log_level is None:
            _sd_webui_log_level = os.environ.get("SD_WEBUI_LOG_LEVEL")
        loglevel = _sd_webui_log_level

    if not loglevel:
        return

    if logging.root.handlers:
        # Already configured, do not interfere
        return

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    global _sd_webui_rich_log
    if _sd_webui_rich_log is None:
        _sd_webui_rich_log = os.environ.get("SD_WEBUI_RICH_LOG")
    rich_log = _sd_webui_rich_log

    if rich_log:
        from rich.logging import RichHandler

        handler = RichHandler()
        # RichHandler uses its own formatting, do not override
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    # Avoid double-wrapping TqdmLoggingHandler or setting formatter twice
    if TqdmLoggingHandler:
        handler = TqdmLoggingHandler(handler)
        # Safe to setFormatter after wrapping, since TqdmLoggingHandler delegates to fallback_handler

    handler.setFormatter(formatter)

    loglevel_upper = loglevel.upper()
    # Use dict lookup for loglevel mapping with fallback
    log_level = logging._nameToLevel.get(loglevel_upper, logging.INFO)

    # Fast path: avoid unnecessary setLevel calls if already set
    if logging.root.level != log_level:
        logging.root.setLevel(log_level)

    logging.root.addHandler(handler)
