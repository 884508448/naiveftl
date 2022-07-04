import logging
import colorlog
import time

__LOGGER = None
log_colors_config = {
    "DEBUG": "white",  # cyan white
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def get_logger():
    global __LOGGER
    if __LOGGER is None:
        __LOGGER = logging.getLogger()
        __LOGGER.setLevel(logging.DEBUG)

        fileHandler = logging.FileHandler(f"logs/ftl_log_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}.log", mode="w")
        fileHandler.setLevel(logging.DEBUG)
        consoleHandler = colorlog.StreamHandler()
        consoleHandler.setLevel(colorlog.DEBUG)

        # set formatter
        file_formatter = logging.Formatter("[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",)
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_colors_config,
        )
        
        fileHandler.setFormatter(file_formatter)
        consoleHandler.setFormatter(console_formatter)

        # add
        __LOGGER.addHandler(fileHandler)
        __LOGGER.addHandler(consoleHandler)

        __LOGGER.debug("LOGGER initialized")
        return __LOGGER

    else:
        return __LOGGER


LOGGER = get_logger()
