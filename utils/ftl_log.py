import logging

__LOGGER = None


def get_logger():
    global __LOGGER
    if __LOGGER is None:
        __LOGGER = logging.getLogger()
        __LOGGER.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler("data/ftl_log.log", mode='w')
        fileHandler.setLevel(logging.DEBUG)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)

        # set formatter
        formatter = logging.Formatter('[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        consoleHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)

        # add
        __LOGGER.addHandler(fileHandler)
        __LOGGER.addHandler(consoleHandler)

        __LOGGER.debug("LOGGER initialized")
        return __LOGGER

    else:
        return __LOGGER


LOGGER = get_logger()
