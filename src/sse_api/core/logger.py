import json
import logging


class Logger:
    level = "DEBUG"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __init__(self, logger_name: str, logger_config: dict = None):
        self._logger_name = logger_name
        self._logger_config = logger_config
        self._apply()

    @property
    def obj(self):
        return self._logger

    def load_from_dict(self, logger_config: dict):
        if logger_config is not None:
            self._logger_config = logger_config
            self._apply()

    def _apply(self):
        # self._logger = logging.getLogger(logger_name)

        if self._logger_config is not None:
            logging.config.dictConfig(self._logger_config)
            # TODO: which logger to use?
            # TODO: set self._logger as proper
        else:
            # TODO: handle when no config is given
            pass
