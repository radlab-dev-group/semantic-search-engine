import abc
from typing import Any, Dict


class ExecutorI:
    def __init__(self, name_str: str, options: Dict | None = None):
        assert len(name_str)
        self._name_str = name_str
        self._options = options or {}

    @property
    def name(self):
        return self._name_str

    @property
    def options(self):
        return self._options

    @abc.abstractmethod
    def apply(self, body) -> Dict[Any, Any]:
        pass
