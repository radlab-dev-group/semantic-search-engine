import abc
from typing import Any, Dict, List


class ProcessorI(abc.ABC):
    def __init__(self, name: str, options: Dict):
        assert len(name) and len(name.strip())

        self._name = name
        self._options = options

        self.executor = None

    @property
    def use_executor(self):
        if self._options is None or not len(self._options):
            return False
        return self._options.get("use_executor", False)

    def __str__(self):
        return self._name

    @property
    def name(self):
        return self._name

    @property
    def options(self):
        return self._options

    @abc.abstractmethod
    def process(self, text: str) -> List | Dict[Any, Any]:
        pass
