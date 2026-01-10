import re
from typing import Any, List, Dict

from radlab_content_supervisor.processors.processors import ProcessorI
from radlab_content_supervisor.executors.www_content_grabber import (
    WWWContentGrabberExecutor,
)


class RegexProcessor(ProcessorI):
    def __init__(self, pattern: str | List[str], name: str, options: Dict):
        super().__init__(name, options)

        assert len(pattern)
        if type(pattern) is str:
            pattern = [pattern]
        self._name = name
        self._patterns = pattern

    def process(self, text: str) -> List | Dict[Any, Any]:
        all_matches = []
        for pattern in self._patterns:
            patter_matches = re.findall(pattern, text)
            patter_matches = [x[0] for x in patter_matches]
            if len(patter_matches):
                all_matches.extend(patter_matches)

        if self.use_executor and len(all_matches):
            assert self.executor is not None
            return self.executor.apply(all_matches)

        return all_matches


class URLRegexProcessor(RegexProcessor):
    SELF_PROCESSOR_NAME = "RegexProcessor-URL"

    SELF_URL_PATTERNS = [r"((https|http)?://[^\s]+)"]

    def __init__(self, options: Dict):
        super().__init__(
            pattern=self.SELF_URL_PATTERNS,
            name=self.SELF_PROCESSOR_NAME,
            options=options,
        )

        if self.use_executor:
            self.executor = WWWContentGrabberExecutor()
