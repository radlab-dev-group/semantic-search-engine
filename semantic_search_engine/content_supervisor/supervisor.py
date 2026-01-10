from typing import List, Dict, Any

from content_supervisor.processors.regex_processors import URLRegexProcessor


class ContentSupervisorOutput:
    """
    Single supervisor processor output
    """

    def __init__(self, content_type: str | None = None, content_body: Any = None):
        self._content_type = content_type
        self._content_body = content_body

    @property
    def content_type(self):
        return self._content_type

    @property
    def content_body(self):
        return self._content_body


class ContentSupervisor:
    def __init__(self, options: Dict):
        self._options = options
        self._processors_chain = None

        self.__prepare_processors_chain()

    def check_text(self, text_str: str) -> List[ContentSupervisorOutput]:
        text_content = []
        for processor in self._processors_chain:
            content_body = processor.process(text_str)
            if content_body is not None and len(content_body):
                text_content.append(
                    ContentSupervisorOutput(
                        content_type=processor.name, content_body=content_body
                    )
                )
        return text_content

    def __prepare_processors_chain(self):
        self._processors_chain = [URLRegexProcessor(self._options)]
