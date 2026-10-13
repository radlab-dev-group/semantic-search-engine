import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sse_api.tests.settings_django")
import django

django.setup()

_controllers_path = Path(__file__).resolve().parents[1] / "controllers"
_temporary_modules = {
    "sse_api.engine.controllers": ModuleType(
        "sse_api.engine.controllers"
    ),
    "sse_api.engine.controllers.search": ModuleType(
        "sse_api.engine.controllers.search"
    ),
    "sse_api.engine.controllers.search.semantic": ModuleType(
        "sse_api.engine.controllers.search.semantic"
    ),
}
_temporary_modules["sse_api.engine.controllers"].__path__ = [
    str(_controllers_path)
]
_temporary_modules["sse_api.engine.controllers.search"].__path__ = [
    str(_controllers_path / "search")
]
_temporary_modules[
    "sse_api.engine.controllers.search.semantic"
].DBSemanticSearchController = MagicMock()
_previous_modules = {
    name: sys.modules.get(name)
    for name in (
        *_temporary_modules,
        "sse_api.engine.controllers.models_logic",
        "sse_api.engine.controllers.models_logic.generative",
    )
}
sys.modules.update(_temporary_modules)
try:
    generative = importlib.import_module(
        "sse_api.engine.controllers.models_logic.generative"
    )
finally:
    for name, module in _previous_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

from llm_router_lib.data_models.builtin_chat import ConversationWithModelRequest
from llm_router_lib.data_models.builtin_utils import GenerativeAnswerModel
from llm_router_lib.data_models.response import (
    ConversationResponse,
    GenerativeAnswerResponse,
)


def _controller():
    controller = object.__new__(generative.GenerativeModelControllerApi)
    controller.models_config = SimpleNamespace(
        active_local_models_hosts={"test-model": "http://localhost:8000"}
    )
    return controller


def test_router_client_uses_model_config_when_env_is_blank(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_API", " ")
    monkeypatch.setenv("LLM_ROUTER_TOKEN", " ")
    client = MagicMock()

    with patch.dict(generative.__dict__, {"LLMRouterClient": client}):
        _controller()._create_router_client("test-model")

    client.assert_called_once_with(
        api="http://localhost:8000",
        token=None,
        timeout=120,
    )


def test_model_config_adds_http_scheme_to_scheme_less_host(tmp_path):
    config_path = tmp_path / "generative-models.json"
    config_path.write_text(
        json.dumps(
            {
                "api_hosts": {
                    "local-model": "localhost:8080",
                    "secure-model": "https://models.example.com",
                },
                "active_api_models": ["local-model", "secure-model"],
            }
        ),
        encoding="utf-8",
    )

    config = generative.GenerativeModelConfig(str(config_path))

    assert config.active_local_models_hosts == {
        "local-model": "http://localhost:8080",
        "secure-model": "https://models.example.com",
    }


def test_generative_answer_uses_typed_payload_and_router_env(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_API", "router.example.com/")
    monkeypatch.setenv("LLM_ROUTER_TOKEN", "test-token")
    client = MagicMock()
    client.return_value.generative_answer.return_value = GenerativeAnswerResponse(
        response="answer", generation_time=0.25
    )

    with (
        patch.dict(generative.__dict__, {"LLMRouterClient": client}),
        patch.dict(
            generative.__dict__,
            {
                "convert_search_results_to_doc2answer": MagicMock(
                    return_value={"doc": ["context"]}
                )
            },
        ),
    ):
        result = _controller().generative_answer_local_api_model(
            question_str="question",
            search_results={},
            qa_gen_model="test-model",
            generation_options={"top_k": 12},
            system_prompt="system",
        )

    payload = client.return_value.generative_answer.call_args.kwargs["payload"]
    assert isinstance(payload, GenerativeAnswerModel)
    assert payload.question_str == "question"
    assert payload.texts == {"doc": ["context"]}
    assert payload.top_k == 12
    assert payload.system_prompt == "system"
    assert result == ("answer", 0.25)
    client.assert_called_once_with(
        api="http://router.example.com",
        token="test-token",
        timeout=120,
    )


def test_conversation_uses_typed_payload_and_router_env(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_API", "https://router.example.com")
    monkeypatch.setenv("LLM_ROUTER_TOKEN", "conversation-token")
    client = MagicMock()
    client.return_value.conversation_with_model.return_value = ConversationResponse(
        response=" answer ", generation_time=0.5
    )

    with patch.dict(generative.__dict__, {"LLMRouterClient": client}):
        result = _controller().conversation_with_local_model(
            history=[{"role": "user", "content": "prior"}],
            last_user_message="latest",
            options={"top_k": 12},
            model_name_path="test-model",
        )

    payload = client.return_value.conversation_with_model.call_args.kwargs[
        "payload"
    ]
    assert isinstance(payload, ConversationWithModelRequest)
    assert payload.user_last_statement == "latest"
    assert payload.historical_messages == [
        {"role": "user", "content": "prior"}
    ]
    assert payload.top_k == 12
    assert result == ("answer", 0.5)
    client.assert_called_once_with(
        api="https://router.example.com",
        token="conversation-token",
        timeout=120,
    )