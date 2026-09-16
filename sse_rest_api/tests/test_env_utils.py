"""
test_env_utils.py
-----------------

Value parsing is a classic deployment footgun.  Every non-empty string coming
from the environment or from a json config file is truthy in Python, so a
mistyped ``ENV_DEBUG=off`` silently enables the debug page with tracebacks.
A comma separated list fails in the same quiet way: Django iterates an
unsplit ``ENV_ALLOWED_HOSTS`` character by character, so every request -
including the container healthcheck - is rejected with ``DisallowedHost``.

These tests pin the accepted vocabulary of ``main.src.env_utils`` and verify
that the environment helper and the config-file helper agree on it.  They are
framework-free: no Django, no configuration file, no database.
"""

import pytest

from main.src.env_utils import (
    TRUE_CONFIG_VALUES,
    bool_config_value,
    bool_env_value,
    list_env_values,
    split_list_value,
)

TRUTHY = [
    True,
    1,
    "1",
    "true",
    "TRUE",
    "True",
    " true ",
    "t",
    "y",
    "yes",
    "tak",
    "on",
]
FALSEY = [False, 0, "0", "false", "FALSE", "off", "no", "n", "", "  ", "None", None]


@pytest.mark.parametrize("value", TRUTHY)
def test_truthy_values_are_recognized(value):
    assert bool_config_value(value) is True


@pytest.mark.parametrize("value", FALSEY)
def test_falsey_values_are_rejected(value):
    assert bool_config_value(value) is False


def test_accepted_vocabulary_is_lowercase_and_stripped():
    for value in TRUE_CONFIG_VALUES:
        if isinstance(value, str):
            assert value == value.strip().lower()


def test_unset_variable_is_false():
    assert bool_env_value("SSE_TEST_UNSET_FLAG") is False


@pytest.mark.parametrize("value", TRUTHY)
def test_environment_reads_every_truthy_value(monkeypatch, value):
    monkeypatch.setenv("SSE_TEST_FLAG", str(value))
    assert bool_env_value("SSE_TEST_FLAG") is True


@pytest.mark.parametrize("value", FALSEY)
def test_environment_reads_every_falsey_value(monkeypatch, value):
    monkeypatch.setenv("SSE_TEST_FLAG", str(value))
    assert bool_env_value("SSE_TEST_FLAG") is False


@pytest.mark.parametrize("value", TRUTHY + FALSEY)
def test_environment_and_config_file_share_one_vocabulary(monkeypatch, value):
    """ENV_DEBUG=0 and "debug": "0" must not drift apart."""
    monkeypatch.setenv("SSE_TEST_FLAG", str(value))
    assert bool_env_value("SSE_TEST_FLAG") is bool_config_value(value)


LIST_CASES = [
    ("localhost,127.0.0.1", ["localhost", "127.0.0.1"]),
    ("localhost", ["localhost"]),
    (" localhost , 127.0.0.1 ", ["localhost", "127.0.0.1"]),
    ("a,,b,", ["a", "b"]),
    (" , , ", []),
    ("", []),
    # A port is part of the host value, not a separator (Django accepts both).
    ("example.com:8000", ["example.com:8000"]),
    ("a, b, a", ["a", "b"]),
]


@pytest.mark.parametrize(("value", "expected"), LIST_CASES)
def test_comma_separated_value_is_split(value, expected):
    assert split_list_value(value) == expected


@pytest.mark.parametrize(("value", "expected"), LIST_CASES)
def test_multi_value_is_never_handed_over_as_a_bare_string(value, expected):
    """ALLOWED_HOSTS iterates its entries - a string would mean one-char hosts."""
    result = split_list_value(value)
    assert isinstance(result, list)
    assert len(result) == len(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["localhost", "127.0.0.1"], ["localhost", "127.0.0.1"]),
        (("localhost", "127.0.0.1"), ["localhost", "127.0.0.1"]),
        ([" localhost ", ""], ["localhost"]),
        (["a,b", "c"], ["a", "b", "c"]),
        ([], []),
        ((), []),
        (None, []),
    ],
)
def test_json_array_is_normalised(value, expected):
    assert split_list_value(value) == expected


def test_duplicates_collapse_and_order_is_kept():
    assert split_list_value("b, a, b, c , a") == ["b", "a", "c"]


def test_unset_variable_produces_no_entries():
    assert list_env_values("SSE_TEST_UNSET_LIST") == []


@pytest.mark.parametrize(("value", "expected"), LIST_CASES)
def test_environment_and_config_file_share_one_list_syntax(
    monkeypatch, value, expected
):
    """ENV_ALLOWED_HOSTS=a,b and "allowed_hosts": "a,b" must mean the same."""
    monkeypatch.setenv("SSE_TEST_LIST", value)
    assert list_env_values("SSE_TEST_LIST") == expected
    assert split_list_value(value) == expected
