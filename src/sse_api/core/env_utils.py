"""
env_utils.py
------------

Single source of truth for parsing non-boolean values that arrive from
environment variables or JSON config files.

In Python every non-empty string is truthy, so a mistyped ``ENV_DEBUG=off``
silently enables the debug page, and an unsplit ``ENV_ALLOWED_HOSTS`` makes
Django iterate the value character by character.  These helpers pin one
vocabulary for both sources so the two can never drift apart.
"""

import os

# Values (lowercase, stripped) that count as "true".  ``True``/``1`` are the
# native JSON/Python truthies; the rest cover common human spellings.
TRUE_CONFIG_VALUES = {
    True,
    1,
    "1",
    "true",
    "t",
    "y",
    "yes",
    "tak",
    "on",
}


def bool_config_value(value) -> bool:
    """
    Interpret a value read from a JSON config file as a boolean.

    Strings are matched case-insensitively and trimmed, so ``" TRUE "`` is
    true and ``"off"`` is false.  Numbers keep their native meaning.
    :param value: Raw config value (str, int, float or bool).
    :return: Parsed boolean.
    """
    if isinstance(value, str):
        return value.strip().lower() in TRUE_CONFIG_VALUES
    return bool(value)


def bool_env_value(env_name: str) -> bool:
    """
    Interpret an environment variable as a boolean.

    :param env_name: Name of the environment variable.
    :return: Parsed boolean; unset variables are ``False``.
    """
    return bool_config_value(os.getenv(env_name, ""))


def split_list_value(value) -> list:
    """
    Normalise a multi-value setting into a de-duplicated list.

    Accepts a comma-separated string, a list/tuple, or ``None``.  Entries are
    trimmed and empty entries dropped.  A comma is the only separator, so
    ``example.com:8000`` stays a single host.  Order is preserved, duplicates
    collapse.
    :param value: Raw string, iterable or ``None``.
    :return: List of non-empty string entries.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw_entries = value.split(",")
    else:
        # Each item may itself be comma separated, so flatten them all.
        raw_entries = [str(item) for item in value if item is not None]
        raw_entries = [part for chunk in raw_entries for part in chunk.split(",")]
    result = []
    for entry in raw_entries:
        entry = entry.strip()
        if entry and entry not in result:
            result.append(entry)
    return result


def list_env_values(env_name: str) -> list:
    """
    Read a comma-separated environment variable as a list.
    :param env_name: Name of the environment variable.
    :return: Normalised list; unset variables yield an empty list.
    """
    return split_list_value(os.getenv(env_name))
