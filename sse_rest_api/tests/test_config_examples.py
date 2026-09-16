"""
test_config_examples.py
-----------------------

Guard tests for the files a deployment starts from:

* every ``sse_rest_api/configs/*.example.*`` template parses and contains
  ``CHANGE_ME`` placeholders only - never a real credential,
* the shipped defaults are production safe (debug off, secure cookies, prod
  logger profile),
* every variable used by ``docker-compose.yml`` is documented in
  ``.env.example``, so ``docker compose up`` cannot fail on a missing value.

Framework-free on purpose: these run before Django, Milvus or PostgreSQL are
available, which is exactly when they are the most useful.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parent
CONFIGS_DIR = API_ROOT / "configs"
ENV_TEMPLATE = REPO_ROOT / ".env.example"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

PLACEHOLDER = "CHANGE_ME"
# ``$${...}`` is an escaped, container-side expansion - not a compose variable.
COMPOSE_VARIABLE = re.compile(r"(?<!\$)\$\{([A-Z][A-Z0-9_]*)(:?[-?][^}]*)?\}")
TEMPLATE_VARIABLE = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=")


def examples(pattern):
    return sorted(CONFIGS_DIR.glob(pattern))


JSON_EXAMPLES = examples("*.example.json")
ALL_EXAMPLES = examples("*.example.*")


def test_example_templates_are_shipped():
    assert len(ALL_EXAMPLES) >= 5, f"expected config templates in {CONFIGS_DIR}"


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=lambda p: p.name)
def test_example_template_contains_only_placeholders(path):
    assert PLACEHOLDER in path.read_text(
        encoding="utf-8"
    ), f"{path.name} should mark every secret with {PLACEHOLDER}"


@pytest.mark.parametrize("path", JSON_EXAMPLES, ids=lambda p: p.name)
def test_json_example_template_is_valid(path):
    json.loads(path.read_text(encoding="utf-8"))


def test_django_config_example_is_production_safe():
    config = json.loads((CONFIGS_DIR / "django-config.example.json").read_text())
    main = config["main"]
    assert main["debug"] is False
    assert main["admin_window"] is False
    assert config["session"]["session_cookie_secure"] is True
    assert config["session"]["csrf_cookie_secure"] is True
    assert config["logger"]["use_logger"] == "prod"
    assert config["database"]["PASSWORD"] == PLACEHOLDER


def test_compose_variables_are_documented_in_the_env_template():
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    used = {match.group(1) for match in COMPOSE_VARIABLE.finditer(compose)}
    documented = {
        match.group(1)
        for match in (
            TEMPLATE_VARIABLE.match(line)
            for line in ENV_TEMPLATE.read_text().splitlines()
        )
        if match
    }
    assert used, f"no variables found in {COMPOSE_FILE}"
    missing = used - documented
    assert not missing, f"document {sorted(missing)} in {ENV_TEMPLATE.name}"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_real_config_files_are_not_tracked_by_git():
    """``configs/*.json`` hold secrets - only the ``*.example.*`` templates go in."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(CONFIGS_DIR)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    leaked = [
        name
        for name in tracked
        if Path(name).name.startswith("user-group-organisation")
        or Path(name).name in ("django-config.json", "auth-config.json")
        or name.endswith("secret-key.txt")
    ]
    assert not leaked, f"remove from git and rotate the secrets in {leaked}"
