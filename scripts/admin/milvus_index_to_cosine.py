#!/usr/bin/env python3
"""
milvus_index_to_cosine.py
-------------------------

Migrates existing Milvus collections to a **COSINE** metric index without
re-embedding the data.

For each requested collection the script:

1. releases the collection,
2. drops the existing ``emb_idx`` index,
3. recreates the index with ``metric_type=COSINE`` (parameters taken from
   ``INDEX_QUERY_PARAMS`` of the application code, so the metric always
   matches the collection),
4. loads the collection again.

Usage (run from the ``sse_rest_api`` directory so that the ``engine`` and
``data`` packages are importable):

    python scripts/admin/milvus_index_to_cosine.py --all
    python scripts/admin/milvus_index_to_cosine.py --collection my_coll --collection other_coll
    python scripts/admin/milvus_index_to_cosine.py --all --index HNSW
    python scripts/admin/milvus_index_to_cosine.py --collection my_coll --dry-run

Environment overrides (same names as the application uses):
    ENV_MILVUS_HOST, ENV_MILVUS_PORT, ENV_MILVUS_DBNAME, ENV_MILVUS_USER,
    ENV_MILVUS_PASSWORD
"""

import argparse
import json
import os
import sys

# Allow running as a script from the repository root or from sse_rest_api.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_APP_DIR = os.path.join(_REPO_ROOT, "sse_rest_api")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# The index parameter registry lives in a module whose import chain touches
# Django models, so minimal Django settings must be configured first
# (only when Django is not already set up, e.g. inside ``manage.py``).
import django  # noqa: E402
from django.conf import settings  # noqa: E402

if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django.contrib.postgres",
            "data",
            "system",
            "engine",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        DEFAULT_APP_LANGUAGE="pl",
        MAIN_API_URL="/api/",
        MAIN_LOGGER=__import__("logging").getLogger("milvus_index_to_cosine"),
    )
    django.setup()

from pymilvus import MilvusClient  # noqa: E402

from engine.controllers.database.milvus import INDEX_QUERY_PARAMS  # noqa: E402

CONFIG_JSON_FIELD = "milvus_db_connection"
DEFAULT_CONFIG_PATH = os.path.join("configs", "milvus_config.json")
DEFAULT_INDEX_NAME = "emb_idx"
DEFAULT_EMBEDDING_FIELD = "embedding"


def load_connection_config(config_path: str) -> dict:
    """Load connection settings from the JSON config with env overrides."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)[CONFIG_JSON_FIELD]

    env_map = {
        "host": "ENV_MILVUS_HOST",
        "db_name": "ENV_MILVUS_DBNAME",
        "port": "ENV_MILVUS_PORT",
        "user": "ENV_MILVUS_USER",
        "password": "ENV_MILVUS_PASSWORD",
    }
    for key, env_name in env_map.items():
        value = os.environ.get(env_name)
        if value is not None:
            cfg[key] = int(value) if key == "port" else value

    for required in ("host", "port", "db_name", "user", "password"):
        if cfg.get(required) in (None, ""):
            raise ValueError(
                f"Missing '{required}' in Milvus config (set the matching "
                f"environment variable or fix the config file)."
            )

    return cfg


def build_client(cfg: dict) -> MilvusClient:
    uri = str(cfg.get("uri", "http")).rstrip("/").rstrip(":")
    host = str(cfg["host"]).strip("/")
    uri_connection = f"{uri}://{host}:{cfg['port']}"
    return MilvusClient(
        uri=uri_connection,
        user=cfg["user"],
        password=cfg["password"],
        db_name=cfg["db_name"],
    )


def migrate_collection(
    client: MilvusClient, collection_name: str, index_type: str
) -> None:
    """Drop and recreate the index of a single collection with COSINE."""
    index_params_cfg = INDEX_QUERY_PARAMS[index_type]["INDEX_PARAMS"]
    index_name = index_params_cfg.get("index_name", DEFAULT_INDEX_NAME)
    embedding_field = DEFAULT_EMBEDDING_FIELD

    print(f"[migrate] {collection_name}: releasing collection ...")
    try:
        client.release_collection(collection_name)
    except Exception as exc:  # collection may not be loaded
        print(f"[migrate] {collection_name}: release skipped ({exc})")

    # Drop whatever indexes exist on the embedding field (the name is not
    # guaranteed to be ``emb_idx`` — e.g. AUTOINDEX collections use the
    # field name).
    existing_indexes = list(client.list_indexes(collection_name))
    if existing_indexes:
        print(
            f"[migrate] {collection_name}: dropping existing indexes: "
            f"{existing_indexes}"
        )
        for existing in existing_indexes:
            client.drop_index(collection_name, existing)
    else:
        print(f"[migrate] {collection_name}: no existing index found")

    # Some Milvus versions (2.5+) automatically recreate a default
    # AUTOINDEX (COSINE) as soon as the last index of the field is
    # dropped.  In that case there is nothing left to create.
    current_indexes = [str(name) for name in client.list_indexes(collection_name)]
    if current_indexes:
        auto_metric = client.describe_index(collection_name, current_indexes[0]).get(
            "metric_type"
        )
        if auto_metric == "COSINE":
            print(
                f"[migrate] {collection_name}: server auto-created a COSINE "
                f"index '{current_indexes[0]}' — nothing to create."
            )
        else:
            raise RuntimeError(
                f"Unexpected non-COSINE index '{current_indexes[0]}' "
                f"(metric={auto_metric}) after dropping old indexes — "
                f"manual intervention required."
            )
    else:
        print(
            f"[migrate] {collection_name}: creating index '{index_name}' "
            f"({index_type}, COSINE) ..."
        )
        index_params = client.prepare_index_params(field_name=embedding_field)
        index_params.add_index(
            field_name=embedding_field,
            index_type=index_params_cfg["index_type"],
            metric_type=index_params_cfg["metric_type"],
            params=index_params_cfg["params"],
            index_name=index_name,
        )
        client.create_index(collection_name, index_params=index_params)

    print(f"[migrate] {collection_name}: loading collection ...")
    client.load_collection(collection_name)

    print(f"[migrate] {collection_name}: done.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recreate Milvus collection indexes with metric_type=COSINE."
    )
    parser.add_argument(
        "--collection",
        action="append",
        default=[],
        help="Collection to migrate (repeatable). Mutually exclusive with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate every collection in the database.",
    )
    parser.add_argument(
        "--index",
        choices=sorted(INDEX_QUERY_PARAMS.keys()),
        default="IVF_FLAT",
        help="Index type used for the new index (default: IVF_FLAT).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to milvus_config.json (default: configs/milvus_config.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned operations; do not touch the database.",
    )
    args = parser.parse_args()

    if not args.collection and not args.all:
        parser.error("Provide at least one --collection or use --all.")
    if args.collection and args.all:
        parser.error("Use either --collection ... or --all, not both.")

    cfg = load_connection_config(args.config)

    if args.all:
        client = build_client(cfg)
        collections = list(client.list_collections())
    else:
        collections = list(args.collection)
        client = None

    print(f"Index type: {args.index} (COSINE)")
    print(f"Collections to migrate: {collections}")

    if args.dry_run:
        for col in collections:
            print(
                f"[dry-run] {col}: release -> drop index -> "
                f"create index ({args.index}, COSINE) -> load"
            )
        return 0

    client = client or build_client(cfg)
    failures = []
    for col in collections:
        try:
            migrate_collection(client, col, args.index)
        except Exception as exc:
            print(f"[error] {col}: {exc}", file=sys.stderr)
            failures.append(col)

    if failures:
        print(f"\nFailed collections: {failures}", file=sys.stderr)
        return 1
    print("\nAll collections migrated to COSINE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
