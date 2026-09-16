"""
health.py
---------

Dependency checks used by the public ``healthz`` endpoint.

Only boolean results are returned to the caller.  Full tracebacks and
connection strings (which contain credentials) are written to the
application logger instead of the HTTP response.
"""

import json
import os

from django.db import DatabaseError, connection

from main.src.constants import CONFIG_DIR, get_logger

MILVUS_CONFIG_FILENAME = "milvus_config.json"
MILVUS_CONNECTION_JSON_FIELD = "milvus_db_connection"

MILVUS_ENV_OVERRIDES = {
    "host": "ENV_MILVUS_HOST",
    "port": "ENV_MILVUS_PORT",
    "db_name": "ENV_MILVUS_DBNAME",
    "user": "ENV_MILVUS_USER",
    "password": "ENV_MILVUS_PASSWORD",
}


def milvus_connection_config() -> dict:
    """
    Resolve the Milvus connection parameters, the same way the engine does:
    json configuration file first, then environment variable overrides.
    :return: Connection configuration dictionary
    """
    config_path = os.path.join(CONFIG_DIR, MILVUS_CONFIG_FILENAME)
    with open(config_path, "r") as config_file:
        config = json.load(config_file)[MILVUS_CONNECTION_JSON_FIELD]

    for field, env_name in MILVUS_ENV_OVERRIDES.items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            config[field] = env_value
    return config


def check_database() -> bool:
    """
    Check that the relational database accepts connections.
    :return: True when the database is reachable
    """
    try:
        connection.ensure_connection()
        return True
    except DatabaseError as exc:
        get_logger().error(f"Database health check failed: {exc}")
        return False


def check_milvus() -> bool:
    """
    Check that the vector database answers a lightweight request.
    ``pymilvus`` is imported lazily, so a missing or broken optional
    dependency is reported as an unhealthy check instead of a server error.
    :return: True when Milvus is reachable
    """
    try:
        from pymilvus import (  # pylint: disable=import-outside-toplevel
            MilvusClient,
        )

        config = milvus_connection_config()
        uri = str(config["uri"]).rstrip("/").rstrip(":")
        host = str(config["host"]).strip("/")
        client = MilvusClient(
            uri=f"{uri}://{host}:{config['port']}",
            user=config["user"],
            password=config["password"],
            db_name=config["db_name"],
        )
        try:
            client.list_collections()
        finally:
            client.close()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        get_logger().error(f"Milvus health check failed: {exc}")
        return False


def dependencies_status() -> dict:
    """
    Run every dependency check.
    :return: Dictionary with the checks results
    """
    return {"database": check_database(), "milvus": check_milvus()}
