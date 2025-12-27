import os


def bool_env_value(env_name: str) -> bool:
    env_value = os.getenv(env_name, "false").lower()
    if env_value in [1, "true", "1", "t", "y", "yes", "tak"]:
        env_value = True
    else:
        env_value = False
    return env_value
