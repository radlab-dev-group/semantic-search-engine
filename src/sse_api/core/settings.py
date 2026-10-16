import os
import copy
import json
import logging
import logging.config


from sse_api.core.constants import (
    AVAILABLE_CONFIG_DIRS,
    ENV_INSTALLED_APPS,
    ENV_ALLOWED_HOSTS,
    ENV_SECRET_KEY,
    ENV_DEBUG,
    ENV_SHOW_ADMIN_WINDOW,
    ENV_PUBLIC_API_AVAILABLE,
    ENV_DB_NAME,
    ENV_DB_HOST,
    ENV_DB_PORT,
    ENV_DB_ENGINE,
    ENV_DB_USERNAME,
    ENV_DB_PASSWORD,
    ENV_LOGGER,
    ENV_EXTERNAL_API,
    ENV_USE_KC_AUTH,
    ENV_USE_OAUTH_V1_AUTH,
    ENV_USE_OAUTH_V2_AUTH,
    ENV_USE_AWS,
    ENV_USE_CELERY,
    ENV_DEFAULT_LANGUAGE,
    SYSTEM_CONFIG_FILENAME,
    SYSTEM_CONFIG_FILENAME_DEFAULT,
    SECRET_KEY_FILENAME,
    SECRET_KEY_FILENAME_DEFAULT,
    ENV_USE_INTROSPECT,
    ENV_INTROSPECT_EMAIL_VERIFICATION,
    ENV_CELERY_BROKER_URL,
    ENV_CELERY_RESULT_BACKEND,
    ENV_CELERY_TASK_TIME_LIMIT,
    CONFIG_DIR,
    ENV_USE_RADLAB_PROXY,
    ENV_USE_RADLAB_PROXY_SEE,
)

from .env_utils import (
    bool_config_value,
    bool_env_value,
    list_env_values,
    split_list_value,
)


class SystemSettingsHandler:
    """
    Handler for system settings
    """

    system_cfg_filename = SYSTEM_CONFIG_FILENAME
    default_system_cfg_filename = SYSTEM_CONFIG_FILENAME_DEFAULT

    secret_key_filename = SECRET_KEY_FILENAME
    default_secret_key_filename = SECRET_KEY_FILENAME_DEFAULT

    def __init__(
        self,
        system_cfg_filename: str = None,
        secret_key_filename: str = None,
        load_config: bool = True,
        use_environment: bool = True,
    ):
        """
        :param system_cfg_filename: Path to system config file
        :param secret_key_filename:  Path to secret key filename
        """
        self.json_config = None

        if system_cfg_filename:
            self._check_django_config_exists(system_cfg_filename)
            self.system_cfg_filename = system_cfg_filename

        if secret_key_filename:
            self._check_secret_key_exists(secret_key_filename)
            self.secret_key_filename = secret_key_filename

        self.use_environment = use_environment

        if load_config:
            self.load_system_config()

        self.use_celer_tasks = self.use_celery()
        self.use_aws_storage = self.use_aws()
        self.use_kc_auth = self.use_keycloak_authentication()
        self.use_oauth_v1_auth = self.use_oauth_v1_authentication()
        self.use_oauth_v2_auth = self.use_oauth_v2_authentication()
        self.use_introspect = self.use_authorization_introspect()
        self.introspect_email_verification = (
            self.introspect_check_user_email_is_verified()
        )

        self._last_logger = None
        self._aws_handler = None

        if self.use_aws_storage:
            self.__prepare_aws_handler()

    @property
    def aws_handler(self):
        return self._aws_handler

    def load_secret_key(self, filepath: str = None):
        """
        Load and return secret key.

        :param filepath: Path to file with secret key
        :return: result of function to load secret key
        """
        self._properly_config_read()
        sk = self._load_secret_key_from_file(filepath)
        if self.use_environment:
            sk_e = self._load_secret_key_from_env(ENV_SECRET_KEY)
            if sk_e is not None and len(sk_e.strip()):
                sk = sk_e
        if sk is None:
            self._no_param_def_exception("secret_key", ENV_SECRET_KEY)
        return sk

    def database(self):
        """
        Return database configuration. Depends on `from_config` argument value,
        database is read from json file or from environment.
        :return: Database configuration
        """
        db_config = self._get_configuration_for_variable(
            sec_name="database",
            variable_name=None,
            mapping_function=None,
            env_names=[
                ENV_DB_NAME,
                ENV_DB_HOST,
                ENV_DB_PORT,
                ENV_DB_ENGINE,
                ENV_DB_USERNAME,
                ENV_DB_PASSWORD,
            ],
        )
        return {"default": db_config}

    def allowed_hosts(self):
        """
        Return allowed hosts configuration. Depends on `from_config` argument value,
        allowed host is read from json file or from environment.
        :return: allowed hosts configuration
        """
        if (
            self.use_environment
            and ENV_ALLOWED_HOSTS in os.environ
            and os.getenv(ENV_ALLOWED_HOSTS, "").strip()
        ):
            return list_env_values(ENV_ALLOWED_HOSTS)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return split_list_value(
                self.json_config["main"].get("allowed_hosts", [])
            )
        return []

    def get_aws_config_value(self, aws_section: str) -> str:
        return self._get_configuration_for_variable(
            sec_name="aws",
            variable_name=aws_section,
            mapping_function=aws_section,
            env_names=f"ENV_AWS_{aws_section}".upper(),
        )

    def celery_broker_url(self):
        """
        Return celery broker url
        :return: URL
        """
        if (
            self.use_environment
            and ENV_CELERY_BROKER_URL in os.environ
            and os.getenv(ENV_CELERY_BROKER_URL, "").strip()
        ):
            return os.getenv(ENV_CELERY_BROKER_URL).strip()

        self._properly_config_read()
        if self.json_config and "celery" in self.json_config:
            return self.json_config["celery"].get("broker_url")
        return None

    def celery_result_backend(self):
        """
        Return celery result backend url (``None`` keeps the results out of
        the backend: the progress of the indexing lives in the relational
        database and is read from there by the API).
        :return: URL or None
        """
        if (
            self.use_environment
            and ENV_CELERY_RESULT_BACKEND in os.environ
            and os.getenv(ENV_CELERY_RESULT_BACKEND, "").strip()
        ):
            return os.getenv(ENV_CELERY_RESULT_BACKEND).strip()

        self._properly_config_read()
        if self.json_config and "celery" in self.json_config:
            return self.json_config["celery"].get("result_backend")
        return None

    def celery_task_time_limit(self, default: int = 14400) -> int:
        """
        Hard time limit of one task, in seconds (default: 4 hours).
        :return: seconds
        """
        if (
            self.use_environment
            and ENV_CELERY_TASK_TIME_LIMIT in os.environ
            and os.getenv(ENV_CELERY_TASK_TIME_LIMIT, "").strip()
        ):
            return int(os.getenv(ENV_CELERY_TASK_TIME_LIMIT).strip())

        self._properly_config_read()
        if self.json_config and "celery" in self.json_config:
            return int(
                self.json_config["celery"].get("task_time_limit", default) or default
            )
        return default

    def installed_apps(self):
        """
        Return installed apps configuration. Depends on `from_config` argument value,
        installed apps is read from json file or from environment.
        :return: installed apps configuration
        """
        if (
            self.use_environment
            and ENV_INSTALLED_APPS in os.environ
            and os.getenv(ENV_INSTALLED_APPS, "").strip()
        ):
            return list_env_values(ENV_INSTALLED_APPS)

        self._properly_config_read()
        if self.json_config:
            return split_list_value(self.json_config.get("installed_apps", []))
        return []

    def debug(self):
        """
        Return debug value.
        :return: Database configuration
        """
        if (
            self.use_environment
            and ENV_DEBUG in os.environ
            and os.getenv(ENV_DEBUG, "").strip()
        ):
            return bool_env_value(ENV_DEBUG)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(self.json_config["main"].get("debug", False))
        return False

    def use_keycloak_authentication(self):
        """
        Return boolean value to use_keycloak_authentication
        :return: Boolean value to use_keycloak_authentication
        """
        if (
            self.use_environment
            and ENV_USE_KC_AUTH in os.environ
            and os.getenv(ENV_USE_KC_AUTH, "").strip()
        ):
            return bool_env_value(ENV_USE_KC_AUTH)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("use_keycloak_authentication", False)
            )
        return False

    def use_oauth_v2_authentication(self):
        """
        Return boolean value to use_oauth_v2_authentication
        :return: Boolean value to use_oauth_v2_authentication
        """
        if (
            self.use_environment
            and ENV_USE_OAUTH_V2_AUTH in os.environ
            and os.getenv(ENV_USE_OAUTH_V2_AUTH, "").strip()
        ):
            return bool_env_value(ENV_USE_OAUTH_V2_AUTH)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("use_oauth_v2_authentication", False)
            )
        return False

    def use_oauth_v1_authentication(self):
        """
        Return boolean value to use_oauth_v2_authentication
        :return: Boolean value to use_oauth_v2_authentication
        """
        if (
            self.use_environment
            and ENV_USE_OAUTH_V1_AUTH in os.environ
            and os.getenv(ENV_USE_OAUTH_V1_AUTH, "").strip()
        ):
            return bool_env_value(ENV_USE_OAUTH_V1_AUTH)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("use_oauth_v1_authentication", False)
            )
        return False

    def use_celery(self):
        """
        Return boolean value to use_celery.
        :return: Database configuration
        """
        if (
            self.use_environment
            and ENV_USE_CELERY in os.environ
            and os.getenv(ENV_USE_CELERY, "").strip()
        ):
            return bool_env_value(ENV_USE_CELERY)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("use_celery", False)
            )
        return False

    def use_aws(self):
        """
        Return boolean value to use_celery.
        :return: Database configuration
        """
        if (
            self.use_environment
            and ENV_USE_AWS in os.environ
            and os.getenv(ENV_USE_AWS, "").strip()
        ):
            return bool_env_value(ENV_USE_AWS)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(self.json_config["main"].get("use_aws", False))
        return False

    @staticmethod
    def use_radlab_proxy():
        """
        Returns boolean value which determines whether to use proxy or not
        :return: Boolean value of using radlab proxy
        """
        if ENV_USE_RADLAB_PROXY in os.environ:
            return bool_env_value(ENV_USE_RADLAB_PROXY)
        return False

    @staticmethod
    def use_radlab_proxy_sse():
        """
        Returns boolean value which determines whether to use proxy-sse or not
        :return: Boolean value of using radlab proxy-see
        """
        if ENV_USE_RADLAB_PROXY_SEE in os.environ:
            return bool_env_value(ENV_USE_RADLAB_PROXY_SEE)
        return False

    def use_authorization_introspect(self):
        """
        Return retrospect value.
        :return: Boolean value of using retrospection
        """
        if (
            self.use_environment
            and ENV_USE_INTROSPECT in os.environ
            and os.getenv(ENV_USE_INTROSPECT, "").strip()
        ):
            return bool_env_value(ENV_USE_INTROSPECT)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("use_authorization_introspect", False)
            )
        return False

    def introspect_check_user_email_is_verified(self):
        if (
            self.use_environment
            and ENV_INTROSPECT_EMAIL_VERIFICATION in os.environ
            and os.getenv(ENV_INTROSPECT_EMAIL_VERIFICATION, "").strip()
        ):
            return bool_env_value(ENV_INTROSPECT_EMAIL_VERIFICATION)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("introspect_email_verification", False)
            )
        return False

    def show_admin_window(self):
        """
        Return boolean value if admin_window will be enabled.
        :return: Decision if admin window should be showed
        """
        if (
            self.use_environment
            and ENV_SHOW_ADMIN_WINDOW in os.environ
            and os.getenv(ENV_SHOW_ADMIN_WINDOW, "").strip()
        ):
            return bool_env_value(ENV_SHOW_ADMIN_WINDOW)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("admin_window", False)
            )
        return False

    def is_public_api_available(self):
        """
        Return boolean value if public api have to be available.
        :return: Decision about public api availability
        """
        if (
            self.use_environment
            and ENV_PUBLIC_API_AVAILABLE in os.environ
            and os.getenv(ENV_PUBLIC_API_AVAILABLE, "").strip()
        ):
            return bool_env_value(ENV_PUBLIC_API_AVAILABLE)

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return bool_config_value(
                self.json_config["main"].get("public_api_available", True)
            )
        return True

    def logger(self):
        """
        Return logger build based on the configuration
        :return: Object of build and configured logger
        """
        which_logger = None
        if (
            self.use_environment
            and ENV_LOGGER in os.environ
            and os.getenv(ENV_LOGGER, "").strip()
        ):
            which_logger = os.getenv(ENV_LOGGER).strip()
        elif self.json_config and "logger" in self.json_config:
            which_logger = self.json_config["logger"].get("use_logger")

        if which_logger is None:
            which_logger = "prod"

        if which_logger == self._last_logger:
            return logging

        self._properly_config_read()
        logger_config = copy.deepcopy(self.json_config["logger"]["config"])
        logger_config["root"] = logger_config[which_logger]
        logging.config.dictConfig(logger_config)

        self._last_logger = which_logger

        return logging

    def get_external_api(self, api_name: str) -> dict:
        """
        Return external api configuration based on the api_name
        :param api_name: Api name from config to read
        :return: Api configuration as dict
        """
        external_api = self._get_configuration_for_variable(
            sec_name="external_api",
            variable_name=api_name,
            mapping_function="__dynamic_json_read__",
            env_names=ENV_EXTERNAL_API,
            env_subname=api_name,
        )
        if type(external_api) in [dict]:
            return external_api["host"]
        return external_api

    def default_language(self):
        """
        Return default language
        :return:
        """
        if (
            self.use_environment
            and ENV_DEFAULT_LANGUAGE in os.environ
            and os.getenv(ENV_DEFAULT_LANGUAGE, "").strip()
        ):
            return os.getenv(ENV_DEFAULT_LANGUAGE).strip()

        self._properly_config_read()
        if self.json_config and "main" in self.json_config:
            return self.json_config["main"].get("default_language", "pl")
        return "pl"

    def api_version_url(self):
        """
        Returns versioned base api url (without any endpoint)
        :return: main url with (optionally) the version
        """
        api_configuration = self._get_configuration_for_variable(
            sec_name="main",
            variable_name="api",
            mapping_function="api",
            env_names=None,
        )

        root_url = api_configuration.get("root_url", "api")
        major_version = str(api_configuration.get("major_version", "")).strip()
        minor_version = str(api_configuration.get("minor_version", "")).strip()
        rc_version = str(api_configuration.get("release_candidate", "")).strip()

        if not root_url or not len(root_url):
            raise Exception(
                "Root url (root_url field in config file) of api is required!"
            )

        version_url = None
        if major_version and len(major_version):
            version_url = f"v{major_version}"
            if minor_version and len(minor_version):
                version_url += f".{minor_version}"
            if rc_version and len(rc_version):
                version_url += f"_{rc_version}"

        version_api_url = root_url
        if version_url and len(version_url):
            version_api_url += f"/{version_url}"

        return version_api_url.strip()

    def load_system_config(self, config_path: str = None):
        """
        Load config from given json config path
        :param config_path: The path to config
        :return: Read config as dictionary
        """
        if config_path is None:
            config_path = self.system_cfg_filename

        with open(config_path, "rt") as fin:
            self.json_config = json.load(fin)
        return self.json_config

    @staticmethod
    def _load_secret_key_from_env(env_name: str = None):
        """
        Load secret key from given environment variable name
        :param env_name: The name of environment variable with secret key
        :return: Read secret key
        """
        if env_name is None:
            env_name = ENV_SECRET_KEY
        return os.getenv(env_name)

    def _load_secret_key_from_file(self, filepath: str):
        """
        Load secret key from given file path
        :param filepath: The path to file with secret key
        :return: Read secret key
        """
        if filepath is None:
            filepath = self.secret_key_filename

        if filepath is None or not os.path.exists(filepath):
            return None

        with open(filepath, "rt") as fin:
            return fin.read().strip()

    def _check_django_config_exists(self, config_path):
        return self._check_file_exists(config_path)

    def _check_kc_config_exists(self, config_path):
        return self._check_file_exists(config_path)

    def _check_secret_key_exists(self, secret_key_filepath):
        return self._check_file_exists(secret_key_filepath)

    def _check_file_exists(self, config_path):
        if not os.path.exists(config_path):
            if os.path.exists(self.default_system_cfg_filename):
                raise Exception(
                    f"Only default config file exists "
                    f"{self.default_system_cfg_filename}.\n"
                    f"Based on this file, prepare your own configuration!"
                )
            for cdir in AVAILABLE_CONFIG_DIRS:
                if os.path.exists(
                    os.path.join(cdir, self.default_system_cfg_filename)
                ):
                    raise Exception(
                        f"Cannot find file {config_path}! "
                        f"Only default config file exists "
                        f"{self.default_system_cfg_filename}.\n"
                        f"Based on this file, prepare your own configuration!"
                    )
            raise Exception(f"Cannot find file {config_path}")

    def _no_param_def_exception(self, param_name, env_param_names):
        raise Exception(
            f"Cannot find definition of parameter {param_name}."
            f"Update your json config file or export system environment value "
            f"named as {env_param_names}"
        )

    def _properly_config_read(self):
        if self.json_config is None and not self.use_environment:
            raise Exception("Config file have to be read!")

    def _get_configuration_for_variable(
        self, sec_name, variable_name, mapping_function, env_names, env_subname=None
    ):
        """
        General method to check param and get settings from env/config for param
        :param sec_name: Section name from config
        :param variable_name: Variable name to read
        :param mapping_function: One of {None, __dynamic_json_read__}
        :param env_names: List of env names or single env name
        :param env_subname: Subname of env (works only for single env)
        :return:
        """
        if env_subname is not None:
            env_subname = str(env_subname).strip()
            env_subname = env_subname.upper() if len(env_subname) else None
        config_variable = None
        self._properly_config_read()
        if self.json_config:
            if sec_name is not None and variable_name is not None:
                config_variable = self.json_config.get(sec_name, {}).get(
                    variable_name
                )
            elif sec_name is not None:
                config_variable = self.json_config.get(sec_name)

        if self.use_environment and env_names is not None:
            if sec_name and "database" in sec_name.lower():
                return self._db_config_from_env(env_names, config_variable)
            else:
                if isinstance(env_names, list):
                    if env_subname is not None:
                        raise Exception(
                            "Env subname is available only for single environment!"
                        )
                    o_env_values = []
                    for e in env_names:
                        e_val = os.getenv(e)
                        if e_val is not None and len(e_val.strip()):
                            o_env_values.extend(split_list_value(e_val))
                    if len(o_env_values):
                        return o_env_values
                else:
                    if env_subname is not None:
                        env_names = f"{env_names}_{env_subname}"
                    e_val = os.getenv(env_names)
                    if e_val is not None and len(e_val.strip()):
                        if env_subname is not None:
                            return e_val.strip()
                        if mapping_function == "__dynamic_json_read__":
                            return self._get_configuration_for_variable(
                                sec_name, e_val.strip(), None, env_names
                            )
                        return e_val.strip()

        if config_variable is None:
            self._no_param_def_exception(
                param_name=variable_name, env_param_names=env_names
            )
        return config_variable

    @staticmethod
    def _db_config_from_env(env_names, config_variable):
        config_variable = copy.deepcopy(config_variable) if config_variable else {}
        for env_name in env_names:
            e_val = os.getenv(env_name)
            if e_val is None or not len(e_val.strip()):
                continue
            e_val = e_val.strip()
            if env_name == ENV_DB_NAME:
                config_variable["NAME"] = e_val
            elif env_name == ENV_DB_HOST:
                config_variable["HOST"] = e_val
            elif env_name == ENV_DB_PORT:
                config_variable["PORT"] = int(e_val) if e_val.isdigit() else e_val
            elif env_name == ENV_DB_ENGINE:
                config_variable["ENGINE"] = e_val
            elif env_name == ENV_DB_USERNAME:
                config_variable["USER"] = e_val
            elif env_name == ENV_DB_PASSWORD:
                config_variable["PASSWORD"] = e_val
        return config_variable

    def __prepare_aws_handler(self):
        assert self.json_config is not None, "JSON Config cannot be None!"

        from .aws_handler import AwsHandler

        self._aws_handler = AwsHandler(prepare=True)


system_handler = SystemSettingsHandler(
    system_cfg_filename=str(
        os.path.join(CONFIG_DIR, SystemSettingsHandler.system_cfg_filename)
    ),
    secret_key_filename=str(
        os.path.join(CONFIG_DIR, SystemSettingsHandler.secret_key_filename)
    ),
    load_config=True,
    use_environment=True,
)
