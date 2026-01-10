import json


class RdlAuthConfig:
    """
    RdlAuth config parser/reader/handler
    """

    RDL_AUTH_JSON_FIELD = "authorization"

    def __init__(self, cfg_path: str = "configs/auth-config.json"):
        self._auth_host = None
        self._realm = None
        self._client_secret = None
        self._client_id = None
        self._grant_type = None
        self._redirect_uri = None
        self._public_sign_key = None
        self._scope = None
        self._audience = None
        self._algorithms = []

        self._accepted_user_roles = None
        self._user_role_main_scope = None
        self._user_role_scope_variable = None

        self._rdl_auth_config = None
        self._rdl_auth_json_config_path = cfg_path

        self._default_group = None
        self._default_organisation = None

        self.load_config()

    @property
    def auth_host(self):
        return self._auth_host

    @property
    def realm(self):
        return self._realm

    @property
    def client_secret(self):
        return self._client_secret

    @property
    def client_id(self):
        return self._client_id

    @property
    def grant_type(self):
        return self._grant_type

    @property
    def redirect_uri(self):
        return self._redirect_uri

    @property
    def public_sign_key(self):
        return self._public_sign_key

    @property
    def audience(self):
        return self._audience

    @property
    def scope(self):
        return self._scope

    @property
    def algorithms(self):
        return self._algorithms

    @property
    def accepted_user_roles(self):
        return self._accepted_user_roles

    @property
    def user_role_main_scope(self):
        return self._user_role_main_scope

    @property
    def user_role_scope_variable(self):
        return self._user_role_scope_variable

    @property
    def default_organisation(self):
        return self._default_organisation

    @property
    def default_group(self):
        return self._default_group

    def load_config(self, cfg_path: str | None = None):
        if cfg_path is not None:
            self._rdl_auth_json_config_path = cfg_path
        assert self._rdl_auth_json_config_path is not None

        with open(self._rdl_auth_json_config_path, "rt") as kc_f:
            self._rdl_auth_config = json.load(kc_f)[self.RDL_AUTH_JSON_FIELD]

            self._realm = self._rdl_auth_config["realm"]
            self._client_secret = self._rdl_auth_config["client_secret"]
            self._client_id = self._rdl_auth_config["client_id"]
            self._grant_type = self._rdl_auth_config["grant_type"]
            self._redirect_uri = self._rdl_auth_config["redirect_uri"]
            self._auth_host = self._rdl_auth_config["auth_host"].rstrip("/")

            self._public_sign_key = self._rdl_auth_config.get("public_sign_key", "")
            self._audience = self._rdl_auth_config.get("audience", "")
            self._scope = self._rdl_auth_config.get("scope", "")
            self._algorithms = self._rdl_auth_config.get("algorithms", [])

            self._accepted_user_roles = self._rdl_auth_config.get(
                "accepted_user_roles", []
            )
            self._user_role_main_scope = self._rdl_auth_config.get(
                "user_role_main_scope", ""
            )
            self._user_role_scope_variable = self._rdl_auth_config.get(
                "user_role_scope_variable", ""
            )
            self._default_organisation = self._rdl_auth_config.get(
                "default_organisation", None
            )
            self._default_group = self._rdl_auth_config.get("default_group", None)
