app_name = "authorization"

default_token_model = f"{app_name}.models"
default_app_config = f"{app_name}.apps.RdlAuthConfig"
default_auth_class = f"{app_name}.core.authentication.GATokenAuthentication"
default_middleware_class = f"{app_name}.core.middleware.AuthAuthenticationMiddleware"
