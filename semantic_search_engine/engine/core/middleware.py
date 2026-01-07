"""
middleware.py
--------------

Utility that prepares authentication‑related middleware callbacks.  The
function inspects the project settings to decide which external auth
mechanisms (Keycloak, OAuth v1, OAuth v2) are enabled and, if any are
active, registers a callback that automatically creates an organisation,
organisation group and organisation‑user entry for newly‑authenticated
Django ``User`` instances.
"""

from django.conf import settings
from django.contrib.auth.models import User


def prepare_middleware_callback():
    """
    Initialise the global ``middleware_callback`` used by the external
    ``rdl_authorization`` package.

    The callback, when registered, adds a default organisation,
    organisation group and an ``OrganisationUser`` record for the Django
    user that has just been authenticated.  It is only installed when at
    least one of the supported auth back‑ends is enabled in the project
    settings.
    """
    auths = [
        settings.SYSTEM_HANDLER.use_kc_auth,
        settings.SYSTEM_HANDLER.use_oauth_v1_auth,
        settings.SYSTEM_HANDLER.use_oauth_v2_auth,
    ]

    if any(auths):
        from rdl_authorization.core import middleware
        from system.controllers import SystemController
        from rdl_authorization.utils.config import RdlAuthConfig

        if middleware.middleware_callback is None:
            if settings.SYSTEM_HANDLER.use_introspect:

                def __check_add_user_organisation(user: User):
                    """
                    Callback that creates the default organisation hierarchy
                    for the given ``user`` if it does not already exist.

                    Parameters
                    ----------
                    user : User
                        Django authentication user instance.
                    """
                    if user is None:
                        return
                    sc = SystemController()
                    aut_cfg = RdlAuthConfig()

                    organisation = sc.add_organisation(
                        name=aut_cfg.default_organisation["name"],
                        description=aut_cfg.default_organisation["description"],
                    )

                    org_group = sc.add_organisation_group(
                        organisation=organisation,
                        name=aut_cfg.default_group["name"],
                        description=aut_cfg.default_group["description"],
                    )

                    _ = sc.add_organisation_user(
                        name=user.username,
                        email=user.email,
                        password="DEFAULT_USER_PASS",
                        organisation=organisation,
                        user_groups=[org_group],
                    )

                middleware.middleware_callback = __check_add_user_organisation


prepare_middleware_callback()
