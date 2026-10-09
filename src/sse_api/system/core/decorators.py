from functools import wraps
from rest_framework.request import Request
from rest_framework.exceptions import PermissionDenied

from sse_api.system.controllers import SystemController


def get_organisation_user(method):
    """
    Returns OrganisationUser for auth user
    :param method: Method to wrap
    :return:
    """

    @wraps(method)
    def _check_organisation_user(self, *method_args, **method_kwargs):
        organisation_user = None
        for req_arg in method_args:
            if type(req_arg) is Request:
                organisation_user = SystemController.get_organisation_user(
                    req_arg.user.username
                )
        if organisation_user is None:
            raise PermissionDenied("Organisation profile required.")
        return method(self, organisation_user, *method_args, **method_kwargs)

    return _check_organisation_user
