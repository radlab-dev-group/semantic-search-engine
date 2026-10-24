"""
DRF throttle classes of the SSE API.

The defaults live in ``config.settings.REST_FRAMEWORK``; this module only
holds the classes, so a test can plug in a tight rate without importing the
deployment settings.

* :class:`BurstAnonRateThrottle` — anonymous clients, per IP.  The budget is
  deliberately small: an unauthenticated request cannot be attributed to a
  user, so it must not be able to burn the embedding/LLM budget of the whole
  deployment.
* :class:`BurstUserRateThrottle` — authenticated clients, per user.
* :class:`StrictAuthThrottle` — the login endpoints only.  A tight budget per
  IP is what stands between a script and a user's password.
"""

from rest_framework.throttling import SimpleRateThrottle


class BurstAnonRateThrottle(SimpleRateThrottle):
    scope = "anon"

    def get_cache_key(self, request, view):
        ident = request.META.get("REMOTE_ADDR", "unknown")
        return self.cache_key_format.format(self.scope, ident)


class BurstUserRateThrottle(SimpleRateThrottle):
    scope = "user"

    def get_cache_key(self, request, view):
        if request.user and request.user.is_authenticated:
            ident = request.user.pk
        else:
            ident = request.META.get("REMOTE_ADDR", "unknown")
        return self.cache_key_format.format(self.scope, ident)


class StrictAuthThrottle(SimpleRateThrottle):
    scope = "auth"

    def get_cache_key(self, request, view):
        ident = request.META.get("REMOTE_ADDR", "unknown")
        return self.cache_key_format.format(self.scope, ident)
