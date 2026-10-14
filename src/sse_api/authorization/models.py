from django.db import models
from django.contrib.auth.models import User


class SessionState(models.Model):
    """
    Generated session state
    """

    state = models.TextField(primary_key=True)
    session_state = models.TextField(null=True)
    grant_code = models.TextField(null=True)
    state_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.state


class Token(models.Model):
    """
    KeyCloack granted tokens
    """

    acc_token = models.TextField(null=False)
    is_active = models.BooleanField(default=False)
    acc_created = models.DateTimeField(auto_now_add=True)

    # Additional options which are connected with created token
    not_before_policy = models.IntegerField(null=True)
    expires_in = models.IntegerField(null=True)
    refresh_expires_in = models.IntegerField(null=True)
    token_type = models.TextField(null=True)
    refresh_token = models.TextField(null=True)
    scope = models.TextField(null=True)

    state = models.ForeignKey(SessionState, on_delete=models.CASCADE, null=False)
    auth_user = models.ForeignKey(User, on_delete=models.PROTECT, null=False)

    decoded_token = models.JSONField(null=True)

    def __str__(self):
        return self.acc_token
