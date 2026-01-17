from django.db import models
from django.contrib.auth.models import User


class Organisation(models.Model):
    """
    General Organisation definition
    """

    name = models.TextField(primary_key=True)
    description = models.TextField(null=True)


class OrganisationGroup(models.Model):
    """
    All organisation groups
    """

    # pk: auto id
    group_name = models.TextField(null=False)
    description = models.TextField(null=True)

    organisation = models.ForeignKey(
        Organisation, on_delete=models.PROTECT, null=False
    )

    class Meta:
        unique_together = ("group_name", "organisation")


class OrganisationUser(models.Model):
    """
    Each organisation user
    """

    auth_user = models.ForeignKey(User, on_delete=models.PROTECT, null=False)

    organisation = models.ForeignKey(
        Organisation, on_delete=models.PROTECT, null=False
    )
    user_groups = models.ManyToManyField(OrganisationGroup)
