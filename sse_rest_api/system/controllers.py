import json

from typing import List
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password

from system.models import Organisation, OrganisationGroup, OrganisationUser


class UserConfigReader:
    def __init__(self, config_path: str = "configs/default-user.json"):
        self.config_path = config_path

        self._organisation = None
        self._group = None
        self._users = None

        self.__load_config()

    @property
    def organisation(self):
        return self._organisation

    @property
    def group(self):
        return self._group

    @property
    def users(self):
        return self._users

    def __load_config(self):
        with open(self.config_path) as config_file:
            whole_config = json.load(config_file)
        self._users = whole_config["users"]
        self._group = whole_config["group"]
        self._organisation = whole_config["organisation"]


class SystemController:
    def __init__(self):
        pass

    @staticmethod
    def add_organisation(name: str, description: str) -> Organisation:
        org, created = Organisation.objects.get_or_create(name=name)
        if created:
            org.description = description
            org.save()
        return org

    @staticmethod
    def add_organisation_group(
        organisation: Organisation, name: str, description: str
    ) -> OrganisationGroup:
        org_group, created = OrganisationGroup.objects.get_or_create(
            group_name=name, organisation=organisation
        )
        if created:
            org_group.description = description
            org_group.save()
        return org_group

    @staticmethod
    def add_organisation_user(
        name: str,
        email: str,
        password: str,
        organisation: Organisation,
        user_groups: List[OrganisationGroup],
    ) -> OrganisationUser:
        user, created = User.objects.get_or_create(username=name)

        if created:
            user.email = email
            user.password = make_password(password)
            user.is_active = True
            user.is_staff = False
            user.is_superuser = False
            user.save()

        org_user, created = OrganisationUser.objects.get_or_create(
            auth_user=user, organisation=organisation
        )
        if created:
            for user_group in user_groups:
                org_user.user_groups.add(user_group)

        return org_user

    @staticmethod
    def get_organisation_user(username: str) -> OrganisationUser | None:
        """
        Based on the given branch username find SystemUser
        :param username: organisation username
        :return:
        """
        try:
            return OrganisationUser.objects.get(auth_user__username=username)
        except OrganisationUser.DoesNotExist:
            return None

    @staticmethod
    def get_organisation(organisation_name: str) -> Organisation | None:
        """
        Finds Organisation based on the given organisation name.
        :param organisation_name: organisation name
        :return: Organisation or None
        """
        try:
            return Organisation.objects.get(name=organisation_name)
        except OrganisationUser.DoesNotExist:
            return None

    @staticmethod
    def get_organisation_group(
        organisation: Organisation, group_name: str
    ) -> OrganisationGroup or None:
        try:
            return OrganisationGroup.objects.get(
                organisation=organisation, group_name=group_name
            )
        except OrganisationGroup.DoesNotExist:
            return None
