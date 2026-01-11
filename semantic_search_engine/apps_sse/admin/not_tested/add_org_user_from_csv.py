import os
import django
import logging
import argparse

import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from system.core.constants import (
    DEFAULT_ORGANISATION_NAME,
    DEFAULT_ORGANISATION_DESCRIPTION,
    DEFAULT_USER_GROUP_NAME,
    DEFAULT_USER_GROUP_DESCRIPTION,
    DEFAULT_USER_NAME,
    DEFAULT_USER_EMAIL,
    DEFAULT_USER_PASS,
)


def prepare_parser(desc=""):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-u",
        "--users-file",
        dest="users_file",
        required=True,
        help="Path to CSV file with defined users",
    )

    return parser


def main(argv=None):
    args = prepare_parser().parse_args(argv)
    sc = SystemController()

    users_df = pd.read_csv(args.users_file, sep="\t", encoding="utf8")
    for _, user_row in users_df.iterrows():
        org_name = user_row["ORGANISATION_NAME"]
        org_desc = user_row["ORGANISATION_DESCRIPTION"]
        user_g_name = user_row["USER_GROUP_NAME"]
        user_g_desc = user_row["USER_GROUP_DESCRIPTION"]
        user_name = user_row["USER_NAME"]
        user_email = user_row["USER_EMAIL"]
        user_pass = user_row["USER_PASS"]

        logging.info(f"Adding organisation {org_name}")
        organisation = sc.add_organisation(name=org_name, description=org_desc)

        logging.info(f"Adding group {user_g_name} for {org_name} organisation")
        org_group = sc.add_organisation_group(
            organisation=organisation,
            name=user_g_name,
            description=user_g_desc,
        )

        logging.info(f"Adding user {user_name}")
        _ = sc.add_organisation_user(
            name=user_name,
            email=user_email,
            password=user_pass,
            organisation=organisation,
            user_groups=[org_group],
        )


if __name__ == "__main__":
    main()
