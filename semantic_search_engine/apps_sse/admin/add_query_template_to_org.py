import os
import django
import logging
import argparse

import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from rdl_authorization.utils.config import RdlAuthConfig
from data.controllers.query_templates import QueryTemplatesLoaderController


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument(
        "--query-config",
        dest="query_config",
        default="./configs/query-templates.json",
        help="Path to query template json config file.",
    )

    p.add_argument(
        "--keycloak-config",
        dest="keycloak_config",
        default="./configs/auth-config.json",
        help="Path to keycloak config with defined organisation.",
    )

    return p


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    sys_controller = SystemController()
    kc_config = RdlAuthConfig(cfg_path=args.keycloak_config)
    qt_controller = QueryTemplatesLoaderController(config_path=args.query_config)

    organisation = sys_controller.get_organisation(
        organisation_name=kc_config.default_organisation["name"]
    )
    if organisation is None:
        raise Exception(
            f"Organisation {kc_config.default_organisation['name']} "
            f"not found in database!"
        )

    print(
        f"Found organisation {organisation.name} "
        f"(will be used to add query templates)"
    )

    query_template = qt_controller.add_templates_to_organisation(
        organisation=organisation
    )
    if query_template is None:
        print("No query templates were added.")
        return

    print("Added query template:", query_template)


if __name__ == "__main__":
    main()
