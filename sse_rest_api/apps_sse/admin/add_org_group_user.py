import os
import json
import django
import logging
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()


from system.controllers import SystemController, UserConfigReader


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("-u", "--user-config", dest="user_config", required=True)
    return p


def main(argv=None):
    args = prepare_parser().parse_args()
    uc_reader = UserConfigReader(config_path=args.user_config)

    print(50 * "- ")
    print("ORGANISATION:")
    print(json.dumps(uc_reader.organisation, indent=2, ensure_ascii=False))
    print(50 * "- ")
    print("GROUP:")
    print(json.dumps(uc_reader.group, indent=2, ensure_ascii=False))
    print(50 * "- ")
    print("USERS:")
    print(json.dumps(uc_reader.users, indent=2, ensure_ascii=False))

    sc = SystemController()
    logging.info(f"Adding organisation {uc_reader.organisation['name']}")
    organisation = sc.add_organisation(
        name=uc_reader.organisation["name"],
        description=uc_reader.organisation["description"],
    )

    logging.info(f"Adding organisation group {uc_reader.group['name']}")
    org_group = sc.add_organisation_group(
        organisation=organisation,
        name=uc_reader.group["name"],
        description=uc_reader.group["description"],
    )

    for user in uc_reader.users:
        logging.info(f"Adding user {user['name']}")
        _ = sc.add_organisation_user(
            name=user["name"],
            email=user["email"],
            password=user["password"],
            organisation=organisation,
            user_groups=[org_group],
        )


if __name__ == "__main__":
    main()
