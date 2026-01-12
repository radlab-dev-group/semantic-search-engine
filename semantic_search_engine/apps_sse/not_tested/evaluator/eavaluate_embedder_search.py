import os
import django
import logging
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from apps_sse.evaluator.src.tests_loader import TestsLoader


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("-u", "--username", required=True, type=str, help="User name")
    p.add_argument(
        "--test-configuration",
        dest="test_configuration",
        required=True,
        type=str,
        help="Test configuration (json file)",
    )
    p.add_argument(
        "-o",
        "--out-xlsx-file",
        dest="out_xlsx_file",
        required=True,
        type=str,
        help="The output xlsx file",
    )
    return p


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)
    user = SystemController.get_organisation_user(username=args.username)
    if not user:
        raise SystemExit(f"User {args.username} not found")
    logging.info(f"User {args.username} will be used as test user.")

    test_loader = TestsLoader(
        test_user=user,
        json_path=args.test_configuration,
        load_tests=True,
        verify_collections=True,
    )
    if args.out_xlsx_file.endswith("xlsx"):
        out_xlsx_chunks_file = args.out_xlsx_file.replace(".xlsx", "-chunks.xlsx")
    else:
        out_xlsx_chunks_file = args.out_xlsx_file + "-chunks.xlsx"

    test_loader.start(
        out_xlsx_file=args.out_xlsx_file, out_xlsx_chunks_file=out_xlsx_chunks_file
    )


if __name__ == "__main__":
    main()
