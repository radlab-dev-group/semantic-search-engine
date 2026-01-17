import logging
import os
import django
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from data.controllers import DBSemanticSearchController


def prepare_parser(desc=""):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-f",
        "--from-collection",
        dest="from_collection_name",
        required=True,
        type=str,
        help="Collection name from text db to read texts",
    )

    parser.add_argument(
        "-t",
        "--to-collection",
        dest="to_collection_name",
        required=True,
        type=str,
        help="Collection name to index texts into semantic database",
    )

    parser.add_argument(
        "-c",
        "--chunk-type",
        dest="chunk_type",
        required=True,
        type=str,
        help="Chunk type to index in semantic db",
    )
    parser.add_argument(
        "-I",
        "--index-name",
        dest="index_name",
        required=True,
        type=str,
        help="Name of index specification, available indexes {IVF_FLAT, HNSW}",
    )

    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        dest="batch_size",
        help="Number of text used in single batch when loading to database",
    )

    return parser


def main(argv=None):
    logging.basicConfig(
        format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
    )

    args = prepare_parser().parse_args(argv)
    semantic_db_controller = DBSemanticSearchController(
        collection_name=args.to_collection_name,
        batch_size=args.batch_size,
        index_name=args.index_name,
    )

    semantic_db_controller.index_texts(
        from_collection=args.from_collection_name, chunk_type=args.chunk_type
    )


if __name__ == "__main__":
    main()
