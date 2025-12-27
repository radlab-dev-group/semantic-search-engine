import os
import time
import django
import logging
import argparse
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from data.models import (
    Document,
    DocumentPage,
    DocumentPageText,
    CollectionOfDocuments,
)
from data.controllers.relational_db import PublicRelationDBController


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--collection-id", dest="collection_id", required=True)
    p.add_argument(
        "--deletion-wait-time",
        dest="deletion_wait_time",
        required=False,
        default=10,
        type=int,
    )
    return p


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    collection = PublicRelationDBController.get_collection_by_id(
        collection_id=args.collection_id
    )
    if not collection:
        raise Exception(f"Couldn't find the collection {args.collection_id}")

    all_documents_qs = Document.objects.filter(collection=collection)
    all_pages_qs = DocumentPage.objects.filter(document__collection=collection)
    all_texts_qs = DocumentPageText.objects.filter(
        page__document__collection=collection
    )

    logging.info(
        f"Collection: "
        f"\n - name            : {collection.name} "
        f"\n - display_name    : {collection.display_name} "
        f"\n - description     : {collection.description} "
        f"\n - texts count     : {len(all_texts_qs)} "
        f"\n - pages count     : {len(all_pages_qs)} "
        f"\n - documents count : {len(all_documents_qs)} "
        f"\nwill be removed. Accept manually decision to drop collection."
    )

    decision = input("Do you really want to delete this collection? Y/n: ")
    if decision != "Y":
        logging.info("Deleting collection is cancelled by the user")
        return

    logging.info(f"Collection will be deleted in {args.deletion_wait_time} seconds")
    seconds = [x for x in range(0, args.deletion_wait_time)]
    with tqdm(desc="Press ctr+c to cancel deletion...", total=len(seconds)) as pbar:
        for _ in seconds:
            time.sleep(1)
            pbar.update()

    logging.info(f"Starting process of deletion {collection.name} in 5 seconds")
    time.sleep(5)

    delete_tables = [all_texts_qs, all_pages_qs, all_documents_qs, collection]
    with tqdm(desc="Deleting db rows", total=len(delete_tables)) as pbar:
        for table in delete_tables:
            table.delete()


if __name__ == "__main__":
    main()
