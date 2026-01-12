import os
import django
import argparse
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from data.models import CollectionOfDocuments, DocumentPageText


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument(
        "--collection-id",
        dest="collection_id",
        required=True,
        help="DB Identifier of collection to be stored.",
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
    try:
        collection = CollectionOfDocuments.objects.get(pk=args.collection_id)
    except CollectionOfDocuments.DoesNotExist as e:
        print("ERROR WHILE FETCHING COLLECTION", args.collection_id)
        print(e)
        return

    pages_texts = DocumentPageText.objects.filter(
        page__document__collection=collection
    ).order_by("page__document", "page__page_number", "text_number")

    print("Converting to xlsx data...")
    xlsx_data = []

    with tqdm(total=len(pages_texts), desc="Converting to xlsx data") as pbar:
        for page_text in pages_texts:
            page_dict = {
                "document_name": page_text.page.document.name,
                "page_number": page_text.page.page_number,
                "text_number": page_text.text_number,
                "text_str": page_text.text_str,
            }
            xlsx_data.append(page_dict)
            pbar.update(1)

    print("Storing to xlsx file...")
    pd.DataFrame(xlsx_data).to_excel(args.out_xlsx_file, engine="xlsxwriter")


if __name__ == "__main__":
    main()
