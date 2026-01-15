import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

import time
import logging

try:
    from apps_sse.installed import any_text_to_json as attj
except Exception:
    from apps_sse.installed import any_text_to_json as attj
finally:
    if attj is None:
        raise Exception("Cannot import installed.any_text_to_json application")

from data.controllers import TextDataDBController


def main(argv=None):
    logging.basicConfig(
        format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
    )

    p = attj.prepare_parser()
    p.add_argument(
        "-c",
        "--collection",
        dest="collection_name",
        required=True,
        type=str,
        help="Collection name of documents",
    )

    args = attj.check_args(p.parse_args(argv))

    tdc = TextDataDBController()

    ts = time.time()
    logging.info(f"Uploading files to destination dir {args.directory}")
    uploaded_docs = tdc.upload_documents_to_destination_dir(
        args.directory, dont_upload=True
    )

    logging.info(f"Loading files from directory {uploaded_docs.dir_path}")
    tdc.add_uploaded_documents_to_db(
        collection_name=args.collection_name,
        uploaded_document_hash=uploaded_docs.dir_hash,
        prepare_proper_pages=args.prepare_proper_pages,
        merge_document_pages=args.merge_document_pages,
        clear_texts=args.clear_texts,
        max_tokens_in_chunk=args.split_to_max_tokens_in_chunk,
        number_of_overlap_tokens=args.overlap_tokens_between_chunks,
        check_text_language=args.check_text_language,
        number_of_process=args.number_of_process,
    )

    logging.info(
        f"Took %s seconds (processes={args.number_of_process}))", time.time() - ts
    )


if __name__ == "__main__":
    main()
