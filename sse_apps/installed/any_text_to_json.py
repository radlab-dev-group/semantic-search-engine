"""
Sample usage

python3 apps/any_text_to_json.py \
    -d /mnt/data/radlab-projekty/EGO-GPT4-20230425/data/etap1 \
    -o /mnt/local/models/aiassistance_custom_clm/datasets/ego-converted-full-dataset-etap1.json
"""

import tqdm
import time
import json
import argparse
import logging

from radlab_data.text.reader import DirectoryFileReader


def prepare_parser(desc=""):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "-d",
        "--directory",
        dest="directory",
        help="Path to directory",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        help="Path to output json(l) file",
        required=True,
    )
    parser.add_argument(
        "--proper-pages",
        action="store_true",
        dest="prepare_proper_pages",
        help="If option is given, then all texts from "
        "the same page will be merged to single.",
    )
    parser.add_argument(
        "--merge-document-pages",
        action="store_true",
        dest="merge_document_pages",
        help="If option is given, then all pages from the same "
        "document will be merged to document with single page.",
    )
    parser.add_argument(
        "--clear-texts",
        action="store_true",
        dest="clear_texts",
        help="If option is given, then all texts will be "
        "clear using cleaner.FullChainProcessor.",
    )

    chunks_tokens = parser.add_argument(
        "--split-to-max-tokens-in-chunk",
        dest="split_to_max_tokens_in_chunk",
        type=int,
        required=False,
        help="If option is given then all text will be split "
        "into chunks with given max tokens count in single chunk",
    )

    parser.add_argument(
        "--overlap-tokens-between-chunks",
        dest="overlap_tokens_between_chunks",
        type=int,
        required=False,
        help=f"If {chunks_tokens.dest} is given, then this option will "
        f"produce overlapped chunks with given number of tokens to overlapping",
    )

    parser.add_argument(
        "--check-text-language",
        action="store_true",
        dest="check_text_language",
        help="If option is given, then language of each texts will be "
        "checked and added to text's metadata dictionary",
    )

    parser.add_argument(
        "--processes",
        required=False,
        type=int,
        dest="number_of_process",
        help="Number of processes to use for files processing. "
        "As default no multithreading is used",
    )

    return parser


def check_args(args):
    if args.overlap_tokens_between_chunks and not args.split_to_max_tokens_in_chunk:
        raise Exception(
            "When using --overlap-tokens-between-chunks option "
            "--split-to-max-tokens-in-chunk have to be given!"
        )

    if args.split_to_max_tokens_in_chunk and args.split_to_max_tokens_in_chunk < 2:
        raise Exception(
            "When using --split-to-max-tokens-in-chunk "
            "size have to be greater than 1!"
        )

    if args.overlap_tokens_between_chunks and args.overlap_tokens_between_chunks < 1:
        raise Exception(
            "When using --overlap-tokens-between-chunks "
            "size have to be greater than 0!"
        )

    if args.overlap_tokens_between_chunks and args.split_to_max_tokens_in_chunk:
        if args.overlap_tokens_between_chunks >= args.split_to_max_tokens_in_chunk:
            raise Exception(
                "Number of overlapping tokens cannot be larger than single chunk size!"
            )
    return args


def read_files_from_args_dir(args):
    dir_reader = DirectoryFileReader(
        main_dir_path=args.directory,
        read_sub_dirs=True,
        processes_count=args.number_of_process,
    )

    dir_reader.load(
        prepare_proper_pages=args.prepare_proper_pages,
        merge_document_pages=args.merge_document_pages,
        clear_texts=args.clear_texts,
        max_tokens_in_chunk=args.split_to_max_tokens_in_chunk,
        number_of_overlap_tokens=args.overlap_tokens_between_chunks,
        check_text_language=args.check_text_language,
    )

    return dir_reader


def main(argv=None):
    logging.basicConfig(
        format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
    )

    args = check_args(prepare_parser().parse_args(argv))

    ts = time.time()
    logging.info(f"Loading files from directory {args.directory}")
    dir_reader = read_files_from_args_dir(args)

    logging.info(f"Storing documents to {args.output_file}")
    if args.output_file.lower().endswith(".json"):
        documents_as_dict = {"documents": []}
        for doc in dir_reader.documents:
            documents_as_dict["documents"].append(doc.as_dict)

        json.dump(documents_as_dict, open(args.output_file, "wt"), indent=2)
    else:
        with tqdm.tqdm(
            total=len(dir_reader.documents), desc=f"Writing to {args.output_file}"
        ) as pbar:
            with open(args.output_file, "wt") as jsonl_out:
                for doc in dir_reader.documents:
                    jsonl_out.write(json.dumps(doc.as_dict))
                    jsonl_out.write("\n")
                    pbar.update()

    logging.info(
        f"Took %s seconds (processes={args.number_of_process}))", time.time() - ts
    )


if __name__ == "__main__":
    main()
