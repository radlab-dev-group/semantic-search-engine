import os
import queue
import django
import argparse


from radlab_data.utils.threads import WorkerCluster


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from data.controllers.relational_db import PublicRelationDBController
from apps_sse.dataset.generator.question_worker import LLamaHandler, FileWriter


LLAMA_SERVICES_HOSTS = [
    "http://192.168.100.66:8000/api/",
    "http://192.168.100.66:8001/api/",
    "http://192.168.100.69:8000/api/",
    "http://192.168.100.70:8000/api/",
    "http://192.168.100.70:8001/api/",
    "http://192.168.100.71:8000/api/",
    "http://192.168.100.71:8001/api/",
]


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument(
        "-c",
        "--collection-name",
        dest="collection_name",
        type=str,
        required=True,
    )

    p.add_argument(
        "--min-text-len",
        dest="min_text_len",
        type=int,
        required=True,
    )

    p.add_argument(
        "--output-file-path",
        dest="output_file_path",
        type=str,
        required=True,
    )

    return p


# def load_prev_datasets(out_dir):
#     pass


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    print(f"Getting collection {args.collection_name}...")
    collection = PublicRelationDBController.get_collection(
        collection_name=args.collection_name,
    )

    print("Received collection:", collection)
    print("Fetching all chunks to prepare questions...")
    collection_chunks = PublicRelationDBController.get_collection_chunks(
        collection=collection, min_chunk_char_len=300
    )
    print("Number of received chunks:", len(collection_chunks))

    # prev_dataset_s2s = load_prev_datasets(args.output_file_path)

    data_to_process_queue = queue.Queue()
    print("Putting chunks to queque")
    for chunk in collection_chunks:
        data_to_process_queue.put(chunk)

    results_queue = queue.Queue()
    llama_handler_cluster = WorkerCluster(
        "LLamaHandlerCluster",
        [
            LLamaHandler(
                base_api_url=base_api_url,
                dataset_read_queue=data_to_process_queue,
                results_write_queue=results_queue,
            )
            for base_api_url in LLAMA_SERVICES_HOSTS
        ],
    )

    file_writer = FileWriter(args.output_file_path, results_queue)

    llama_handler_cluster.start()
    file_writer.start()

    data_to_process_queue.join()
    llama_handler_cluster.disable()
    llama_handler_cluster.join()

    results_queue.join()
    file_writer.disable()
    file_writer.join()


if __name__ == "__main__":
    main()
