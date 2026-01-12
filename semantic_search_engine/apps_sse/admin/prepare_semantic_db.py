import argparse

from engine.controllers.milvus import MilvusHandler


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument(
        "--config",
        default="./configs/milvus_config.json",
        help="Milvus configuration file",
    )

    return p


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    m_handler = MilvusHandler(
        jsonl_config_path=args.config,
        collection_name="",
        collection_description="",
        embedding_size=None,
        embedder_model_path=None,
        create_db_if_not_exists=True,
        load_embedder=False,
        load_collection_and_schema=False,
        extended_schema=False,
        vec_1_size=None,
        vec_2_size=None,
    )

    print("All collection in db:", m_handler.db_collections())


if __name__ == "__main__":
    main()
