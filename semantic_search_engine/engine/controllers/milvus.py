import os
import json
import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


from pymilvus import (
    db,
    MilvusClient,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

INDEX_QUERY_PARAMS = {
    "HNSW": {
        "INDEX_PARAMS": {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
            "index_name": "emb_idx",
        },
        "QUERY_PARAMS": {
            "metric_type": "L2",
            "params": {"ef": 100},
        },
    },
    "IVF_FLAT": {
        "INDEX_PARAMS": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1536},
            "index_name": "emb_idx",
        },
        "QUERY_PARAMS": {
            "metric_type": "L2",
            "params": {"nprobe": 128},
            "offset": 0,
        },
    },
}

CACHED_MODELS = {}


class MilvusHandler:
    DEFAULT_INDEX_TYPE = "IVF_FLAT"
    DB_CONNECTION_JSON_FIELD = "milvus_db_connection"
    DEFAULT_COLLECTION_METRIC_TYPE = "COSINE"

    DB_FIELD_TEXT = "text"
    DB_FIELD_METADATA = "metadata"
    DB_FIELD_IS_ACTIVE = "is_active"
    DB_FIELD_EMBEDDING = "embedding"

    DEFAULT_INDEX_PARAMS = INDEX_QUERY_PARAMS[DEFAULT_INDEX_TYPE]["INDEX_PARAMS"]
    DEFAULT_INDEX_PARAMS["field_name"] = DB_FIELD_EMBEDDING

    BASE_FILTER_EXPR = f"{DB_FIELD_IS_ACTIVE} == true"
    SEARCH_FIELDS = [DB_FIELD_TEXT, DB_FIELD_METADATA]

    SEMANTIC_SEARCH_FACTOR_MAX_RESULTS = 10

    def __init__(
        self,
        jsonl_config_path: str,
        collection_name: str,
        collection_description: str | None = None,
        embedder_model_path: str | None = None,
        embedding_size: int | None = None,
        create_db_if_not_exists: bool = True,
        load_embedder: bool = True,
        load_collection_and_schema: bool = True,
        extended_schema: bool = False,
        vec_1_size: int = None,
        vec_2_size: int = None,
        collection_index_params: dict = None,
        index_name: str = DEFAULT_INDEX_TYPE,
        reranker_model_path: str | None = None,
        load_reranker: bool = False,
        use_cached_models: bool = True,
        embedder_device: str = "cpu",
        reranker_device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        """
        Collection constructor
        :param jsonl_config_path:
        :param collection_name:
        :param embedder_model_path:
        :param create_db_if_not_exists:
        :param load_embedder:
        :param load_collection_and_schema:
        :return:
        """
        if index_name is not None and index_name not in INDEX_QUERY_PARAMS:
            raise Exception(f"{index_name} is not a valid index name")

        self.index_name = index_name
        self.vec_1_size = vec_1_size
        self.vec_2_size = vec_2_size
        self.embedder_model_path = embedder_model_path
        self.embedding_size = embedding_size
        self.extended_schema = extended_schema
        self.jsonl_config_path = jsonl_config_path
        self.collection_description = collection_description

        self.use_cached_models = use_cached_models
        self.reranker_model_path = reranker_model_path

        self._emb_device = embedder_device
        self._rer_device = reranker_device
        self._normalize_embeddings = normalize_embeddings

        self._database = None
        self._is_connected = False
        self._reranker_model = None
        self._embedder_model = None
        self._schema_fields = None
        self._milvus_client = None
        self._collection = None
        self._collection_schema = None
        self._collection_index_params = None

        self._connection_config = self.__load_json_config(jsonl_config_path)
        self._collection_name = self.__prepare_collection_name(
            collection_name=collection_name
        )

        if create_db_if_not_exists:
            self.__prepare_collection_index_params(params=collection_index_params)
            self._connect_to_milvus_db(check_db=True)

        if load_collection_and_schema:
            self.__get_add_collection_and_schema()

        if load_embedder:
            self.__load_embedder_model_from_path()

        if load_reranker:
            self.__load_reranker_model_from_path()

    @property
    def collection_name(self):
        return self._collection_name

    def close_connection(self):
        self._disconnect_from_milvus_db()

    def db_collections(self) -> List[str]:
        not_connected = False
        if self._milvus_client is None:
            not_connected = True
            self.__prepare_milvus_client()
        collections = self._milvus_client.list_collections()
        if not_connected:
            self.close_connection()
        return collections

    def add_texts(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        max_text_len: int | None = None,
    ) -> None:
        """
        Add text to milvus collection
        :param texts: List of text string to prepare embeddings
               and add to milvus Collection
        :param metadata: List of metadata
        :param max_text_len: If text is longer will be
              truncated to this length (number of characters)
        :return: None
        """
        if self._embedder_model is None:
            raise Exception("Embedder model is not initialized!")

        if max_text_len is not None and max_text_len:
            texts = [t[:max_text_len] for t in texts]

        texts_embeddings = self.prepare_embeddings(texts=texts)

        data_to_insert = []
        for t, e, m in zip(texts, texts_embeddings, metadata):
            data_to_insert.append(
                {
                    self.DB_FIELD_TEXT: t,
                    self.DB_FIELD_EMBEDDING: e,
                    self.DB_FIELD_METADATA: m,
                    self.DB_FIELD_IS_ACTIVE: True,
                }
            )

        self._milvus_client.insert(
            collection_name=self._collection_name, data=data_to_insert
        )

    def add_single_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        max_text_len: int | None = None,
    ) -> None:
        """
        Add text to milvus collection
        :param text: Text to be added
        :param metadata: Text metadata
        :param max_text_len: If text is longer will be
              truncated to this length (number of characters)
        :return: None
        """
        return self.add_texts(
            texts=[text], metadata=[metadata], max_text_len=max_text_len
        )

    def prepare_embeddings(self, texts: list) -> list:
        """
        For given list of string prepare and returns embeddings
        :param texts: List of texts to prepare embeddings
        :return: List of prepared embeddings
        """
        with torch.no_grad():
            texts_embeddings = self._embedder_model.encode(
                sentences=texts,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
            )
        return texts_embeddings

    def search(
        self,
        search_text: str,
        max_results: int = 100,
        additional_output_fields: list | None = None,
        post_search_options: dict | None = None,
        metadata_filter: dict | None = None,
    ) -> list:
        """
        Main search function for milvus collection.
        :param search_text: Text to search into database
        :param max_results: Number of results to return
        :param additional_output_fields: List of additional fields to return
        from milvus db. These fields will be returned with defined self.SEARCH_FIELDS
        :param post_search_options: Additional options to pass after db
        search is finished, like reranking
        :param metadata_filter: Filtering options
        :return: List of results
        """
        self.__prepare_milvus_client()

        search_text_emb = self.prepare_embeddings([search_text])
        filter_expr = self.BASE_FILTER_EXPR

        filter_expr += self.__prepare_filtering_options(
            metadata_filter=metadata_filter
        )

        return_fields = self.SEARCH_FIELDS
        if additional_output_fields is not None and len(additional_output_fields):
            return_fields += additional_output_fields

        whole_res = self._milvus_client.search(
            collection_name=self.collection_name,
            data=search_text_emb,
            filter=filter_expr,
            search_params=INDEX_QUERY_PARAMS[self.index_name]["QUERY_PARAMS"],
            limit=max_results * self.SEMANTIC_SEARCH_FACTOR_MAX_RESULTS,
            output_fields=return_fields,
        )

        all_queries_results = []
        for results in whole_res:
            query_results = []
            for hit in results:
                res_dict = {"score": hit["distance"]}
                for q_param in return_fields:
                    res_dict[q_param] = hit["entity"].get(q_param, "")
                query_results.append(res_dict)
            all_queries_results.append(query_results)

        if (
            post_search_options is not None
            and post_search_options["rerank_results"]
            and len(all_queries_results)
        ):
            self.__load_reranker_model_from_path()
            all_queries_results = self._rerank_search_results(
                search_text, all_queries_results
            )

        if (
            post_search_options is not None
            and post_search_options["return_with_factored_fields"]
        ):
            return all_queries_results

        all_queries_results_head = []
        for r in all_queries_results:
            all_queries_results_head.append(r[:max_results])
        return all_queries_results_head

    # ----------------------------------------------------------------------------
    def __load_json_config(self, json_config_path: str):
        """
        Loads connection configuration from json file
        :param json_config_path: Path to json file
        :return:
        """
        self._connection_config = json.load(open(json_config_path, "r"))[
            self.DB_CONNECTION_JSON_FIELD
        ]
        self.__get_connection_from_env()
        self.__check_connection_configuration()
        return self._connection_config

    def __check_connection_configuration(self):
        if self._connection_config["host"] is None or not len(
            self._connection_config["host"]
        ):
            raise Exception(
                "Milvus host is not defined, please set "
                "ENV_MILVUS_HOST environment variable or use config file"
            )

        if self._connection_config["db_name"] is None or not len(
            self._connection_config["db_name"]
        ):
            raise Exception(
                "Milvus database name is not defined, please set "
                "ENV_MILVUS_DBNAME environment variable or use config file"
            )

        if self._connection_config["user"] is None or not len(
            self._connection_config["user"]
        ):
            raise Exception(
                "Milvus user name is not defined, please set ENV_MILVUS_USER "
                "environment variable or use config file"
            )

        if self._connection_config["password"] is None or not len(
            self._connection_config["password"]
        ):
            raise Exception(
                "Password for Milvus user name is not defined, please set "
                "ENV_MILVUS_USER environment variable or use config file"
            )

        if self._connection_config["port"] is None:
            raise Exception(
                "Milvus port is not defined, please set ENV_MILVUS_PORT "
                "environment variable or use config file"
            )

    def __get_connection_from_env(self):
        host = os.environ.get("ENV_MILVUS_HOST", None)
        if host is not None:
            self._connection_config["host"] = host
        db_name = os.environ.get("ENV_MILVUS_DBNAME", None)
        if db_name is not None:
            self._connection_config["db_name"] = db_name
        port = os.environ.get("ENV_MILVUS_PORT", None)
        if port is not None:
            self._connection_config["port"] = int(port)
        user = os.environ.get("ENV_MILVUS_USER", None)
        if user is not None:
            self._connection_config["user"] = user
        password = os.environ.get("ENV_MILVUS_PASSWORD", None)
        if password is not None:
            self._connection_config["password"] = password

    # ----------------------------------------------------------------------------

    def _rerank_search_results(
        self, search_text: str, all_queries_results: list
    ) -> list:
        if self._reranker_model is None:
            raise Exception("Reranker model is not loaded")

        re_results = []
        for query_results in all_queries_results:
            ce_texts_pairs = []
            for result in query_results:
                ce_texts_pairs.append([search_text, result[self.DB_FIELD_TEXT]])
            ce_query_result = self._reranker_model.predict(ce_texts_pairs)
            sorted_ce_query_result = sorted(
                {idx: r for idx, r in enumerate(ce_query_result)}.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            q_res = []
            for text_idx, score in sorted_ce_query_result:
                text_res = query_results[text_idx]
                text_res["score"] = score
                q_res.append(text_res)
            re_results.append(q_res)
        return re_results

    # ----------------------------------------------------------------------------
    def _connect_to_milvus_db(self, check_db: bool = True):
        """
        Establish connection to milvus database
        :param check_db: Check if database exists, if not then create
        :return: connection
        """
        if check_db:
            db_name = self._connection_config["db_name"]
            connections.connect(
                host=self._connection_config["host"],
                port=self._connection_config["port"],
            )

            all_databases = db.list_database()
            if check_db and db_name not in all_databases:
                db.create_database(db_name)
        self._is_connected = self._database is not None

    def _disconnect_from_milvus_db(self):
        if self._milvus_client is not None:
            self._milvus_client.close()
        self._is_connected = False

    # ----------------------------------------------------------------------------
    def __prepare_collection_index_params(self, params: dict | None):
        self._collection_index_params = (
            params if params is not None else self.DEFAULT_INDEX_PARAMS
        )

    @staticmethod
    def __prepare_collection_name(collection_name: str | None) -> str | None:
        if collection_name is None:
            return None
        return collection_name.replace(" ", "_")

    def __prepare_schema_fields(self, extended_schema: bool) -> list:
        self._schema_fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name=self.DB_FIELD_TEXT,
                dtype=DataType.VARCHAR,
                max_length=10240,
            ),
            FieldSchema(
                name=self.DB_FIELD_EMBEDDING,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_size,
            ),
            FieldSchema(
                name=self.DB_FIELD_METADATA,
                dtype=DataType.JSON,
            ),
            FieldSchema(name=self.DB_FIELD_IS_ACTIVE, dtype=DataType.BOOL),
        ]

        if extended_schema:
            if self.vec_1_size is not None:
                self._schema_fields.append(
                    FieldSchema(
                        name="vec_1",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.vec_1_size,
                    ),
                )
            if self.vec_2_size is not None:
                self._schema_fields.append(
                    FieldSchema(
                        name="vec_2",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.vec_2_size,
                    ),
                )

        return self._schema_fields

    # ----------------------------------------------------------------------------
    def __prepare_filtering_options(self, metadata_filter: dict | None) -> str:
        out_filter_opts = ""
        if metadata_filter is None:
            return out_filter_opts

        md_field = self.DB_FIELD_METADATA
        if "text_language" in metadata_filter and len(
            metadata_filter["text_language"].strip()
        ):
            text_lang = metadata_filter["text_language"].strip()
            out_filter_opts += f' and {md_field}["text_language"] == "{text_lang}"'

        if "filenames" in metadata_filter and len(metadata_filter["filenames"]):
            in_docs_str = ", ".join([f'"{d}"' for d in metadata_filter["filenames"]])
            out_filter_opts += f' and {md_field}["filename"] in [{in_docs_str}]'

        if "relative_paths" in metadata_filter and len(
            metadata_filter["relative_paths"]
        ):
            rel_paths_str = ", ".join(
                [f'"{d}"' for d in metadata_filter["relative_paths"]]
            )
            out_filter_opts += (
                f' and {md_field}["relative_path"] in [{rel_paths_str}]'
            )

        return out_filter_opts

    # ----------------------------------------------------------------------------
    def __add_milvus_collection(self, load_collection: bool):
        self._milvus_client.create_collection(
            collection_name=self._collection_name,
            collection_description=self.collection_description,
            dimension=self.embedding_size,
            primary_field_name="pk",
            # id_type="int",
            vector_field_name=self.DB_FIELD_EMBEDDING,
            metric_type=self.DEFAULT_COLLECTION_METRIC_TYPE,
            # auto_id=True,
            schema=self._collection_schema,
        )
        self.__add_index_to_collection()

        if load_collection:
            self._milvus_client.load_collection(
                collection_name=self._collection_name, replica_number=1
            )

    def __add_index_to_collection(self):
        index_params = self._milvus_client.prepare_index_params()
        index_params.add_index(
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024},
            field_name=self.DB_FIELD_EMBEDDING,
            index_name="emb_idx",
        )

        self._milvus_client.create_index(
            collection_name=self._collection_name, index_params=index_params
        )

    def __prepare_collection_schema(self):
        self._schema_fields = self.__prepare_schema_fields(
            extended_schema=self.extended_schema
        )
        self._collection_schema = CollectionSchema(
            fields=self._schema_fields,
            description=self.collection_description,
        )

    def __get_add_collection_and_schema(self):
        """
        Creates schema or extended schema
        :return: Milvus Collection object
        """
        self.__prepare_milvus_client()
        self.__prepare_collection_schema()
        if not self._milvus_client.has_collection(
            collection_name=self._collection_name
        ):
            if self.embedding_size is None:
                raise Exception("Collection embedding size must be set")
            self.__add_milvus_collection(load_collection=True)

    def __prepare_milvus_client(self, load_collection: bool = False):
        if self._milvus_client is None:
            port = self._connection_config["port"]
            host = self._connection_config["host"].strip("/")
            uri = self._connection_config["uri"].rstrip("/").rstrip(":")
            uri_connection = f"{uri}://{host}:{port}"
            self._milvus_client = MilvusClient(
                uri=uri_connection,
                user=self._connection_config["user"],
                password=self._connection_config["password"],
                db_name=self._connection_config["db_name"],
            )

        if load_collection:
            self._milvus_client.load_collection(
                collection_name=self._collection_name
            )

    # ----------------------------------------------------------------------------
    def __load_embedder_model_from_path(self):
        if self.embedder_model_path is None:
            raise Exception("Embedder model path must be set!")

        if self._embedder_model is None:
            if self.use_cached_models:
                if self.embedder_model_path in CACHED_MODELS:
                    self._embedder_model = CACHED_MODELS[self.embedder_model_path]
                    return self._embedder_model

            load_opts = {}
            if self._emb_device is not None and len(self._emb_device):
                load_opts["device"] = self._emb_device

            if "trust_remote_code" not in load_opts:
                load_opts["trust_remote_code"] = True

            self._embedder_model = SentenceTransformer(
                self.embedder_model_path, **load_opts
            )

            if self.use_cached_models:
                CACHED_MODELS[self.embedder_model_path] = self._embedder_model

        return self._embedder_model

    def __load_reranker_model_from_path(self):
        if self.reranker_model_path is None:
            raise Exception("Reranker model path must be set!")

        if self._reranker_model is None:
            if self.use_cached_models:
                if self.reranker_model_path in CACHED_MODELS:
                    self._reranker_model = CACHED_MODELS[self.reranker_model_path]
                    return self._reranker_model

            load_opts = {}
            if self._rer_device is not None and len(self._rer_device):
                load_opts["device"] = self._rer_device

            if "trust_remote_code" not in load_opts:
                load_opts["trust_remote_code"] = True

            self._reranker_model = CrossEncoder(
                self.reranker_model_path, **load_opts
            )

            if self.use_cached_models:
                CACHED_MODELS[self.reranker_model_path] = self._reranker_model
        return self._reranker_model
