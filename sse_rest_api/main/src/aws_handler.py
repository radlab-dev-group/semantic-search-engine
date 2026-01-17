import os
import json
import boto3

import logging
from main.src.constants import CONFIG_DIR, AWS_CONFIG_FILENAME, get_logger


class AwsHandler:
    AWS_JSON_SECTION = "aws"

    REGION_NAME = "REGION_NAME"
    ENDPOINT_URL = "ENDPOINT_URL"
    ACCESS_KEY = "ACCESS_KEY_ID"
    BUCKET_NAME = "STORAGE_BUCKET_NAME"
    SECRET_ACCESS_KEY = "SECRET_ACCESS_KEY"

    ACCEPT_FILE_TYPES = ["json", "jsonl"]

    def __init__(self, prepare: bool = True):
        self._bucket = None
        self._client = None
        self._json_config = None
        self._last_error = None

        try:
            self.logger = get_logger()
        except Exception:
            self.__set_default_logger()

        self._is_loaded = False
        if prepare:
            self.prepare_resource()

    @property
    def last_error(self):
        return str(self._last_error) if self._last_error is not None else None

    def reset_errors(self):
        self._last_error = None

    def prepare_resource(self):
        self.__load_aws_config()
        conn_opts = {
            "endpoint_url": self._json_config[self.ENDPOINT_URL],
            "aws_access_key_id": self._json_config[self.ACCESS_KEY],
            "aws_secret_access_key": self._json_config[self.SECRET_ACCESS_KEY],
            "region_name": self._json_config[self.REGION_NAME],
        }
        for c_o, c_v in conn_opts.items():
            assert c_v is not None, f"{c_o} value must be defined!"
            assert len(c_v), f"{c_o} value must be defined!"

        self._bucket = self._json_config[self.BUCKET_NAME]
        assert len(self._bucket), f"Bucket name must be defined!"

        self._client = boto3.client("s3", **conn_opts)

        assert self._client is not None, f"Problem while connecting to AWS!"

        self._is_loaded = True

    def mkdir(self, path: str):
        self.__assert_params()
        return self.__handle_response_from_aws(
            response=self._client.put_object(
                Bucket=self._bucket, Body="", Key=self.__proper_path(path=path)
            )
        )

    def rm(self, file_or_dirpath: str):
        self.__assert_params()
        return self.__handle_response_from_aws(
            response=self._client.put_object(
                Bucket=self._bucket, Key=file_or_dirpath
            )
        )

    def add_file_from_buffer(self, buffer, destination_file_path: str):
        self.__assert_params()
        return self.__handle_response_from_aws(
            response=self._client.put_object(
                Body=buffer, Bucket=self._bucket, Key=destination_file_path
            )
        )

    def add_file_from_path(self, file_path: str, destination_dir: str):
        self.__assert_params()
        with open(file_path) as f:
            object_data = f.read()

        return self.add_file_from_buffer(
            buffer=object_data,
            destination_file_path=os.path.join(
                destination_dir, os.path.basename(file_path)
            ),
        )

    def ls(self, dir_to_list: str = None, extensions: list = None):
        self.reset_errors()

        list_opts = {"Bucket": self._bucket}
        if dir_to_list is not None:
            list_opts["Prefix"] = dir_to_list
        out_files_paths = []

        ls_out = self._client.list_objects(**list_opts)
        if not self.__handle_response_from_aws(response=ls_out):
            self._last_error = "Problem while listing objects from AWS (ls method)"
            self.logger.error(self._last_error)
            return []

        for f_out in ls_out.get("Contents", []):
            f_out_path = f_out.get("Key", None)
            if f_out_path is None:
                continue
            out_files_paths.append(f_out_path)

        out_files_paths = self.__filter_files_with_extensions(
            files_paths=out_files_paths, extensions=extensions
        )

        return out_files_paths

    def load_file(self, file_path: str, file_type: str = None):
        self.reset_errors()

        if file_type is None:
            file_type = self.__resolve_file_type(file_path=file_path)
            if file_type is None:
                if self.logger:
                    error_str = f"Invalid type of file: {file_path}. "
                    error_str += f"Choose from {self.ACCEPT_FILE_TYPES}"
                    self._last_error = error_str
                    self.logger.error(error_str)
                    return None
                else:
                    raise Exception(f"Invalid file type file: {file_path}")

        return self.__load_file_from_aws_path(
            file_path=file_path, file_type=file_type
        )

    @staticmethod
    def __filter_files_with_extensions(files_paths: list, extensions: list):
        if extensions is None or not len(extensions):
            return files_paths
        filter_files = []
        for f in files_paths:
            for e in extensions:
                if not f.endswith(e):
                    continue
                filter_files.append(f)
        return filter_files

    def __resolve_file_type(self, file_path: str) -> str or None:
        f_ext = os.path.basename(file_path).strip().lower().split(".")[-1]
        if f_ext not in self.ACCEPT_FILE_TYPES:
            return None
        return f_ext

    def __handle_response_from_aws(self, response: dict):

        r_md = response.get("ResponseMetadata", {})
        if not self.__is__aws__response__metadata__ok(response):
            self._last_error = (
                "Cannot parse response from AWS (handle__response__from__aws)"
            )
            self.logger.error(self._last_error)
            return None

        if int(r_md["HTTPStatusCode"]) != 200:
            return None
        return True

    def __load_aws_config(self):
        with open(os.path.join(CONFIG_DIR, AWS_CONFIG_FILENAME), "r") as f:
            self._json_config = json.load(f)[self.AWS_JSON_SECTION]

    def __assert_params(self):
        assert self._is_loaded, f"AWS Handler config is not loaded!"

    def __load_file_from_aws_path(self, file_path: str, file_type: str):
        try:
            file_bytes = self.__load_file_bytes(file_path=file_path)
        except Exception as e:
            self._last_error = e
            self.logger.error(e)
            return None

        if file_bytes is None:
            return None
        file_bytes = file_bytes.decode()

        r_file = None
        if type(file_bytes) in [dict]:
            r_file = file_bytes
        elif type(file_bytes) in [str, bytes]:
            try:
                if file_type == "json":
                    r_file = json.loads(file_bytes)
                elif file_type == "jsonl":
                    r_file = []
                    for line in file_bytes.splitlines():
                        try:
                            r_file.append(json.loads(line))
                        except Exception as e:
                            self.logger.error(f"Error while parsing jsonl file: {e}")
                            continue
            except Exception:
                r_file = file_bytes
        else:
            raise Exception(f"Invalid file_bytes type: {type(file_bytes)}")
        return r_file

    def __load_file_bytes(self, file_path: str):
        obj = self._client.get_object(Bucket=self._bucket, Key=file_path)
        obj_body = obj.get("Body", None).read()
        return obj_body

    @staticmethod
    def __proper_path(path: str):
        return path.rstrip("/") + "/"

    @staticmethod
    def __is__aws__response__metadata__ok(response: dict):
        r_md = response.get("ResponseMetadata", {})
        return int(r_md.get("HTTPStatusCode", -1)) in [200]

    def __set_default_logger(self):
        logging.basicConfig(
            level=logging.WARNING,
            format="{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )
        self.logger = logging.getLogger(AWS_CONFIG_FILENAME)
