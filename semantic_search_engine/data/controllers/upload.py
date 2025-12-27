import json
import os
import hashlib
import zipfile
import datetime

from pathlib import Path
from typing import List

from django.utils import timezone
from django.core.files.uploadedfile import TemporaryUploadedFile

from system.models import OrganisationUser
from data.models import UploadedDocuments, DocumentPageText, CollectionOfDocuments
from data.controllers.relational_db import RelationalDBController
from engine.controllers.search import DBSemanticSearchController


class UploadDocumentsController:
    def __init__(
        self,
        upload_dir: str = "./upload/sse/collections/",
        db_indexing_process_count: int = 5,
        store_to_db: bool = True,
    ):
        self.rel_db_controller = RelationalDBController(store_to_db=store_to_db)

        self.upload_dir = upload_dir
        self.store_to_db = store_to_db
        self.db_indexing_process_count = db_indexing_process_count

    def store_and_index_files_rel_db_post_request(
        self,
        organisation_user: OrganisationUser,
        files: List[TemporaryUploadedFile],
        collection: CollectionOfDocuments,
        indexing_options: dict,
        semantic_config_path: str,
    ) -> UploadedDocuments:
        upload_dest_dir = self._prepare_upload_dir(
            collection.name, organisation_user
        )

        upl_doc = self.get_add_uploaded_documents(
            dir_path=upload_dest_dir, organisation_user=organisation_user
        )

        # Store files to destination upload directory
        uploaded_files_paths = []
        for file_to_save in files:
            uploaded_files_paths.extend(
                self._store_single_file_to_upload_dir(
                    upload_dir=upload_dest_dir, file_to_save=file_to_save
                )
            )
        upl_doc.number_of_uploaded_documents = len(uploaded_files_paths)

        upl_doc.begin_indexing_time = timezone.now()
        if self.store_to_db:
            upl_doc.save()

        (
            upl_doc,
            doc_pages_texts,
        ) = self.rel_db_controller.add_uploaded_documents_to_db(
            collection_name=collection.name,
            organisation_user=organisation_user,
            uploaded_document=upl_doc,
            prepare_proper_pages=indexing_options["prepare_proper_pages"],
            merge_document_pages=indexing_options["merge_document_pages"],
            clear_texts=indexing_options["clear_text"],
            use_text_denoiser=indexing_options["use_text_denoiser"],
            max_tokens_in_chunk=indexing_options["max_tokens_in_chunk"],
            number_of_overlap_tokens=indexing_options["number_of_overlap_tokens"],
            check_text_language=indexing_options["check_text_lang"],
            number_of_process=self.db_indexing_process_count,
        )

        self._index_in_semantic_db(
            upl_doc=upl_doc,
            collection=collection,
            doc_pages_texts=doc_pages_texts,
            semantic_config_path=semantic_config_path,
        )

        upl_doc.end_indexing_time = timezone.now()
        if self.store_to_db:
            upl_doc.save()

        return upl_doc

    def _index_in_semantic_db(
        self,
        upl_doc: UploadedDocuments,
        collection: CollectionOfDocuments,
        doc_pages_texts: List[DocumentPageText],
        semantic_config_path: str,
    ):
        if not len(doc_pages_texts):
            return

        sem_db_controller = DBSemanticSearchController(
            collection_name=collection.name,
            index_name=collection.embedder_index_type,
            batch_size=100,
            embedder_model=collection.model_embedder,
            cross_encoder_model=collection.model_reranker,
            jsonl_config_path=semantic_config_path,
        )

        sem_db_controller.index_texts_from_list(
            all_texts=doc_pages_texts, collection=collection
        )

        upl_doc.number_of_indexed_documents_vec_db = len(doc_pages_texts)
        if self.store_to_db:
            upl_doc.save()

    def get_add_uploaded_documents(
        self, dir_path: str, organisation_user: OrganisationUser
    ) -> UploadedDocuments:
        dir_hash = self.hash_from_text(dir_path)
        upl_doc, _ = UploadedDocuments.objects.get_or_create(
            dir_path=dir_path, dir_hash=dir_hash, uploaded_by=organisation_user
        )
        return upl_doc

    @staticmethod
    def hash_from_text(text_str):
        return hashlib.md5(text_str.encode("utf8")).hexdigest()

    @staticmethod
    def unzip_uploaded_zip_file(full_upload_path, zip_file_obj) -> list:
        out_file_path = os.path.join(full_upload_path, zip_file_obj.name)
        with open(out_file_path, "wb") as fout:
            fout.write(zip_file_obj.read())
        zipfile.ZipFile(str(out_file_path)).extractall(str(full_upload_path))

        root_directory = Path(full_upload_path)
        for f in root_directory.glob("*"):
            print(f)

        return [out_file_path]

    def _prepare_upload_dir(
        self, collection_name: str, organisation_user: OrganisationUser
    ) -> str:
        """
        Prepare destination upload dir. Destination directory path is created as:
        base_upload_dir/organisation_name/username/collection_name/date_str.
        Creates destination dir when not exists.

        :param collection_name:
        :param organisation_user:
        :return:
        """
        organisation_name = organisation_user.organisation.name
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%s")
        full_upload_path = os.path.join(
            self.upload_dir,
            organisation_name,
            organisation_user.auth_user.username,
            collection_name,
            date_str,
        )
        os.makedirs(full_upload_path, exist_ok=True)
        return full_upload_path

    def _store_single_file_to_upload_dir(
        self, upload_dir: str, file_to_save: TemporaryUploadedFile
    ) -> list:
        """
        Store temporary file to destination dir
        :param upload_dir:
        :param file_to_save:
        :return: Path to stored file
        """
        if file_to_save.name.lower().endswith(".zip"):
            return self.unzip_uploaded_zip_file(upload_dir, file_to_save)

        out_file_path = os.path.join(upload_dir, file_to_save.name)
        with open(out_file_path, "wb") as upl_file:
            for up_file_chunk in file_to_save.chunks():
                upl_file.write(up_file_chunk)
        return [out_file_path]
