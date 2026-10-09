import zipfile
import zlib
import datetime
import shutil
import stat
import tempfile
import math

from pathlib import Path
from typing import List

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from django.utils import timezone
from django.core.files.uploadedfile import TemporaryUploadedFile

from sse_api.system.models import OrganisationUser

from sse_api.data.models import (
    UploadedDocuments,
    DocumentPageText,
    CollectionOfDocuments,
)

from sse_api.engine.controllers.search.semantic import DBSemanticSearchController
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.core.utils import compute_text_hash


class UploadRejected(ValueError):
    pass


class UploadBudget:
    chunk_size = 64 * 1024

    def __init__(self):
        self.max_files = getattr(settings, "UPLOAD_MAX_FILES", 1000)
        self.max_bytes = getattr(settings, "UPLOAD_MAX_UNCOMPRESSED_BYTES", 100 * 1024 * 1024)
        self.max_ratio = getattr(settings, "UPLOAD_MAX_COMPRESSION_RATIO", 100)
        self.max_upload_bytes = getattr(settings, "UPLOAD_MAX_INPUT_BYTES", 100 * 1024 * 1024)
        limits = (self.max_files, self.max_bytes, self.max_upload_bytes)
        if any(type(value) is not int or value <= 0 for value in limits):
            raise ImproperlyConfigured("Upload count and byte limits must be positive integers")
        if not isinstance(self.max_ratio, (int, float)) or not math.isfinite(self.max_ratio) or self.max_ratio <= 0:
            raise ImproperlyConfigured("Upload compression ratio must be positive and finite")
        self.files = 0
        self.bytes = 0
        self.input_bytes = 0

    def check(self, files=0, size=0):
        if self.files + files > self.max_files or self.bytes + size > self.max_bytes:
            raise UploadRejected("Upload limit exceeded")

    def copy(self, source, target, compressed_size=None):
        self.check(files=1)
        self.files += 1
        size = 0
        while True:
            chunk = source.read(self.chunk_size)
            if not chunk:
                break
            size += len(chunk)
            self.check(size=len(chunk))
            if compressed_size is None:
                self.input_bytes += len(chunk)
                if self.input_bytes > self.max_upload_bytes:
                    raise UploadRejected("Upload input limit exceeded")
            if compressed_size is not None and size > compressed_size * self.max_ratio:
                raise UploadRejected("Compression ratio exceeded")
            self.bytes += len(chunk)
            target.write(chunk)


def contained_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise UploadRejected("Unsafe upload path")
    parts = name.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in parts) or name.startswith("/"):
        raise UploadRejected("Unsafe upload path")
    root = Path(root).resolve()
    target = root.joinpath(*parts)
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise UploadRejected("Symlink upload path")
        if parent.parent.is_dir() and any(
            sibling.name.casefold() == parent.name.casefold() and sibling.name != parent.name
            for sibling in parent.parent.iterdir()
        ):
            raise UploadRejected("Upload path collision")
    if not target.resolve().is_relative_to(root):
        raise UploadRejected("Unsafe upload path")
    return target


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

        # Store files to destination upload directory
        uploaded_files_paths = []
        try:
            budget = UploadBudget()
            for file_to_save in files:
                uploaded_files_paths.extend(
                    self._store_single_file_to_upload_dir(
                        upload_dir=upload_dest_dir, file_to_save=file_to_save, budget=budget
                    )
                )
        except (UploadRejected, ImproperlyConfigured):
            shutil.rmtree(upload_dest_dir)
            raise
        upl_doc = self.get_add_uploaded_documents(
            dir_path=upload_dest_dir, organisation_user=organisation_user
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
        dir_hash = compute_text_hash(dir_path)
        upl_doc, _ = UploadedDocuments.objects.get_or_create(
            dir_path=dir_path, dir_hash=dir_hash, uploaded_by=organisation_user
        )
        return upl_doc

    @staticmethod
    def unzip_uploaded_zip_file(full_upload_path, zip_file_obj, budget=None) -> list:
        budget = budget or UploadBudget()
        contained_path(full_upload_path, zip_file_obj.name)
        created = []
        paths = []
        try:
            with tempfile.TemporaryFile() as archive:
                while True:
                    chunk = zip_file_obj.read(budget.chunk_size)
                    if not chunk:
                        break
                    budget.input_bytes += len(chunk)
                    if budget.input_bytes > budget.max_upload_bytes:
                        raise UploadRejected("Upload input limit exceeded")
                    archive.write(chunk)
                archive.seek(0)
                with zipfile.ZipFile(archive) as zipped:
                    entries = zipped.infolist()
                    if len(entries) > budget.max_files * 2:
                        raise UploadRejected("Too many ZIP entries")
                    seen = set()
                    components = {}
                    declared_bytes = 0
                    declared_files = 0
                    for entry in entries:
                        target = contained_path(full_upload_path, entry.orig_filename)
                        key = str(target).casefold()
                        mode = entry.external_attr >> 16
                        kind = stat.S_IFMT(mode)
                        if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or (kind == stat.S_IFDIR and not entry.is_dir()):
                            raise UploadRejected("Non-regular ZIP entry")
                        if key in seen or target.exists():
                            raise UploadRejected("Upload path collision")
                        seen.add(key)
                        for path in (target, *target.parents):
                            if path == Path(full_upload_path).resolve():
                                break
                            folded = str(path).casefold()
                            if folded in components and components[folded] != str(path):
                                raise UploadRejected("Upload path collision")
                            components[folded] = str(path)
                        if not entry.is_dir():
                            declared_files += 1
                            declared_bytes += entry.file_size
                            if entry.file_size > entry.compress_size * budget.max_ratio:
                                raise UploadRejected("Compression ratio exceeded")
                        budget.check(files=declared_files, size=declared_bytes)
                    for entry in entries:
                        target = contained_path(full_upload_path, entry.orig_filename)
                        missing = []
                        parent = target if entry.is_dir() else target.parent
                        while not parent.exists():
                            missing.append(parent)
                            parent = parent.parent
                        for directory in reversed(missing):
                            directory.mkdir()
                            created.append(directory)
                        if entry.is_dir():
                            continue
                        with target.open("xb") as output:
                            created.append(target)
                            with zipped.open(entry) as source:
                                budget.copy(source, output, entry.compress_size)
                        paths.append(str(target))
            return paths
        except (UploadRejected, zipfile.BadZipFile, zipfile.LargeZipFile, OSError, RuntimeError, EOFError, NotImplementedError, zlib.error) as exc:
            for target in reversed(created):
                if target.is_dir():
                    target.rmdir()
                else:
                    target.unlink()
            raise UploadRejected("ZIP upload rejected") from exc

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
        root = Path(self.upload_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for component in (organisation_name, organisation_user.auth_user.username, collection_name):
            if "/" in component:
                raise UploadRejected("Unsafe upload directory")
            root = contained_path(root, component)
            root.mkdir(exist_ok=True)
        return tempfile.mkdtemp(prefix=date_str + "_", dir=root)

    def _store_single_file_to_upload_dir(
        self, upload_dir: str, file_to_save: TemporaryUploadedFile, budget=None
    ) -> list:
        """
        Store temporary file to destination dir
        :param upload_dir:
        :param file_to_save:
        :return: Path to stored file
        """
        budget = budget or UploadBudget()
        if file_to_save.name.lower().endswith(".zip"):
            return self.unzip_uploaded_zip_file(upload_dir, file_to_save, budget)

        target = contained_path(upload_dir, file_to_save.name)
        if target.exists() or not target.parent.is_dir():
            raise UploadRejected("Upload path collision")
        created = False
        try:
            with target.open("xb") as output:
                created = True
                budget.copy(file_to_save, output)
            return [str(target)]
        except (UploadRejected, OSError) as exc:
            if created:
                target.unlink()
            raise UploadRejected("File upload rejected") from exc
