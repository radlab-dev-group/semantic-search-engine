"""
file_cleanup.py
---------------

The files of the uploads on the disk.

The documents are indexed from the directory in which the API stored them
(``<upload root>/<organisation>/<user>/<collection>/<timestamp>_...``), so a
deletion which stops at the databases leaves the original files behind.  The
removal happens in two steps, exactly like the deletion of a collection:
the paths are read from the databases while their rows are still there, and
the files are removed once nothing points at them anymore::

    paths = document_paths(documents)
    ...                       # rows and vectors first
    remove_paths(paths)       # the disk last

Two rules make the removal safe:

* only the files inside the root of the uploads may be removed, so a path
  written into the database can never touch anything else;
* a directory disappears only when its last file is gone, so a directory of
  another collection, which still holds files, stays as it is.

Nothing here raises for a file which cannot be removed (a read-only mount, a
file deleted by hand): a deletion must not fail because of the disk, so the
problems are counted and reported instead.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sse_api.data.models import Document, UploadedDocuments

MAIN_LOGGER = logging.getLogger("sse_api")

#: Where the engine stores the files of the uploads; the same directory which
#: the upload controller is given in ``sse_api.data.api``.
DEFAULT_UPLOAD_ROOT = "./upload_sse"


@dataclass
class FileCleanupResult:
    """
    What the removal of the files did with them: how many disappeared and
    which ones could not be removed.
    """

    deleted: int = 0
    errors: list = field(default_factory=list)

    def as_dict(self) -> dict:
        """
        The outcome as it leaves the API.
        :return: Dictionary with the number of the removed files and the
        messages of the files which could not be removed
        """
        return {"deleted": self.deleted, "errors": self.errors}


def upload_root(root=None) -> Path:
    """
    The root of the uploads, resolved.
    :param root: Alternative root of the uploads (tests, other deployments)
    :return: Resolved path of the root
    """
    return Path(root if root is not None else DEFAULT_UPLOAD_ROOT).resolve()


def inside_uploads(path, root=None) -> bool:
    """
    Whether a path may be touched at all: only the files of the uploads of
    the engine, never anything above that root.
    :param path: Path to look at
    :param root: Alternative root of the uploads
    :return: Whether the path lies inside the root of the uploads
    """
    root_path = upload_root(root)
    try:
        return Path(path).resolve().is_relative_to(root_path)
    except (OSError, ValueError):
        return False


def _unique(paths) -> list:
    """
    Paths without repetitions, in the order in which they were given.
    :param paths: Paths to keep
    :return: List of unique paths
    """
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def document_paths(documents, root=None) -> list:
    """
    The files of the documents which belong to the uploads of the engine.
    :param documents: Documents (or any objects with a ``path``)
    :param root: Alternative root of the uploads
    :return: Paths of the files to remove
    """
    paths = []
    for document in documents:
        path = Path(document.path)
        if not inside_uploads(path, root):
            MAIN_LOGGER.warning("Keeping %s: outside of the uploads", path)
            continue
        paths.append(path)
    return _unique(paths)


def upload_directories_of_collection(collection, root=None) -> list:
    """
    The directories of the uploads of a collection.
    :param collection: Collection whose uploads are looked for
    :param root: Alternative root of the uploads
    :return: Directories of the uploads of the collection
    """
    directories = UploadedDocuments.objects.filter(
        document__collection=collection
    ).values_list("dir_path", flat=True)
    return _unique(
        path for path in map(Path, directories) if inside_uploads(path, root)
    )


def collection_paths(collection, root=None) -> list:
    """
    Everything of a collection which can be removed from the disk: the files
    of its documents and the directories of its uploads.  Has to be called
    while the rows are still in the database, which is where the paths come
    from.
    :param collection: Collection whose files are looked for
    :param root: Alternative root of the uploads
    :return: Files and directories of the collection
    """
    documents = Document.objects.filter(collection=collection)
    return document_paths(documents, root=root) + upload_directories_of_collection(
        collection, root=root
    )


def remove_paths(paths, root=None) -> FileCleanupResult:
    """
    Remove the given files, then the directories which are empty because of
    it.  A directory of the list is removed only when it holds nothing, so
    the directories of an upload may be given without any checking.
    :param paths: Files and directories to remove
    :param root: Alternative root of the uploads
    :return: FileCleanupResult
    """
    result = FileCleanupResult()
    removed = []
    for path in _unique([Path(path) for path in paths]):
        if not inside_uploads(path, root):
            MAIN_LOGGER.warning("Keeping %s: outside of the uploads", path)
            continue
        if path.is_dir():
            removed.append(path)
            continue
        if _remove_file(path, result.errors):
            result.deleted += 1
        removed.append(path)
    prune_empty_directories(removed, root=root)
    return result


def _remove_file(path: Path, errors: list) -> bool:
    """
    Remove one file, reporting a problem instead of raising.
    :param path: File to remove
    :param errors: The problems are appended here
    :return: Whether the file is gone (a file which was gone already counts)
    """
    try:
        if path.exists():
            path.unlink()
        return not path.exists()
    except OSError as error:
        errors.append(f"{path.name}: {error}")
        return False


def prune_empty_directories(paths, root=None) -> None:
    """
    Remove the directories which the removed files left empty, from the
    deepest one upwards, stopping at the root of the uploads.

    A directory is removed when it holds nothing, and the walk goes on to its
    parent, so the whole chain of the directories which the last file of an
    upload left behind disappears.  A directory which still holds something
    stops only its own walk: another file, removed later, may be the one
    which empties it.
    :param paths: Files which were removed, or directories to start from
    :param root: Alternative root of the uploads
    """
    root_path = upload_root(root)
    removed = set()
    for given in paths:
        path = Path(given)
        if not path.is_dir():
            # A removed file is not a directory anymore: its parent is where
            # the pruning starts; a directory of the list is pruned itself.
            path = path.parent
        while path != root_path:
            if path in removed:  # its own walk already took it away
                path = path.parent
                continue
            if not path.is_dir():
                break
            try:
                path.rmdir()  # refuses a directory which still holds something
            except OSError:
                break
            removed.add(path)
            path = path.parent


def delete_document_files(documents, root=None) -> FileCleanupResult:
    """
    Remove the files of the given documents with the directories which become
    empty because of it.
    :param documents: Documents whose files are removed
    :param root: Alternative root of the uploads
    :return: FileCleanupResult
    """
    return remove_paths(document_paths(documents, root=root), root=root)


def delete_document_file(document, root=None) -> FileCleanupResult:
    """
    Remove the file of one document, the way it is done once the indexing of
    a document is over (see ``sse_api.data.indexing_pipeline``).
    :param document: Document whose file is removed
    :param root: Alternative root of the uploads
    :return: FileCleanupResult
    """
    return delete_document_files([document], root=root)


def delete_collection_files(collection, root=None) -> FileCleanupResult:
    """
    Remove everything of a collection from the disk.
    :param collection: Collection whose files are removed
    :param root: Alternative root of the uploads
    :return: FileCleanupResult
    """
    return remove_paths(collection_paths(collection, root=root), root=root)
