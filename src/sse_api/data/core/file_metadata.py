"""Per-file upload metadata, independent of chunking and of the task transport."""

import json
from pathlib import Path


class FileMetadataError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate file_metadata key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("file_metadata must contain finite JSON values")


def _basename(name):
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def resolve_file_metadata(value, filenames):
    """Validate a complete mapping of ordinary uploads and normalize its keys.

    None means the field was omitted. A supplied mapping must cover every
    non-ZIP file exactly once; archives and their members cannot be targeted.
    Exact names take precedence over an unambiguous basename match.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(
            value, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    if not isinstance(value, dict):
        raise ValueError("file_metadata must be an object")
    json.dumps(value, allow_nan=False)
    filenames = list(filenames)
    if len(set(filenames)) != len(filenames):
        raise ValueError("Ambiguous upload names")
    resolved = {}
    for name, metadata in value.items():
        if not isinstance(name, str) or not name or not _basename(name):
            raise ValueError("Invalid file_metadata name")
        matches = [filename for filename in filenames if filename == name]
        if not matches:
            matches = [
                filename
                for filename in filenames
                if _basename(filename) == _basename(name)
            ]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous file_metadata target")
        target = matches[0]
        if target.lower().endswith(".zip") or target in resolved:
            raise ValueError("Duplicate or ZIP file_metadata target")
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"category", "options"}
            or not isinstance(metadata["category"], str)
            or not isinstance(metadata["options"], dict)
        ):
            raise ValueError("file_metadata entries require category and options")
        resolved[target] = metadata
    ordinary_files = {name for name in filenames if not name.lower().endswith(".zip")}
    if set(resolved) != ordinary_files:
        raise ValueError("Missing file_metadata for an uploaded file")
    return resolved


def apply_file_metadata(documents, file_metadata, upload_dir):
    """Merge metadata before hashing/saving, without touching ZIP members.

    Returns (document dictionary, explicit category) pairs. All matches are
    checked before any document is saved, including files ignored by a reader.
    """
    if file_metadata is None:
        return [(document, False) for document in documents]
    root = Path(upload_dir).resolve()
    remaining = set(file_metadata)
    prepared = []
    for document in documents:
        try:
            name = Path(document["filepath"]).resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise FileMetadataError("Reader result outside upload directory") from exc
        metadata = file_metadata.get(name)
        if metadata is not None:
            if name not in remaining:
                raise FileMetadataError("Ambiguous file_metadata reader result")
            remaining.remove(name)
            document = dict(document)
            document["category"] = metadata["category"]
            document["options"] = {
                **(document.get("options") or {}),
                **metadata["options"],
            }
        prepared.append((document, metadata is not None))
    if remaining:
        raise FileMetadataError("Uploaded files with metadata were not read")
    return prepared