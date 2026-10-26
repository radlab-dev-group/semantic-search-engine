import json
import math
from collections.abc import Mapping


def request_params(request):
    if hasattr(request, "validated_params"):
        return request.validated_params
    body = request.data
    params = _as_params(body)
    if request.method == "GET":
        params.update(_as_params(request.query_params))
    return params


#: Suffix which names a list parameter of the API (``texts[]``, ``chunk_ids[]``).
LIST_PARAM_SUFFIX = "[]"


def _as_params(body):
    """
    Parameters of a request body as a plain dictionary.

    A form-encoded body arrives as a ``QueryDict`` whose ``items()`` keeps only
    the **last** value of a repeated key, so ``chunk_ids[]=1`` would be a
    string rather than the one-element list the validators of the list
    parameters expect.  Keys named with the ``[]`` suffix are therefore always
    read with ``getlist``, which makes a form-encoded request behave exactly
    like the JSON one.
    """
    if not isinstance(body, Mapping):
        return {}
    params = dict(body.items())
    getlist = getattr(body, "getlist", None)
    if callable(getlist):
        for key in params:
            if isinstance(key, str) and key.endswith(LIST_PARAM_SUFFIX):
                params[key] = getlist(key)
    return params


def required_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty text")
    return value


def optional_text(value):
    if value is not None and not isinstance(value, str):
        raise ValueError("Expected text")
    return value


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Expected positive identifier")
    if isinstance(value, str) and not value.isdecimal():
        raise ValueError("Expected positive identifier")
    value = int(value)
    if value <= 0 or value > 9223372036854775807:
        raise ValueError("Identifier out of range")
    return value


def boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError("Expected boolean")


def options_object(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Expected options object")
    return value.copy()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Expected number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("Number out of range") from exc
    if not math.isfinite(value):
        raise ValueError("Expected finite number")
    return value


def rating_integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Expected integer rating")
    if isinstance(value, str) and not value.isdecimal():
        raise ValueError("Expected integer rating")
    value = int(value)
    if not 0 <= value <= 2147483647:
        raise ValueError("Rating out of range")
    return value


def search_options(value):
    value = options_object(value)
    for key in (
        "use_and_operator",
        "only_template_documents",
        "hybrid_search",
        "rerank_results",
        "return_with_factored_fields",
    ):
        if key in value:
            value[key] = boolean(value[key])
    for key in ("max_results", "rerank_max_results", "rrf_k"):
        if key in value:
            value[key] = identifier(value[key])
    for key in (
        "categories",
        "documents",
        "documents_names",
        "relative_paths",
        "relative_path_contains",
    ):
        if key in value and value[key] is not None:
            if not isinstance(value[key], list):
                raise ValueError("Expected filter list")
            for item in value[key]:
                required_text(item)
    if "templates" in value and value["templates"] is not None:
        templates = value["templates"]
        if isinstance(templates, list):
            value["templates"] = [identifier(item) for item in templates]
        else:
            value["templates"] = identifier(templates)
    return value


def generation_options(value):
    value = options_object(value)
    value["generative_model"] = required_text(value.get("generative_model"))
    value["percentage_rank_mass"] = number(value.get("percentage_rank_mass"))
    if not 0 <= value["percentage_rank_mass"] <= 100:
        raise ValueError("Percentage out of range")
    value["use_doc_names_in_response"] = boolean(
        value.get("use_doc_names_in_response", False)
    )
    if "translate_answer" in value:
        value["translate_answer"] = boolean(value["translate_answer"])
    if value.get("translate_answer"):
        value["answer_language"] = required_text(value.get("answer_language"))
    return value


def chunk_overlap(value, max_tokens_in_chunk=None):
    """
    Validate an overlap between chunks: a whole number of tokens which is
    smaller than the chunk itself (when that chunk size is already known).
    """
    overlap = number(value)
    if not overlap.is_integer():
        raise ValueError("Invalid overlap")
    overlap = int(overlap)
    if max_tokens_in_chunk is not None and not 0 <= overlap < max_tokens_in_chunk:
        raise ValueError("Invalid overlap")
    if overlap < 0:
        raise ValueError("Invalid overlap")
    return overlap


def chunking(value):
    """
    Chunking settings of a collection: how its texts are split into chunks.

    Accepts an object (or a JSON string) with ``max_tokens_in_chunk`` and the
    optional ``number_of_overlap_tokens``, which defaults to no overlap. The
    limit imposed by the embedder of the collection is checked later, where
    that model is known.

    Returns
    -------
    dict
        ``{"max_tokens_in_chunk": int, "number_of_overlap_tokens": int}``.
    """
    value = options_object(value)
    max_tokens = identifier(value.get("max_tokens_in_chunk"))
    overlap = chunk_overlap(value.get("number_of_overlap_tokens", 0), max_tokens)
    return {
        "max_tokens_in_chunk": max_tokens,
        "number_of_overlap_tokens": overlap,
    }


def indexing_options(value):
    value = options_object(value)
    for key in (
        "prepare_proper_pages",
        "merge_document_pages",
        "clear_text",
        "use_text_denoiser",
        "check_text_lang",
    ):
        value[key] = boolean(value.get(key))
    # Optional, unlike the switches above: a client which does not know it
    # yet simply keeps the files of its uploads on the disk.
    if "delete_files_after_indexing" in value:
        value["delete_files_after_indexing"] = boolean(
            value["delete_files_after_indexing"]
        )
    # The size of a chunk belongs to the collection and may be declared there
    # once, so both keys are optional here. When they are given, they still
    # have to describe a consistent split.
    max_tokens = value.get("max_tokens_in_chunk")
    if max_tokens is None:
        value.pop("max_tokens_in_chunk", None)
    else:
        value["max_tokens_in_chunk"] = identifier(max_tokens)
    overlap = value.get("number_of_overlap_tokens")
    if overlap is None:
        value.pop("number_of_overlap_tokens", None)
    else:
        value["number_of_overlap_tokens"] = chunk_overlap(
            overlap, value.get("max_tokens_in_chunk")
        )
    return value


def identifiers(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Expected list of identifiers")
    return [identifier(item) for item in value]


def text_list(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Expected list of texts")
    return [required_text(item) for item in value]


def texts(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Expected texts list")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Expected document object")
        for key in ("filepath", "relative_filepath"):
            required_text(item.get(key))
        optional_text(item.get("category"))
        if (
            "category" not in item
            or not isinstance(item.get("pages"), list)
            or not item["pages"]
        ):
            raise ValueError("Expected document pages and category")
        for page in item["pages"]:
            if not isinstance(page, dict):
                raise ValueError("Expected page object")
            required_text(page.get("page_content"))
    return value
