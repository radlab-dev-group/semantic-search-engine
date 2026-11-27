import json
import math
from collections.abc import Mapping

from sse_api.data.core.knowledge_extensions import keyword_match_mode


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


#: Strategies of merging results of a multi-collection search.  ``global_limit``
#: is the default: it keeps the LLM context from growing with the number of
#: selected collections.
MULTI_COLLECTION_MODES = ("per_collection", "global_limit")
DEFAULT_MULTI_COLLECTION_MODE = "global_limit"


def multi_collection_mode(value):
    if not isinstance(value, str) or value.strip() not in MULTI_COLLECTION_MODES:
        raise ValueError("Expected per_collection or global_limit")
    return value.strip()


def collection_names(value):
    """Validate the ``collection_names`` parameter of a search request.

    The UI sends a JSON-encoded list in a form field; JSON clients can send a
    real list.  A single (unquoted) collection name is accepted as a one-item
    list so the endpoint stays friendly to hand-written requests.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            parsed = [value]
        if isinstance(parsed, str):
            parsed = [parsed]
    elif isinstance(value, (list, tuple)):
        parsed = list(value)
    else:
        raise ValueError("Expected list of collection names")
    if not isinstance(parsed, list):
        raise ValueError("Expected list of collection names")
    cleaned = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Expected nonempty string collection name")
        cleaned.append(item.strip())
    return cleaned


def required_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty text")
    return value


def optional_text(value):
    if value is not None and not isinstance(value, str):
        raise ValueError("Expected text")
    return value


def conversation_history(value):
    if not isinstance(value, list):
        raise ValueError("Expected a list of conversation turns")
    for turn in value:
        if (
            not isinstance(turn, dict)
            or set(turn) != {"user", "assistant"}
            or not all(isinstance(text, str) for text in turn.values())
        ):
            raise ValueError("Expected user and assistant text in each turn")
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
        "lexical_only",
        "rerank_results",
        "return_with_factored_fields",
    ):
        if key in value:
            value[key] = boolean(value[key])
    for key in ("max_results", "rerank_max_results", "rrf_k"):
        if key in value:
            value[key] = identifier(value[key])
    for key in ("dense_weight", "lexical_weight"):
        if key in value and value[key] is not None:
            weight = number(value[key])
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"{key} must be between 0.0 and 1.0")
            value[key] = float(weight)
    if "min_similarity" in value and value["min_similarity"] is not None:
        similarity = number(value["min_similarity"])
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")
        value["min_similarity"] = float(similarity)
    if "max_chunks_per_document" in value:
        value["max_chunks_per_document"] = identifier(
            value["max_chunks_per_document"]
        )
    for key in (
        "categories",
        "documents",
        "documents_names",
        "relative_paths",
        "relative_path_contains",
        "keywords",
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
    if "keywords_match" in value:
        value["keywords_match"] = keyword_match_mode(value["keywords_match"])
    if "multi_collection_mode" in value and value["multi_collection_mode"] is not None:
        value["multi_collection_mode"] = multi_collection_mode(
            value["multi_collection_mode"]
        )
    groups = value.get("metadata_filter_groups", [])
    if not isinstance(groups, list) or len(groups) > 8:
        raise ValueError("Expected up to eight metadata filter groups")
    for group in groups:
        if not isinstance(group, list) or not 1 <= len(group) <= 32:
            raise ValueError("Expected nonempty metadata filter group")
        for expr in group:
            if (
                not isinstance(expr, dict)
                or expr.get("operator")
                not in ("eq", "ne", "in", "hse", "gt", "lt", "ge", "le")
                or not isinstance(expr.get("field"), dict)
                or len(expr["field"]) != 1
            ):
                raise ValueError("Invalid metadata expression")
    return value


#: Aliases of the sampling parameters of a generation request.
GENERATION_OPTION_ALIASES = {
    "rep_penalty": "repetition_penalty",
    "topp": "top_p",
    "topk": "top_k",
    "typicalp": "typical_p",
    "max_tokens": "max_new_tokens",
}


def _canonical_generation_options(value):
    """``value`` with the aliases of the sampling parameters renamed."""
    for alias, canonical in GENERATION_OPTION_ALIASES.items():
        if alias in value and canonical not in value:
            value[canonical] = value.pop(alias)
    return value


def _validate_sampling_options(value):
    """
    Check the ranges of the sampling parameters of a generation request.

    Recognised (after the alias renaming): ``temperature``,
    ``repetition_penalty``, ``top_p``, ``top_k``, ``typical_p`` and
    ``max_new_tokens``.  Absent keys are left alone, so the same helper
    serves a per-request query and the profile of a collection.
    """
    if "temperature" in value and value["temperature"] is not None:
        val = number(value["temperature"])
        if not 0.0 <= val <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        value["temperature"] = float(val)
    if "repetition_penalty" in value and value["repetition_penalty"] is not None:
        val = number(value["repetition_penalty"])
        if not 0.1 <= val <= 10.0:
            raise ValueError("repetition_penalty must be between 0.1 and 10.0")
        value["repetition_penalty"] = float(val)
    if "top_p" in value and value["top_p"] is not None:
        val = number(value["top_p"])
        if not 0.0 < val <= 1.0:
            raise ValueError("top_p must be between 0.0 (exclusive) and 1.0")
        value["top_p"] = float(val)
    if "top_k" in value and value["top_k"] is not None:
        val = number(value["top_k"])
        if val < 0 or not float(val).is_integer():
            raise ValueError("top_k must be a non-negative integer")
        value["top_k"] = int(val)
    if "typical_p" in value and value["typical_p"] is not None:
        val = number(value["typical_p"])
        if not 0.0 < val <= 1.0:
            raise ValueError("typical_p must be between 0.0 (exclusive) and 1.0")
        value["typical_p"] = float(val)
    if "max_new_tokens" in value and value["max_new_tokens"] is not None:
        val = number(value["max_new_tokens"])
        if val <= 0 or not float(val).is_integer():
            raise ValueError("max_new_tokens must be a positive integer")
        value["max_new_tokens"] = int(val)
    return value


def generation_options(value):
    value = options_object(value)
    value = _canonical_generation_options(value)
    value["generative_model"] = required_text(value.get("generative_model"))
    _validate_sampling_options(value)
    value["percentage_rank_mass"] = number(value.get("percentage_rank_mass"))
    if not 0 <= value["percentage_rank_mass"] <= 100:
        raise ValueError("Percentage out of range")
    value["use_doc_names_in_response"] = boolean(
        value.get("use_doc_names_in_response", False)
    )
    if "system_prompt_with_enumerated_documents" in value:
        value["system_prompt_with_enumerated_documents"] = boolean(
            value["system_prompt_with_enumerated_documents"]
        )
    if "translate_answer" in value:
        value["translate_answer"] = boolean(value["translate_answer"])
    if value.get("translate_answer"):
        value["answer_language"] = required_text(value.get("answer_language"))
    return value


def collection_search_options(value):
    """
    The search profile saved on a collection.

    The same vocabulary as the ``options`` of a search request, except that
    every key is optional: a profile stores only what the collection wants
    to pin, and the engine fills the rest with its own defaults.
    """
    value = options_object(value)
    for key in (
        "use_and_operator",
        "only_template_documents",
        "hybrid_search",
        "lexical_only",
        "rerank_results",
        "return_with_factored_fields",
    ):
        if key in value and value[key] is not None:
            value[key] = boolean(value[key])
    for key in (
        "max_results",
        "rerank_max_results",
        "rrf_k",
        "max_chunks_per_document",
    ):
        if key in value and value[key] is not None:
            value[key] = identifier(value[key])
    for key in ("dense_weight", "lexical_weight", "min_similarity"):
        if key in value and value[key] is not None:
            val = number(value[key])
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{key} must be between 0.0 and 1.0")
            value[key] = float(val)
    return value


def collection_generation_options(value):
    """
    The RAG profile saved on a collection.

    The sampling parameters (temperature, repetition penalty, top-p, top-k,
    typical-p, maximum new tokens) with the same aliases and ranges as a
    per-request ``query_options``.  ``generative_model`` is optional here,
    because a collection without a pinned model answers with the model the
    request names, or the default of the router.
    """
    value = options_object(value)
    value = _canonical_generation_options(value)
    if "generative_model" in value and value["generative_model"] is not None:
        value["generative_model"] = required_text(value["generative_model"])
    _validate_sampling_options(value)
    if "percentage_rank_mass" in value and value["percentage_rank_mass"] is not None:
        val = number(value["percentage_rank_mass"])
        if not 0 <= val <= 100:
            raise ValueError("percentage_rank_mass must be between 0 and 100")
        value["percentage_rank_mass"] = int(val)
    if "system_prompt" in value and value["system_prompt"] is not None:
        value["system_prompt"] = str(value["system_prompt"]).strip()
    if "answer_language" in value and value["answer_language"] is not None:
        value["answer_language"] = required_text(value["answer_language"])
    if "translate_answer" in value and value["translate_answer"] is not None:
        value["translate_answer"] = boolean(value["translate_answer"])
    if (
        "use_doc_names_in_response" in value
        and value["use_doc_names_in_response"] is not None
    ):
        value["use_doc_names_in_response"] = boolean(
            value["use_doc_names_in_response"]
        )
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
