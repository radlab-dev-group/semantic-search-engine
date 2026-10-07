import json
import math
from collections.abc import Mapping


def request_params(request):
    if hasattr(request, "validated_params"):
        return request.validated_params
    body = request.data
    params = dict(body.items()) if isinstance(body, Mapping) else {}
    if request.method == "GET":
        params.update(request.query_params.items())
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
    for key in ("use_and_operator", "only_template_documents", "hybrid_search",
                "rerank_results", "return_with_factored_fields"):
        if key in value:
            value[key] = boolean(value[key])
    for key in ("max_results", "rerank_max_results", "rrf_k"):
        if key in value:
            value[key] = identifier(value[key])
    for key in ("categories", "documents", "documents_names", "relative_paths",
                "relative_path_contains"):
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
    value["use_doc_names_in_response"] = boolean(value.get("use_doc_names_in_response", False))
    if "translate_answer" in value:
        value["translate_answer"] = boolean(value["translate_answer"])
    if value.get("translate_answer"):
        value["answer_language"] = required_text(value.get("answer_language"))
    return value


def indexing_options(value):
    value = options_object(value)
    for key in ("prepare_proper_pages", "merge_document_pages", "clear_text",
                "use_text_denoiser", "check_text_lang"):
        value[key] = boolean(value.get(key))
    value["max_tokens_in_chunk"] = identifier(value.get("max_tokens_in_chunk"))
    overlap = number(value.get("number_of_overlap_tokens"))
    if not overlap.is_integer() or not 0 <= overlap < value["max_tokens_in_chunk"]:
        raise ValueError("Invalid overlap")
    value["number_of_overlap_tokens"] = int(overlap)
    return value


def texts(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Expected texts list")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Expected document object")
        for key in ("filepath", "relative_filepath"):
            required_text(item.get(key))
        optional_text(item.get("category"))
        if "category" not in item or not isinstance(item.get("pages"), list) or not item["pages"]:
            raise ValueError("Expected document pages and category")
        for page in item["pages"]:
            if not isinstance(page, dict):
                raise ValueError("Expected page object")
            required_text(page.get("page_content"))
    return value