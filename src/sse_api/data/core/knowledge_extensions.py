"""Pure helpers for the version-one agentic knowledge contract."""


def lexical_query(question, keywords):
    return " ".join(keywords) if keywords else question


def intersect_filter_groups(groups, resolve):
    sets = [set(resolve(group)) for group in groups]
    return list(set.intersection(*sets)) if sets else []


def merge_quality(metadata, upvotes, downvotes):
    if any(type(v) is not int or v < 0 or v > 10**12 for v in (upvotes, downvotes)):
        raise ValueError("Invalid quality counter")
    result = dict(metadata or {})
    old_up = result.get("upvotes", 0)
    old_down = result.get("downvotes", 0)
    up = max(upvotes, old_up if type(old_up) is int and old_up >= 0 else 0)
    down = max(downvotes, old_down if type(old_down) is int and old_down >= 0 else 0)
    result.update(
        upvotes=up,
        downvotes=down,
        times_used=up + down,
        success_rate=(up + 1) / (up + down + 2),
        confidence_score=(up + 1) / (up + down + 2),
    )
    return result


def keyword_match_mode(value="and"):
    """Accept exactly the public API's two modes; omission means AND."""
    if not isinstance(value, str) or value not in ("and", "any"):
        raise ValueError("keywords_match must be 'and' or 'any'")
    return value


def build_fts_query(query_str, keywords, keywords_match, config, query_factory):
    """OR/AND between keyword entries, with plain FTS parsing inside each.

    Never interpolate untrusted keywords into a raw tsquery. Without keywords
    the historical query_str/plain behavior stays unchanged, even in any mode.
    """
    mode = keyword_match_mode(keywords_match)
    if not keywords:
        return query_factory(query_str, config=config, search_type="plain")
    query = None
    for keyword in keywords:
        part = query_factory(keyword, config=config, search_type="plain")
        query = (
            part
            if query is None
            else (query | part if mode == "any" else query & part)
        )
    return query
