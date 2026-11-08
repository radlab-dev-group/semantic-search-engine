import re


def source_key(source):
    if source.get("text_id") is not None:
        return (source["text_id"], source.get("text_str"))
    return tuple(
        source.get(key)
        for key in (
            "text_id",
            "document_id",
            "document_name",
            "page_number",
            "char_start",
            "char_end",
            "text_str",
        )
    )


def merge_sources(sources, registry):
    known = {source_key(source): source for source in registry}
    next_number = max((source["n"] for source in registry), default=0) + 1
    result = []
    seen = set()
    for source in sources:
        key = source_key(source)
        if key not in known:
            entry = dict(source, n=next_number)
            registry.append(entry)
            known[key] = entry
            next_number += 1
        else:
            known[key].update(
                {name: value for name, value in source.items() if name != "n"}
            )
        if key not in seen:
            result.append(dict(known[key]))
            seen.add(key)
    return result


def build_memory(history, max_turns=8, max_chars=24000, initial_registry=None):
    registry = list(initial_registry or [])
    turns = []
    pending = None
    for message in history:
        if message.role == "user":
            state = getattr(message, "state", None)
            rag = getattr(state, "rag_message_state", None)
            answer = getattr(rag, "sse_answer", None)
            metadata = getattr(answer, "generative_qa_result_json", None) or {}
            merge_sources(metadata.get("citation_registry", []), registry)
            citations = metadata.get("citations", [])
            mapped = [merge_sources([source], registry)[0] for source in citations]
            numbering = {old["n"]: new["n"] for old, new in zip(citations, mapped)}
            pending = (message.text, mapped, numbering)
        elif message.role == "assistant" and pending is not None:
            question, citations, numbering = pending
            text = re.sub(
                r"\[(\d+)\]",
                lambda match: (
                    f"[{numbering[int(match[1])]}]"
                    if int(match[1]) in numbering
                    else ""
                ),
                message.text,
            )
            turns.append(({"user": question, "assistant": text}, citations))
            pending = None

    selected = []
    remaining = max_chars
    for turn, citations in reversed(turns[-max_turns:] if max_turns else []):
        size = len(turn["user"]) + len(turn["assistant"])
        if size > remaining:
            break
        selected.append((turn, citations))
        remaining -= size
    selected.reverse()
    sources = merge_sources(
        [source for _, citations in selected for source in citations], registry
    )
    return [turn for turn, _ in selected], registry, sources


def scoped_history(history, collection_id, search_options):
    selected = []
    previous_rag = None
    pending = None
    for item in history:
        if item.role == "user":
            state = getattr(item, "state", None)
            rag = getattr(state, "rag_message_state", None)
            query = getattr(rag, "sse_query", None)
            same_collection = getattr(query, "collection_id", None) == collection_id
            same_scope = (item.search_options or {}) == (search_options or {})
            pending = (item, rag) if same_collection and same_scope else None
        elif item.role == "assistant" and pending is not None:
            selected.extend([pending[0], item])
            previous_rag = pending[1]
            pending = None
    return selected, previous_rag


def is_history_transformation(text):
    return bool(
        re.fullmatch(
            r"(?:skróć(?: to| odpowiedź)?|zrób (?:z tego |z odpowiedzi )?tabelę|"
            r"przedstaw (?:to|odpowiedź) w tabeli|"
            r"przetłumacz(?: to| odpowiedź)?(?: na [\w-]+)?|"
            r"shorten(?: it| the answer)?|make (?:it|this) a table|"
            r"translate(?: it| the answer)?(?: (?:to|into) [\w-]+)?)"
            r"[.!?\s]*",
            text.strip(),
            re.IGNORECASE,
        )
    )
