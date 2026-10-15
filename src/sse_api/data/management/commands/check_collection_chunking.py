"""
check_collection_chunking
-------------------------

Audits how the texts of every collection are split.

A collection declares its chunking once (``metadata_json``) and the indexing
calls are resolved against it, so chunks of one collection should be built the
same way.  This command reports what a collection declares and - with
``--verify`` - measures the chunks which are really in the database with the
tokenizer of the embedder of the collection.  Chunks longer than the declared
maximum come from an indexing call which used other settings.

Examples::

    python manage.py check_collection_chunking
    python manage.py check_collection_chunking --collection my_docs --verify
    python manage.py check_collection_chunking --json
"""

import json
import logging
from typing import Any, Dict, List, Optional

from django.core.management.base import BaseCommand
from django.db.models import Count

from sse_api.data.controllers.chunking import embedder_max_length
from sse_api.data.core.constants import MAX_TOKENS_IN_CHUNK_KEY
from sse_api.data.models import CollectionOfDocuments, DocumentPageText
from sse_api.engine.core.config.models_config import EmbeddingModelsConfig

logger = logging.getLogger(__name__)


def text_of(chunk: DocumentPageText) -> str:
    """The text which goes to the embedder: the cleared one when it exists."""
    if chunk.text_str_clear and len(chunk.text_str_clear):
        return chunk.text_str_clear
    return chunk.text_str or ""


def load_tokenizer(model_embedder: Optional[str]):
    """
    Tokenizer of the embedder of a collection, ``None`` when unavailable.

    A missing model or a broken configuration must not stop the audit; the
    sizes of the chunks are simply not measured then.
    """
    if not model_embedder:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            EmbeddingModelsConfig.get_embedder_path(model_embedder)
        )
    except Exception:  # missing model files, no network, broken config
        logger.warning(
            "Tokenizer of %r is not available, chunk sizes are not measured",
            model_embedder,
            exc_info=False,
        )
        return None


def measure_chunks(
    collection: CollectionOfDocuments, tokenizer, max_tokens, limit=0
):
    """
    Compare the chunks of a collection with its declared maximum.

    Returns
    -------
    dict or None
        ``None`` when nothing could be measured, otherwise the number of the
        checked chunks, the longest one, the mean length and how many chunks
        are longer than the declared maximum.
    """
    lengths = []
    chunks = DocumentPageText.objects.filter(page__document__collection=collection)
    for chunk in chunks.iterator():
        text = text_of(chunk)
        if not len(text):
            continue
        lengths.append(len(tokenizer(text).input_ids))
        if limit and len(lengths) >= limit:
            break
    if not lengths:
        return None
    return {
        "chunks_checked": len(lengths),
        "longest_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 1),
        "above_declared": len(
            [
                length
                for length in lengths
                if max_tokens is not None and length > max_tokens
            ]
        ),
    }


class Command(BaseCommand):
    help = (
        "Report the chunking settings of the collections and, with --verify, "
        "the chunks which do not match them."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--collection",
            nargs="+",
            metavar="NAME",
            help="Check only those collections (default: all of them)",
        )
        parser.add_argument(
            "--verify",
            action="store_true",
            help="Measure the chunks with the tokenizer of the embedder",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="How many chunks per collection to measure (0 = all)",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            dest="as_json",
            help="Print the report as JSON instead of lines of text",
        )

    def handle(self, *args, **options):
        collections = self.get_collections(options.get("collection"))
        report = [self.describe(collection, options) for collection in collections]
        if options.get("as_json"):
            self.stdout.write(json.dumps(report, indent=2, ensure_ascii=False))
            return
        self.write_lines(report)

    @staticmethod
    def get_collections(names: Optional[List[str]]):
        collections = CollectionOfDocuments.objects.all()
        if names:
            collections = collections.filter(name__in=names)
        return collections.order_by("name")

    def describe(self, collection: CollectionOfDocuments, options) -> Dict[str, Any]:
        """One collection of the report."""
        chunking = collection.chunking()
        max_tokens = chunking.get(MAX_TOKENS_IN_CHUNK_KEY)
        stats = collection.document_set.aggregate(documents=Count("id"))
        entry: Dict[str, Any] = {
            "collection": collection.name,
            "declared_chunking": chunking or None,
            "with_defaults": collection.chunking_with_defaults(),
            "strict_chunking": collection.strict_chunking,
            "model_embedder": collection.model_embedder,
            "model_reranker": collection.model_reranker,
            "embedder_max_model_length": embedder_max_length(
                collection.model_embedder
            ),
            "documents": stats["documents"],
            "chunks": DocumentPageText.objects.filter(
                page__document__collection=collection
            ).count(),
        }
        entry["chunk_too_long_for_embedder"] = (
            max_tokens is not None
            and entry["embedder_max_model_length"] is not None
            and max_tokens > entry["embedder_max_model_length"]
        )
        if options.get("verify"):
            tokenizer = load_tokenizer(collection.model_embedder)
            entry["measured"] = (
                measure_chunks(
                    collection,
                    tokenizer,
                    max_tokens,
                    limit=options.get("limit") or 0,
                )
                if tokenizer is not None
                else None
            )
        return entry

    def write_lines(self, report: List[Dict[str, Any]]) -> None:
        if not report:
            self.stdout.write("No collections.")
            return
        for entry in report:
            declared = entry["declared_chunking"]
            self.stdout.write(
                f"{entry['collection']}: "
                f"chunking "
                f"{declared if declared else 'not declared (defaults apply)'}, "
                f"strict={entry['strict_chunking']}, "
                f"documents={entry['documents']}, chunks={entry['chunks']}"
            )
            if entry["chunk_too_long_for_embedder"]:
                self.stdout.write(
                    self.style.WARNING(
                        f"  {declared[MAX_TOKENS_IN_CHUNK_KEY]} tokens do not "
                        f"fit {entry['model_embedder']} "
                        f"(max_model_length="
                        f"{entry['embedder_max_model_length']}); the end of "
                        "every chunk is cut off by the model"
                    )
                )
            measured = entry.get("measured")
            if measured is None:
                continue
            self.stdout.write(
                f"  measured {measured['chunks_checked']} chunks: longest "
                f"{measured['longest_tokens']} tokens, mean "
                f"{measured['mean_tokens']}"
            )
            if measured["above_declared"]:
                self.stdout.write(
                    self.style.WARNING(
                        f"  {measured['above_declared']} chunks are longer "
                        "than the chunking of the collection"
                    )
                )
