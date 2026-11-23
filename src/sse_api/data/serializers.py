from rest_framework import serializers


from sse_api.data.models import (
    CollectionOfDocuments,
    UploadedDocuments,
    Document,
    QueryTemplate,
)
from django.db.models import Q


class CollectionOfDocumentsSerializer(serializers.ModelSerializer):
    """
    A collection as listed by ``GET collections``.

    ``metadata_json`` carries the chunking settings of the collection
    (``chunking`` and ``strict_chunking``), so the UI can prefill its
    indexing form from what the collection already uses.

    ``default_search_options`` and ``default_generation_options`` are the
    profiles of the collection (the hyperparameters of the search and the
    language model which answers about it) completed with the defaults of
    the engine, so a client always reads a complete set.

    ``document_count`` is every document of the collection and
    ``active_document_count`` the documents which take part in the search
    (``use_in_search=True``).  Both are read from the annotation of
    ``RelationalDBController.with_document_counts``; a collection which
    arrives without it is counted on its own, so the answer never lies.
    """

    document_count = serializers.SerializerMethodField()
    active_document_count = serializers.SerializerMethodField()
    default_search_options = serializers.SerializerMethodField()
    default_generation_options = serializers.SerializerMethodField()

    class Meta:
        model = CollectionOfDocuments
        fields = [
            "id",
            "name",
            "display_name",
            "description",
            "model_embedder",
            "model_reranker",
            "created_on",
            "metadata_json",
            "document_count",
            "active_document_count",
            "default_search_options",
            "default_generation_options",
        ]

    def get_document_count(self, collection: CollectionOfDocuments) -> int:
        return self._count(collection, "document_count", Q())

    def get_default_search_options(self, collection: CollectionOfDocuments) -> dict:
        """Search profile of the collection with the defaults of the engine."""
        return collection.search_options_with_defaults()

    def get_default_generation_options(
        self, collection: CollectionOfDocuments
    ) -> dict:
        """RAG profile of the collection with the defaults of the engine."""
        return collection.generation_options_with_defaults()

    def get_active_document_count(self, collection: CollectionOfDocuments) -> int:
        return self._count(
            collection, "active_document_count", Q(use_in_search=True)
        )

    @staticmethod
    def _count(
        collection: CollectionOfDocuments, annotation: str, filters: Q
    ) -> int:
        counted = getattr(collection, annotation, None)
        if counted is not None:
            return int(counted)
        return (
            Document.objects.filter(collection=collection.pk).filter(filters).count()
        )


class UploadedDocumentsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedDocuments
        fields = "__all__"


class SimpleDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["name"]


class DocumentVisibilitySerializer(serializers.ModelSerializer):
    """Name of the document, whether it takes part in the search, and how far
    its indexing got (``indexing_status`` is ``None`` for documents indexed
    before the indexing progress was kept).
    """

    class Meta:
        model = Document
        fields = ["id", "name", "use_in_search", "indexing_status"]


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = [
            "id",
            "name",
            "path",
            "relative_path",
            "document_hash",
            "metadata_json",
        ]


class SimpleQueryTemplateSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryTemplate
        fields = ["id", "name", "display"]
