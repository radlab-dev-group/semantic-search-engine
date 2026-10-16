from rest_framework import serializers


from sse_api.data.models import (
    CollectionOfDocuments,
    UploadedDocuments,
    Document,
    QueryTemplate,
)


class CollectionOfDocumentsSerializer(serializers.ModelSerializer):
    """
    A collection as listed by ``GET collections``.

    ``metadata_json`` carries the chunking settings of the collection
    (``chunking`` and ``strict_chunking``), so the UI can prefill its
    indexing form from what the collection already uses.
    """

    class Meta:
        model = CollectionOfDocuments
        fields = [
            "id",
            "name",
            "display_name",
            "description",
            "created_on",
            "metadata_json",
        ]


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
