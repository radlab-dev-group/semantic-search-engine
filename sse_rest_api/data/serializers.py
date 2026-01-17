from rest_framework import serializers


from data.models import (
    CollectionOfDocuments,
    UploadedDocuments,
    Document,
    QueryTemplate,
)


class CollectionOfDocumentsSerializer(serializers.ModelSerializer):
    class Meta:
        model = CollectionOfDocuments
        fields = ["id", "name", "display_name", "description", "created_on"]


class UploadedDocumentsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedDocuments
        fields = "__all__"


class SimpleDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["name"]


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
