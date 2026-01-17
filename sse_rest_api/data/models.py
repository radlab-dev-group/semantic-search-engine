import django
from django.db import models

from system.models import OrganisationUser, OrganisationGroup, Organisation


class CollectionOfDocuments(models.Model):
    """
    Collection of documents
    """

    # pk: auto id

    name = models.TextField(null=False)
    display_name = models.TextField(null=False)

    description = models.TextField(null=True)
    model_embedder = models.TextField(null=True)
    model_reranker = models.TextField(null=True)
    model_embedder_vector_size = models.IntegerField(null=True)
    embedder_index_type = models.TextField(null=True)
    embedder_index_params = models.TextField(null=True)
    embedder_search_params = models.TextField(null=True)

    created_on = models.DateTimeField(default=django.utils.timezone.now, null=False)

    created_by = models.ForeignKey(
        OrganisationUser, on_delete=models.PROTECT, null=False
    )
    visible_to_groups = models.ManyToManyField(OrganisationGroup)

    class Meta:
        unique_together = ("name", "created_by")


class UploadedDocuments(models.Model):
    """
    Paths to directories where files were uploaded.
    """

    # pk: auto-id
    dir_hash = models.TextField(null=False, unique=True)
    dir_path = models.TextField(null=False)

    created_on = models.DateTimeField(default=django.utils.timezone.now, null=False)
    is_indexed = models.BooleanField(default=False, null=False)

    begin_indexing_time = models.DateTimeField(null=True)
    end_indexing_time = models.DateTimeField(null=True)
    number_of_uploaded_documents = models.IntegerField(null=True)
    number_of_indexed_documents_rel_db = models.IntegerField(null=True)
    number_of_indexed_documents_vec_db = models.IntegerField(null=True)

    uploaded_by = models.ForeignKey(
        OrganisationUser, on_delete=models.PROTECT, null=False
    )


class Document(models.Model):
    # pk: auto-id
    name = models.TextField(null=False)

    collection = models.ForeignKey(
        CollectionOfDocuments, on_delete=models.PROTECT, null=False
    )

    # Path to file is unique, in case when file comes from same path
    # but different machines, then path have to be set as:
    # machine.url:/path/to/file
    path = models.TextField(null=False)
    relative_path = models.TextField(null=False)
    # Any Hash calculated on the document, f.e. md5 of stringify metadata_json
    document_hash = models.TextField(null=False)
    # Use document during search
    use_in_search = models.BooleanField(default=True, null=False)

    created_on = models.DateTimeField(default=django.utils.timezone.now, null=False)

    category = models.TextField(null=True)
    metadata_json = models.JSONField(null=True)
    upload_process = models.ForeignKey(
        UploadedDocuments, on_delete=models.PROTECT, null=True
    )

    added_by = models.ForeignKey(
        OrganisationUser, on_delete=models.PROTECT, null=False
    )

    visible_to_groups = models.ManyToManyField(OrganisationGroup)

    class Meta:
        unique_together = ("name", "collection", "path", "document_hash")


class DocumentPage(models.Model):
    # pk: auto-id
    page_number = models.IntegerField(null=False)
    document = models.ForeignKey(Document, on_delete=models.PROTECT, null=False)

    created_on = models.DateTimeField(default=django.utils.timezone.now, null=False)

    class Meta:
        unique_together = ("document", "page_number")


class DocumentPageText(models.Model):
    # pk: auto - id
    text_number = models.IntegerField(null=False)
    page = models.ForeignKey(DocumentPage, on_delete=models.PROTECT, null=False)

    text_str = models.TextField(null=False)
    text_str_clear = models.TextField(null=True)
    text_hash = models.TextField(null=False)
    text_chunk_type = models.TextField(null=False)
    language = models.TextField(null=False)

    metadata_json = models.JSONField(null=True)

    # When document is merged to single page and split to f.e. tokens chunks
    # this field may be helpful to connect chunks from merged document to chunks
    # prepared from proper pages -- then each chunk may be mapped to other
    paged_text = models.ForeignKey(
        "DocumentPageText", on_delete=models.PROTECT, null=True
    )

    class Meta:
        unique_together = ("text_number", "page", "text_chunk_type")


class CollectionOfQueryTemplates(models.Model):
    """
    Collection of templates
    """

    # pk: auto id
    name = models.TextField(null=False)
    created_on = models.DateTimeField(default=django.utils.timezone.now, null=False)
    organisation = models.ForeignKey(
        Organisation, on_delete=models.PROTECT, null=False
    )

    class Meta:
        unique_together = ("name", "organisation")


class QueryTemplateGrammar(models.Model):
    template_collection = models.OneToOneField(
        CollectionOfQueryTemplates, on_delete=models.PROTECT, null=False, unique=True
    )

    tokens = models.JSONField(null=False)
    alphabet = models.JSONField(null=False)


class QueryTemplate(models.Model):
    """
    Single template definition (one-to-many QueryTemplateGrammar)
    """

    # pk: auto id
    name = models.TextField(null=False)
    is_active = models.BooleanField(null=False, default=True)

    display = models.TextField(null=True)
    data_connector = models.JSONField(null=True)
    data_filter_expressions = models.JSONField(null=True)
    structured_response_if_exists = models.BooleanField(null=True)
    structured_response_data_fields = models.JSONField(null=True)

    system_prompt = models.TextField(null=True)

    template_grammar = models.ForeignKey(
        QueryTemplateGrammar, on_delete=models.PROTECT, null=False
    )

    class Meta:
        unique_together = ("name", "template_grammar")
