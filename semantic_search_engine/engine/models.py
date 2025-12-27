from django.db import models

from data.models import (
    OrganisationUser,
    CollectionOfDocuments,
    QueryTemplate,
    DocumentPageText,
)


class UserQuery(models.Model):
    # pk: auto-id

    organisation_user = models.ForeignKey(
        OrganisationUser, null=False, on_delete=models.PROTECT
    )
    collection = models.ForeignKey(
        CollectionOfDocuments, null=False, on_delete=models.PROTECT
    )

    query_str_prompt = models.TextField(null=False, blank=False)
    query_options = models.JSONField(null=True, blank=False)

    query_templates = models.ManyToManyField(QueryTemplate)


class UserQueryResponse(models.Model):
    # pk: auto-id

    user_query = models.ForeignKey(UserQuery, null=False, on_delete=models.PROTECT)

    general_stats_json = models.JSONField(null=False)
    detailed_results_json = models.JSONField(null=False)
    structured_results = models.JSONField(null=True)


class UserQueryResponseAnswer(models.Model):
    user_response = models.ForeignKey(
        UserQueryResponse, null=False, on_delete=models.PROTECT
    )
    is_generative = models.BooleanField(null=False)
    answer_options = models.JSONField(null=False, blank=False)

    query_instruction_prompt = models.TextField(null=True)

    generated_answer = models.TextField(null=True)
    generated_answer_translated = models.TextField(null=True)

    extractive_qa_result_json = models.JSONField(null=True)
    generative_qa_result_json = models.JSONField(null=True)

    rate_value = models.IntegerField(null=True)
    rate_nax_value = models.IntegerField(null=True)
    rate_comment = models.TextField(null=True)

    generation_time = models.DurationField(null=True)
#
#
# class UserQueryResultsBox(models.Model):
#     # pk: auto-id
#
#     organisation_user = models.ForeignKey(
#         OrganisationUser, null=False, on_delete=models.PROTECT
#     )
#     collection = models.ForeignKey(
#         CollectionOfDocuments, null=False, on_delete=models.PROTECT
#     )
#
#     name = models.TextField(null=True)
#     description = models.TextField(null=True)
#     note = models.TextField(null=True)
#
#     is_private = models.BooleanField(null=False, default=True)
#
#
# class DocumentPageTextInUserQueryResultsBox(models.Model):
#     results_box = models.ForeignKey(
#         UserQueryResultsBox, null=False, on_delete=models.PROTECT
#     )
#     found_text = models.ForeignKey(
#         DocumentPageText, null=False, on_delete=models.PROTECT
#     )
#
#     # Optional fields
#     user_query_str = models.TextField(null=True)
#     user_query_response = models.ForeignKey(
#         UserQueryResponse, null=True, on_delete=models.PROTECT
#     )
#
#     class Meta:
#         unique_together = ("results_box", "found_text")
