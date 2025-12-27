import os
import json
import datetime
import dateutil

from django.db.models import QuerySet

from data.models import Document

from system.models import Organisation, OrganisationUser
from data.models import (
    CollectionOfQueryTemplates,
    QueryTemplateGrammar,
    QueryTemplate,
)
from data.controllers.constants import VALUE_OF_DATA_EVAL_EXPRESSION


class QueryTemplatesSearchGrammar:
    class Actions:
        SHOW_WHOLE = "show_whole"
        END_DATE_OLDER_THAN_TODAY = "end_date_older_than_today"

    class Types:
        EVENT = "event"
        ADDRESS = "address"
        OTHER = "other"

    class JsonFields:
        SEARCH = "search"
        PRESENTATION = "presentation"

    class MetadataFields:
        document_date = "date"
        document_type = "type"

    ALL_POSSIBLE_TYPE_ACTIONS = [
        Actions.END_DATE_OLDER_THAN_TODAY,
        Actions.SHOW_WHOLE,
    ]
    ALL_POSSIBLE_TYPE_ACTIONS_SEARCH = [Actions.END_DATE_OLDER_THAN_TODAY]
    ALL_POSSIBLE_TYPE_ACTIONS_PRESENTATION = [Actions.SHOW_WHOLE]

    ALL_POSSIBLE_DATA_TYPES = [Types.EVENT, Types.ADDRESS, Types.OTHER]

    CONSTANT_TYPE_ACTIONS = {
        Types.EVENT: {
            JsonFields.SEARCH: [Actions.END_DATE_OLDER_THAN_TODAY],
            JsonFields.PRESENTATION: [Actions.SHOW_WHOLE],
        },
        Types.ADDRESS: {
            JsonFields.SEARCH: [],
            JsonFields.PRESENTATION: [Actions.SHOW_WHOLE],
        },
        Types.OTHER: {
            JsonFields.SEARCH: [],
            JsonFields.PRESENTATION: [],
        },
    }

    class GrammarFunctions:
        def __init__(self):
            self._actions_mapping = {
                QueryTemplatesSearchGrammar.Actions.END_DATE_OLDER_THAN_TODAY: self.__end_date_is_older_than_today
            }

        def accept_document(self, action: str, document: Document) -> bool:
            assert action in self._actions_mapping
            if (
                action
                not in QueryTemplatesSearchGrammar.ALL_POSSIBLE_TYPE_ACTIONS_SEARCH
            ):
                raise Exception(f"Invalid action {action} to accept document!")

            return self._actions_mapping[action](document)

        def __end_date_is_older_than_today(self, d: Document) -> bool:
            metadata = d.metadata_json
            if metadata is None or not len(metadata):
                return False

            document_date = metadata.get(
                QueryTemplatesSearchGrammar.MetadataFields.document_date, {}
            )
            end_date_str = document_date.get("end", "").strip()

            if not len(end_date_str):
                return False

            # end_datetime = self.__datetime_from_str(date_str=end_date_str)
            end_datetime = (
                dateutil.parser.isoparse(end_date_str)
                .astimezone()
                .replace(hour=0, minute=0, second=0, microsecond=0)
            )

            return end_datetime >= datetime.datetime.now().astimezone().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        @staticmethod
        def __datetime_from_str(date_str: str) -> datetime:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    def __init__(self, use_metadata: bool = True):
        """
        Set all params with default values.

        :param use_metadata: Document info is stored in json metadata:
         - type of document: `metadata_json.type`
         - date: `metadata_json.date` and is split to: <date.begin, date.end>.
        """
        self.use_metadata_when_templating = use_metadata
        self._gf = self.GrammarFunctions()

        assert (
            self.use_metadata_when_templating is True
        ), "Templates works only with metadata now!"

    def use_document_in_sse(
        self, document: Document, skip_if_any_problem: bool = True
    ) -> bool:
        """
        Main function o check if document have to be accepted to use in search engine
        :param document: Document to check
        :param skip_if_any_problem: Skip document if any problem, don't raise exception.
        :return: True if document has to be accepted, otherwise False.
        """
        document_type = self.__get_document_type(document)
        if not skip_if_any_problem:
            assert document_type in self.CONSTANT_TYPE_ACTIONS

        type_actions = self.CONSTANT_TYPE_ACTIONS.get(document_type, {})
        if not len(type_actions):
            return False

        for action in type_actions[QueryTemplatesSearchGrammar.JsonFields.SEARCH]:
            try:
                if not self._gf.accept_document(action, document):
                    return False
            except Exception as e:
                if not skip_if_any_problem:
                    raise e
        return True

    def get_document_or_chunk(self):
        pass

    def __get_document_type(self, document: Document) -> str | None:
        if self.use_metadata_when_templating:
            return document.metadata_json.get(
                QueryTemplatesSearchGrammar.MetadataFields.document_type, None
            )
        else:
            raise Exception("Document type may be defined only in metadata!")


class QueryTemplateFilterer:
    def __init__(self):
        pass

    @staticmethod
    def use_document_in_sse(
        query_template: QueryTemplate, document: Document
    ) -> bool:
        doc_metadata = document.metadata_json
        if doc_metadata is None or not len(doc_metadata):
            # print("No metadata!")
            # print("No metadata!")
            # print("No metadata!")
            # print("No metadata!")
            return False

        filter_opts = query_template.data_filter_expressions
        # NOTE:
        # When no filtering options into query template is defined
        # then each document should be accepted
        if filter_opts is None or not len(filter_opts):
            # print("=" * 50)
            # print("No filter opts!")
            # print("No filter opts!")
            # print("No filter opts!")
            # print("No filter opts!")
            # print("=" * 50)
            return True

        all_constraints_ok = False
        for var1, expr1 in filter_opts.items():
            # print("use_document_in_sse, var1: ", var1)
            if doc_metadata.get(var1, None) is None:
                # print("\t->skipped")
                continue

            doc_values = []
            expressions = []
            if type(expr1) in [dict]:
                # print(". - . - dict expression")
                """
                "data_filter_expressions": {
                    "date": {
                        "begin": "datetime.datetime.strptime('DATA_VALUE', '%Y-%m-%d') <= datetime.datetime.now() + datetime.timedelta(days=8)",
                        "end": "datetime.datetime.strptime('DATA_VALUE', '%Y-%m-%d') >= datetime.datetime.now()"
                    }
                },
                """
                for var2, val2 in expr1.items():
                    # print("use_document_in_sse, var2: ", var2)
                    if type(val2) in [dict]:
                        raise Exception("Not supported expression with nested dict!")
                    if doc_metadata[var1].get(var2, None) is None:
                        # print(f"NOT FOUND {var2} (with key {var1})")
                        continue
                    # print(f"\t ok -> {var2} (with key {var1})")
                    expressions.append(val2)
                    doc_values.append(doc_metadata[var1][var2])
            else:
                expressions.append(expr1)
                doc_values.append(doc_metadata[var1])

            # print("\\/" * 40)
            # print("expressions=", expressions)
            # print("doc_values=", doc_values)
            # print("/\\" * 40)

            if not len(doc_values) or not len(expressions):
                continue

            # print("-=" * 50)
            # print("type(expressions): ", type(expressions))
            # print("len(expressions): ", len(expressions))
            # print("expressions: ", expressions)
            # print("type(doc_values): ", type(doc_values))
            # print("len(doc_values): ", len(doc_values))
            # print("doc_values: ", doc_values)
            # print("-=" * 50)

            for expression, doc_value in zip(expressions, doc_values):
                # print("applying expression: ", expression, "on data:", doc_value)
                expression_to_eval = expression.replace(
                    VALUE_OF_DATA_EVAL_EXPRESSION, doc_value
                )
                # print("expression_to_eval=", expression_to_eval)
                try:
                    accept_document = eval(expression_to_eval)
                except Exception as e:
                    accept_document = False
                    print("=" * 100)
                    print(e)
                    print("Error while eval:", expression_to_eval)
                    print("=" * 100)
                if not accept_document:
                    return False

                # print("= " * 50)
                # print("query_template=", query_template.name)
                # print("doc_value=", doc_value)
                # print("expression=", expression)
                # print("expression_to_eval=", expression_to_eval)
                # print("accept_document (eval result)=", accept_document)
                # print("= " * 50)

                all_constraints_ok = True
        return all_constraints_ok


class QueryTemplateConfigReader:
    def __init__(self, config_path: str = "configs/query-templates.json"):
        self.config_path = config_path

        self._whole_config = None
        self.template_name = None
        self.query_templates = None
        self.templates_grammar = None

        self.__load()

    def __load(self):
        with open(self.config_path, "r") as f:
            self._whole_config = json.load(f)

        self.template_name = self._whole_config["template_name"].strip()
        self.query_templates = self._whole_config["query_templates"]
        self.templates_grammar = self._whole_config["templates_grammar"]

        assert len(
            self.template_name
        ), "Length of template_name must be greater than 0!"


class QueryTemplatesLoaderController:
    def __init__(self, config_path: str = "configs/query-templates.json"):
        self.config_reader = QueryTemplateConfigReader(config_path)

    def add_templates_to_organisation(self, organisation: Organisation):
        print("Adding template collection", self.config_reader.template_name)
        return self.__get_add_collection_of_templates_to_organisation(
            organisation=organisation
        )

    def __get_add_collection_of_templates_to_organisation(
        self, organisation: Organisation
    ) -> CollectionOfQueryTemplates or None:
        if (
            self.config_reader.template_name is None
            or not len(self.config_reader.template_name.strip())
            or "<" in self.config_reader.template_name
        ):
            return None

        qqt, _ = CollectionOfQueryTemplates.objects.get_or_create(
            name=self.config_reader.template_name,
            organisation=organisation,
        )
        print(
            "Created collection of query template", self.config_reader.template_name
        )

        qtg, _ = QueryTemplateGrammar.objects.get_or_create(
            template_collection=qqt,
            tokens=self.config_reader.templates_grammar["tokens"],
            alphabet=self.config_reader.templates_grammar["alphabet"],
        )
        print("Created grammar for query template", self.config_reader.template_name)

        print("Setting all templates with grammar as not active...")
        QueryTemplate.objects.filter(template_grammar=qtg).update(is_active=False)

        print("Adding templates to database")
        for q_template in self.config_reader.query_templates:
            print("Adding template:", q_template)
            QueryTemplate.objects.get_or_create(
                name=q_template["name"], template_grammar=qtg, is_active=False
            )

            system_prompt = self.__read_file_if_exists(
                file_path=q_template.get("prompt_file", None)
            )

            QueryTemplate.objects.filter(
                name=q_template["name"],
                template_grammar=qtg,
            ).update(
                display=q_template["display"],
                data_connector=q_template["data_connector"],
                data_filter_expressions=q_template["data_filter_expressions"],
                structured_response_if_exists=q_template[
                    "structured_response_if_exists"
                ],
                structured_response_data_fields=q_template[
                    "structured_response_data_fields"
                ],
                is_active=True,
                system_prompt=system_prompt,
            )

        return qqt

    @staticmethod
    def __read_file_if_exists(file_path: str or None):
        if file_path is None or not len(file_path):
            return None

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return None

        with open(file_path, "r") as f:
            file_content = f.read()
        return file_content


class QueryTemplateController:
    def __init__(self):
        self._template_grammar = QueryTemplatesSearchGrammar()
        self._template_filterer = QueryTemplateFilterer()

    def prepare_templates_for_user(
        self,
        organisation_user: OrganisationUser,
        templates: list,
        return_only_data_connector: bool = False,
    ) -> list[QueryTemplate | dict]:
        if templates is None:
            return []

        if type(templates) not in [list]:
            templates = [templates]

        if not len(templates):
            return []

        all_templates = []
        for template_id in templates:
            template = self.get_template_by_id(
                user=organisation_user,
                template_id=template_id,
            )
            if template is not None:
                if return_only_data_connector:
                    all_templates.append(template.data_connector)
                else:
                    all_templates.append(template)
        return all_templates

    def filter_documents(
        self,
        documents: list[Document] | QuerySet[Document],
        query_templates: list[QueryTemplate],
    ) -> list[Document]:
        f_docs = []
        for d in documents:
            if self._template_grammar.use_document_in_sse(
                document=d, skip_if_any_problem=True
            ):
                f_docs.append(d)

        # print("1. filter_documents")
        # print("len(f_docs)", len(f_docs))
        if not len(f_docs):
            return []

        # print("2. filter_documents")
        n_docs = []
        for d in f_docs:
            accept_doc = True
            for qt in query_templates:
                if not self._template_filterer.use_document_in_sse(
                    query_template=qt, document=d
                ):
                    accept_doc = False
                    break
            if accept_doc:
                n_docs.append(d)
        return n_docs

    @staticmethod
    def get_template_by_id(user: OrganisationUser, template_id):
        try:
            query_templ = QueryTemplate.objects.get(id=template_id)
            if (
                query_templ.template_grammar.template_collection.organisation
                != user.organisation
            ):
                return None
            return query_templ
        except QueryTemplate.DoesNotExist:
            return None
        except ValueError:
            return None
