"""
Utility module that defines the grammar, loading, and filtering logic for
*query‑template* objects used throughout the system.

Key responsibilities:
* **QueryTemplatesSearchGrammar** – static description of supported actions,
  types and how they map to JSON fields.
* **QueryTemplateFilterer** – evaluates a document against a template’s
  ``data_filter_expressions``.
* **QueryTemplateConfigReader** – reads a JSON configuration file that
  describes a collection of templates.
* **QueryTemplatesLoaderController** – creates/updates the DB objects
  (collections, grammars, templates) based on the configuration.
* **QueryTemplateController** – high‑level façade that prepares templates for a
  user and filters documents using the grammar and filterer.

All classes are deliberately thin; heavy lifting is delegated to the
individual helper methods.
"""

import os
import json
import datetime
import dateutil

from django.db.models import QuerySet

from system.models import Organisation, OrganisationUser

from data.models import Document
from data.models import (
    CollectionOfQueryTemplates,
    QueryTemplateGrammar,
    QueryTemplate,
)
from data.controllers.constants import VALUE_OF_DATA_EVAL_EXPRESSION


class QueryTemplatesSearchGrammar:
    """
    Static description of the search‑grammar used for query templates.

    The grammar defines:
    * **Actions** – operations that can be performed on a document
      (e.g. ``show_whole`` or ``end_date_older_than_today``).
    * **Types** – high‑level document categories (event, address, other).
    * **JsonFields** – keys used in a template JSON (``search`` and
      ``presentation``).
    * **MetadataFields** – keys expected inside a document’s
      ``metadata_json`` (date and type).

    The class also provides lookup tables that map *type* → *allowed actions*
    per JSON field.
    """

    class Actions:
        """Supported actions for a document."""

        SHOW_WHOLE = "show_whole"
        END_DATE_OLDER_THAN_TODAY = "end_date_older_than_today"

    class Types:
        """High‑level categories a document may belong to."""

        EVENT = "event"
        ADDRESS = "address"
        OTHER = "other"

    class JsonFields:
        """Top‑level keys inside a template definition."""

        SEARCH = "search"
        PRESENTATION = "presentation"

    class MetadataFields:
        """Keys expected inside ``Document.metadata_json``."""

        document_date = "date"
        document_type = "type"

    # ------------------------------------------------------------------
    # Lookup tables (kept as class attributes for fast access)
    # ------------------------------------------------------------------
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
        """
        Helper that knows how to evaluate a concrete ``action``
        on a :class:`~data.models.Document`.
        """

        def __init__(self):
            # Map public action names to private evaluator methods.
            self._actions_mapping = {
                QueryTemplatesSearchGrammar.Actions.END_DATE_OLDER_THAN_TODAY: self.__end_date_is_older_than_today
            }

        def accept_document(self, action: str, document: Document) -> bool:
            """
            Return ``True`` if *document* satisfies the given *action*.

            Parameters
            ----------
            action: str
                One of the values defined in :class:`QueryTemplatesSearchGrammar.Actions`.
            document: Document
                The document to evaluate.

            Raises
            ------
            Exception
                If *action* is not a valid *search*‑type action.
            """
            assert action in self._actions_mapping
            if (
                action
                not in QueryTemplatesSearchGrammar.ALL_POSSIBLE_TYPE_ACTIONS_SEARCH
            ):
                raise Exception(f"Invalid action {action} to accept document!")

            return self._actions_mapping[action](document)

        def __end_date_is_older_than_today(self, d: Document) -> bool:
            """
            Check whether the document’s ``metadata_json.date.end`` is older
            than (or equal to) today’s date.

            Returns
            -------
            bool
                ``True`` if the end‑date is today or later, ``False`` otherwise.
            """
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
            """
            Parse a ``YYYY‑MM‑DD`` string into a :class:`datetime.datetime`.
            """
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    def __init__(self, use_metadata: bool = True):
        """
        Create a grammar helper.

        Parameters
        ----------
        use_metadata: bool, default ``True``
            When ``True`` the helper expects document information to be stored
            inside ``Document.metadata_json`` (type and date fields). The
            current implementation only works with metadata, hence the
            assertion below.
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
        Determine whether *document* should be indexed by the search engine.

        The method looks up the document’s type, fetches the allowed actions for
        that type (only ``SEARCH`` actions are considered), and evaluates each
        action via :class:`GrammarFunctions`.

        Parameters
        ----------
        document: Document
            Document to evaluate.
        skip_if_any_problem: bool, default ``True``
            If ``True`` any exception raised by an action is ignored and the
            document is simply rejected; if ``False`` the original exception is
            propagated.

        Returns
        -------
        bool
            ``True`` if the document passes **all** applicable actions,
            otherwise ``False``.
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
        """
        Placeholder for future implementation that will return either a full
        document or a chunked representation.
        """
        pass

    def __get_document_type(self, document: Document) -> str | None:
        """
        Extract the document type from its metadata.

        Returns ``None`` if the type field is missing. Raises an exception if
        ``use_metadata_when_templating`` is ``False`` because the current
        implementation relies exclusively on metadata.
        """
        if self.use_metadata_when_templating:
            return document.metadata_json.get(
                QueryTemplatesSearchGrammar.MetadataFields.document_type, None
            )
        else:
            raise Exception("Document type may be defined only in metadata!")


class QueryTemplateFilterer:
    """
    Evaluates a :class:`~data.models.Document` against a
    :class:`~data.models.QueryTemplate`'s ``data_filter_expressions``.

    The filter expressions are stored as strings that may reference a special
    placeholder (``VALUE_OF_DATA_EVAL_EXPRESSION``).  Each expression is
    ``eval``‑ed in a safe context; if any expression evaluates to ``False`` the
    document is rejected.
    """

    def __init__(self):
        pass

    @staticmethod
    def use_document_in_sse(
        query_template: QueryTemplate, document: Document
    ) -> bool:
        """
        Return ``True`` if *document* satisfies *query_template* filters.

        The method walks through ``query_template.data_filter_expressions``,
        substitutes the placeholder with the actual document value,
        and evaluates the resulting Python expression.

        Parameters
        ----------
        query_template: QueryTemplate
            Template that defines the filter expressions.
        document: Document
            Document to be checked.

        Returns
        -------
        bool
            ``True`` when all applicable expressions evaluate to ``True``;
            ``False`` otherwise.
        """
        doc_metadata = document.metadata_json
        if doc_metadata is None or not len(doc_metadata):
            return False

        filter_opts = query_template.data_filter_expressions
        # NOTE:
        # When no filtering options into query template is defined
        # then each document should be accepted
        if filter_opts is None or not len(filter_opts):
            return True

        all_constraints_ok = False
        for var1, expr1 in filter_opts.items():
            if doc_metadata.get(var1, None) is None:
                continue

            doc_values = []
            expressions = []
            if type(expr1) in [dict]:
                """
                "data_filter_expressions": {
                    "date": {
                        "begin": "datetime.datetime.strptime('DATA_VALUE', '%Y-%m-%d') <= datetime.datetime.now() + datetime.timedelta(days=8)",
                        "end": "datetime.datetime.strptime('DATA_VALUE', '%Y-%m-%d') >= datetime.datetime.now()"
                    }
                },
                """
                for var2, val2 in expr1.items():
                    if type(val2) in [dict]:
                        raise Exception("Not supported expression with nested dict!")
                    if doc_metadata[var1].get(var2, None) is None:
                        continue
                    expressions.append(val2)
                    doc_values.append(doc_metadata[var1][var2])
            else:
                expressions.append(expr1)
                doc_values.append(doc_metadata[var1])

            if not len(doc_values) or not len(expressions):
                continue

            for expression, doc_value in zip(expressions, doc_values):
                expression_to_eval = expression.replace(
                    VALUE_OF_DATA_EVAL_EXPRESSION, doc_value
                )
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

                all_constraints_ok = True
        return all_constraints_ok


class QueryTemplateConfigReader:
    """
    Read and validate the JSON configuration that defines a collection of
    query‑template objects.

    The configuration file must contain:
    * ``template_name`` – a human readable identifier.
    * ``query_templates`` – a list of template specifications.
    * ``templates_grammar`` – token/alphabet definitions used by the grammar.
    """

    def __init__(self, config_path: str = "configs/query-templates.json"):
        """
        Create a reader and immediately load the configuration.

        Parameters
        ----------
        config_path: str, default ``"configs/query-templates.json"``
            Path to the JSON file that holds the template definition.
        """
        self.config_path = config_path

        self._whole_config = None
        self.template_name = None
        self.query_templates = None
        self.templates_grammar = None

        self.__load()

    def __load(self):
        """
        Load the JSON file and expose its top‑level keys as attributes.
        """
        with open(self.config_path, "r") as f:
            self._whole_config = json.load(f)

        self.template_name = self._whole_config["template_name"].strip()
        self.query_templates = self._whole_config["query_templates"]
        self.templates_grammar = self._whole_config["templates_grammar"]

        assert len(
            self.template_name
        ), "Length of template_name must be greater than 0!"


class QueryTemplatesLoaderController:
    """
    Controller that creates/updates the database objects representing a
    collection of query templates based on a configuration file.

    It works with the following Django models:
        * :class:`CollectionOfQueryTemplates`
        * :class:`QueryTemplateGrammar`
        * :class:`QueryTemplate`
    """

    def __init__(self, config_path: str = "configs/query-templates.json"):
        """
        Instantiate the controller and read the configuration.
        """
        self.config_reader = QueryTemplateConfigReader(config_path)

    def add_templates_to_organisation(self, organisation: Organisation):
        """
        Create the collection (if needed) and attach all templates
        to the given organization.

        Returns the created :class:`CollectionOfQueryTemplates` instance or
        ``None`` if the configuration is invalid.
        """
        print("Adding template collection", self.config_reader.template_name)
        return self.__get_add_collection_of_templates_to_organisation(
            organisation=organisation
        )

    def __get_add_collection_of_templates_to_organisation(
        self, organisation: Organisation
    ) -> CollectionOfQueryTemplates or None:
        """
        Internal helper that performs the actual DB operations.

        Returns ``None`` when the ``template_name`` is empty or malformed.
        """
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
        """
        Return the file contents if *file_path* points to an existing
        readable file; otherwise return ``None``.
        """
        if file_path is None or not len(file_path):
            return None

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return None

        with open(file_path, "r") as f:
            file_content = f.read()
        return file_content


class QueryTemplateController:
    """
    High‑level façade used by the application layer.

    It combines the grammar logic (``QueryTemplatesSearchGrammar``) and the
    filterer (``QueryTemplateFilterer``) to:
    * Prepare a list of templates that a user is allowed to see.
    * Filter a set of documents according to a list of selected templates.
    """

    def __init__(self):
        """
        Instantiate the internal helpers.
        """
        self._template_grammar = QueryTemplatesSearchGrammar()
        self._template_filterer = QueryTemplateFilterer()

    def prepare_templates_for_user(
        self,
        organisation_user: OrganisationUser,
        templates: list,
        return_only_data_connector: bool = False,
    ) -> list[QueryTemplate | dict]:
        """
        Return a list of template objects (or their ``data_connector`` values)
        that belong to *organisation_user*.

        Parameters
        ----------
        organisation_user: OrganisationUser
            The user whose Organization owns the templates.
        templates: list
            Iterable of template IDs or a single ID.
        return_only_data_connector: bool, default ``False``
            When ``True`` only the ``data_connector`` field of each template is
            returned; otherwise the full :class:`QueryTemplate` instance is
            returned.

        Returns
        -------
        list[QueryTemplate | dict]
            Empty list if ``templates`` is ``None``/empty.
        """
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
        """
        Filter *documents* so that only those accepted by **all**
        ``query_templates`` remain.

        The method first applies the grammar’s ``use_document_in_sse`` check
        (e.g., date‑related actions) and then runs the more granular
        ``QueryTemplateFilterer`` logic.

        Returns
        -------
        list[Document]
            Possibly empty list of documents that satisfy every template.
        """
        f_docs = []
        for d in documents:
            if self._template_grammar.use_document_in_sse(
                document=d, skip_if_any_problem=True
            ):
                f_docs.append(d)

        if not len(f_docs):
            return []

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
        """
        Retrieve a :class:`QueryTemplate` by its primary key, ensuring that it
        belongs to the same organisation as *user*.

        Returns ``None`` if the template does not exist, belongs to another
        organisation, or ``template_id`` is malformed.
        """
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
