# query_templates module

*Part of the **semantic‑search‑engine** package*

---

## Purpose

The `query_templates` package implements a **template‑driven query system** for the semantic search engine.  
A *query template* defines:

* **Which documents** may be considered (via a `data_connector`).
* **How to filter** those documents – JSON‑encoded Python expressions evaluated against each document’s `metadata_json`.
* **What response format** to return (structured fields, system prompt, …).
* **A simple grammar** that maps short tokens (e.g., “today”, “tomorrow”) to concrete templates.

All templates are stored in the Django database (see `models.py`) and can be bulk‑loaded from a JSON configuration
file (`configs/query-templates.json`).

---

## Core Classes

### `QueryTemplatesSearchGrammar`

*Static definition of the grammar and helper functions that operate on a single `Document`.*

| Attribute / Constant         | Meaning                                                                          |
|------------------------------|----------------------------------------------------------------------------------|
| `Actions`                    | `SHOW_WHOLE`, `END_DATE_OLDER_THAN_TODAY` – identifiers for per‑document checks. |
| `Types`                      | `EVENT`, `ADDRESS`, `OTHER` – high‑level document categories.                    |
| `JsonFields`                 | `SEARCH`, `PRESENTATION` – where actions are applied.                            |
| `MetadataFields`             | `document_date`, `document_type` – keys expected inside `metadata_json`.         |
| `ALL_POSSIBLE_TYPE_ACTIONS*` | Lists of actions that are allowed for each context.                              |
| `CONSTANT_TYPE_ACTIONS`      | Mapping **type → {json_field → [actions]}** that drives `use_document_in_sse`.   |

#### Inner class `GrammarFunctions`

*Provides concrete implementations of the actions.*

* `__init__()` – builds a mapping `action → method`.
* `accept_document(action, document)` – validates the action name, ensures it is allowed for the *search* context, and
  delegates to the concrete method.
* `__end_date_is_older_than_today(document)` – extracts `metadata_json['date']['end']`, parses it with
  `dateutil.parser.isoparse`, normalises the time to midnight, and checks that the end date is **≥** today’s midnight.
* `__datetime_from_str(date_str)` – legacy helper (unused in current code).

#### Public method

```python
def use_document_in_sse(self, document: Document, skip_if_any_problem: bool = True) -> bool
```

* Determines whether a document passes the *search‑grammar* checks.
* Looks up the document’s type (`metadata_json['type']`) and retrieves the actions for the `SEARCH` field.
* Executes each action via `GrammarFunctions.accept_document`.
* Returns `True` only if **all** actions succeed; otherwise `False`.

---

### `QueryTemplateFilterer`

*Evaluates the **template‑specific filter expressions** against a document.*

#### Public static method

```python
def use_document_in_sse(query_template: QueryTemplate, document: Document) -> bool
```

* Retrieves `document.metadata_json`. If absent → `False`.
* Obtains `filter_opts = query_template.data_filter_expressions`.
    * If `filter_opts` is empty → **accept every document** (`True`).
* Iterates over each top‑level key (`var1`) in `filter_opts`.
    * If the document does not contain that key → skip the constraint.
    * Handles two shapes of expressions:
        * **Dict** – e.g. `{ "date": { "begin": "...", "end": "..." } }`. Each inner key (`var2`) yields a separate
          expression/value pair.
        * **Scalar** – a single expression for the whole key.
* For every expression/value pair:
    * Replaces the placeholder `VALUE_OF_DATA_EVAL_EXPRESSION` (imported from `constants.py`) with the actual document
      value.
    * Executes `eval(expression_to_eval)`.
    * If any evaluation returns `False` or raises an exception → the document is rejected.
* Returns `True` only when **all** evaluated constraints succeed.

---

### `QueryTemplateConfigReader`

*Loads and parses the JSON configuration file.*

#### Constructor

```python
def __init__(self, config_path: str = "configs/query-templates.json")
```

* Stores `config_path`.
* Calls `self.__load()`.

#### Private method `__load()`

* Opens the JSON file, parses it with `json.load`.
* Exposes three public attributes:
    * `self.template_name` – stripped string identifying the collection.
    * `self.query_templates` – list of template dictionaries (as in the file).
    * `self.templates_grammar` – the grammar definition (`tokens`, `alphabet`).

*Raises* an assertion error if `template_name` is empty.

---

### `QueryTemplatesLoaderController`

*Persists a configuration into the database.*

#### Constructor

```python
def __init__(self, config_path: str = "configs/query-templates.json")
```

* Instantiates a `QueryTemplateConfigReader` for the given path.

#### Public method

```python
def add_templates_to_organisation(self, organisation: Organisation) -> CollectionOfQueryTemplates | None
```

* Validates the `template_name` (non‑empty, no stray “<”).
* Creates (or fetches) a `CollectionOfQueryTemplates` linked to the supplied `organisation`.
* Creates a `QueryTemplateGrammar` with the tokens/alphabet from the config.
* Deactivates any existing templates belonging to this grammar (`is_active=False`).
* For each entry in `self.config_reader.query_templates`:
    * Ensures a `QueryTemplate` row exists (by name + grammar).
    * Reads the optional `prompt_file` (via `__read_file_if_exists`).
    * Updates the row with all fields (`display`, `data_connector`, `data_filter_expressions`, `structured_response_*`,
      `system_prompt`) and sets `is_active=True`.
* Returns the created `CollectionOfQueryTemplates` instance.

#### Private static helper

```python
def __read_file_if_exists(file_path: str | None) -> str | None
```

* Returns the file content if the path exists and points to a regular file; otherwise `None`.

---

### `QueryTemplateController`

*High‑level façade used by application code.*

#### Constructor

```python
def __init__(self)
```

* Instantiates a `QueryTemplatesSearchGrammar` (grammar checks) and a `QueryTemplateFilterer` (template filters).

#### Method `prepare_templates_for_user`

```python
def prepare_templates_for_user(
        organisation_user: OrganisationUser,
        templates: list[int] | int | None,
        return_only_data_connector: bool = False,
) -> list[QueryTemplate] | list[dict]
```

* Normalises `templates` to a list of IDs.
* For each ID, fetches the `QueryTemplate` via `get_template_by_id`.
* If `return_only_data_connector` is `True`, returns only the `data_connector` dicts; otherwise returns the full
  `QueryTemplate` objects.
* Returns an empty list when the input is `None` or empty.

#### Method `filter_documents`

```python
def filter_documents(
        documents: list[Document] | QuerySet[Document],
        query_templates: list[QueryTemplate],
) -> list[Document]
```

1. **Grammar filter** – Calls `self._template_grammar.use_document_in_sse` for each document; only those returning
   `True` are kept (`f_docs`).
2. **Template filters** – For each document in `f_docs`, iterates over the supplied `query_templates` and applies
   `self._template_filterer.use_document_in_sse`.
3. Returns the list of documents that passed **both** stages.

#### Static method `get_template_by_id`

```python
@staticmethod
def get_template_by_id(user: OrganisationUser, template_id) -> QueryTemplate | None
```

* Retrieves the `QueryTemplate` with the given primary key.
* Verifies that the template belongs to the same `organisation` as the `user`.
* Returns `None` if the template does not exist, the ID is invalid, or the organisation check fails.

---

## Typical Workflow

```python
from data.query_templates import (
    QueryTemplatesLoaderController,
    QueryTemplateController,
)
from system.models import Organisation, OrganisationUser
from data.models import Document


# 1️⃣ Load templates into the DB (run once per organisation)
def initialise_templates(org: Organisation):
    loader = QueryTemplatesLoaderController(
        config_path="configs/query-templates.json"
    )
    loader.add_templates_to_organisation(organisation=org)


# 2️⃣ At request time, filter documents according to user‑selected templates
def search(user: OrganisationUser, template_ids: list[int], docs: list[Document]):
    controller = QueryTemplateController()
    templates = controller.prepare_templates_for_user(
        organisation_user=user,
        templates=template_ids,
        return_only_data_connector=False,
    )
    matching = controller.filter_documents(documents=docs, query_templates=templates)
    return matching
```

*`initialise_templates`* populates the DB (creates the collection, grammar, and each template).  
*`search`* fetches the requested templates, runs the grammar check, then applies the per‑template filter expressions,
and finally returns the documents that satisfy all constraints.

---

## File Structure Recap

```
query_templates/
│
├─ constants.py               # VALUE_OF_DATA_EVAL_EXPRESSION, etc.
├─ models.py                  # Django ORM definitions (CollectionOfQueryTemplates, QueryTemplate, …)
├─ query_templates.json       # Example configuration (tokens, templates, grammar)
├─ query_templates.py         # Implementation described above
└─ (optional) resources/…    # Prompt files referenced by templates
```

---

*Feel free to ask for a deeper dive into any specific method or to see usage examples in the Django shell!*