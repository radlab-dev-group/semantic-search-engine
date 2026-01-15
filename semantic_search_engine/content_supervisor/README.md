# content_supervisor package

*Part of the **semantic‑search‑engine** project*

---

## Purpose

`content_supervisor` is a tiny, pluggable pipeline that extracts useful information from raw text.  
It is built around two concepts:

| Concept       | Role                                                                                                                                 |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **Executor**  | Performs a side‑effect operation (e.g., download a web page) and returns a mapping of inputs → results.                              |
| **Processor** | Scans a text fragment, extracts data (usually with regular expressions), and optionally forwards the extracted items to an executor. |

The package ships with a concrete **URL‑extractor** that can also fetch the body of each discovered URL.

---

## Package layout

```
content_supervisor/
│
├─ executors/
│   ├─ __init__.py
│   ├─ executor.py          # abstract base class
│   └─ www_content_grabber.py  # concrete executor that downloads and parses HTML
│
├─ processors/
│   ├─ __init__.py
│   ├─ processors.py        # abstract base class
│   └─ regex_processors.py  # regex‑based processors, including URL extractor
│
├─ __init__.py
└─ supervisor.py           # orchestrator that wires the processors together
```

---

## Core abstractions

### `ExecutorI`  (executors/executor.py)

*Abstract base class for any executor.*

| Member                                       | Description                                                                                                   |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `__init__(self, name_str: str, options: Dict | None = None)`                                                                                                 | Stores a human‑readable name and optional configuration dictionary. |
| `name` (property)                            | Returns the executor name.                                                                                    |
| `options` (property)                         | Returns the options dictionary (empty if not supplied).                                                       |
| `apply(self, body) -> Dict[Any, Any]`        | **Abstract** – concrete subclasses must implement the processing logic and return a mapping `input → result`. |

---

### `WWWContentGrabberExecutor`  (executors/www_content_grabber.py)

*Concrete executor that downloads a list of URLs and extracts plain text from the `<p>` tags.*

* **Constructor** – `WWWContentGrabberExecutor(options: Dict | None = None)` forwards the name
  `"ContentGrabberExecutor"` to the base class.
* **`apply(self, body)`** – Accepts either a single URL string or a list of URLs, normalises to a list, then for each
  URL:
    1. **Sanitises** the URL (`__clear__url`).
    2. **Downloads** the raw bytes using `urllib.request` with a Mozilla‑style `User‑Agent`.
    3. **Parses** the HTML with **BeautifulSoup** (`lxml` parser) and concatenates the text of all `<p>` elements.
    4. Returns a dictionary `{url: extracted_text}`.

*Private helpers*

| Method                                          | Purpose                                                                                       |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `__download_url_content(self, url: str) -> str` | Performs the HTTP request and returns the raw response body (or empty string on error).       |
| `__clear__url(url: str) -> str`                 | Strips trailing punctuation and whitespace.                                                   |
| `__parse_url_content(url_content: str) -> str`  | Extracts the textual content of paragraph tags; returns an empty string if the page is empty. |

---

### `ProcessorI`  (processors/processors.py)

*Abstract base class for a text processor.*

| Member                                     | Description                                                                                  |
|--------------------------------------------|----------------------------------------------------------------------------------------------|
| `__init__(self, name: str, options: Dict)` | Stores the processor name, its options, and a placeholder for an executor (`self.executor`). |
| `use_executor` (property)                  | Returns `True` when the options contain `"use_executor": True`.                              |
| `name` (property)                          | Human‑readable processor name.                                                               |
| `options` (property)                       | Options dictionary supplied at construction.                                                 |
| `process(self, text: str) -> List          | Dict[Any, Any]`                                                                              | **Abstract** – concrete subclasses must implement the extraction logic. |
| `__str__(self)`                            | Returns the processor name (so a processor can be printed directly).                         |

---

### `RegexProcessor`  (processors/regex_processors.py)

*Processor that applies one or more regular‑expression patterns to a text string.*

* **Constructor** – `RegexProcessor(pattern: str | List[str], name: str, options: Dict)`
    * Normalises `pattern` to a list of strings.
    * Stores the list as `self._patterns`.

* **`process(self, text: str) -> List | Dict[Any, Any]`**
    1. Iterates over each pattern and collects **all** matches (`re.findall`).
    2. Flattens the result (expects each match to be a tuple and keeps the first element).
    3. If `use_executor` is `True` **and** matches are found, forwards the list to `self.executor.apply` and returns the
       executor’s mapping.
    4. Otherwise returns the flat list of matches.

*The class does **not** create an executor itself; it must be attached manually (or by a subclass).*

---

### `URLRegexProcessor`  (processors/regex_processors.py)

*Specialised `RegexProcessor` that extracts URLs and optionally downloads their content.*

* **Class constants**
    * `SELF_PROCESSOR_NAME = "RegexProcessor-URL"` – the name exposed to the outside.
    * `SELF_URL_PATTERNS = [r"((https|http)?://[^\s]+)"]` – a simple URL‑matching regex.

* **Constructor** – `URLRegexProcessor(options: Dict)`
    * Calls `super().__init__(pattern=SELF_URL_PATTERNS, name=SELF_PROCESSOR_NAME, options=options)`.
    * If `options.get("use_executor", False)` is `True`, instantiates a `WWWContentGrabberExecutor` and stores it in
      `self.executor`.

* **Behaviour** – Inherits `process` from `RegexProcessor`.
    * When `use_executor` is active, each discovered URL is fetched and its paragraph text is returned as a mapping
      `{url: text}`.
    * When the flag is off, a plain list of URL strings is returned.

---

### `ContentSupervisorOutput`  (supervisor.py)

*Simple data holder representing a single piece of extracted content.*

| Property       | Meaning                                                                                       |
|----------------|-----------------------------------------------------------------------------------------------|
| `content_type` | The processor name that produced this output (e.g., `"RegexProcessor-URL"`).                  |
| `content_body` | The raw result returned by the processor (list of strings **or** mapping of URL → page text). |

Both attributes are read‑only via `@property` descriptors.

---

### `ContentSupervisor`  (supervisor.py)

*Orchestrator that builds a processing chain and runs it against a text block.*

| Method                                                             | Description                                                                                                                                                   |
|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `__init__(self, options: Dict)`                                    | Stores the configuration dictionary and immediately builds the processor chain (`self.__prepare_processors_chain`).                                           |
| `check_text(self, text_str: str) -> List[ContentSupervisorOutput]` | Executes every processor in the chain on `text_str`. For each non‑empty result, creates a `ContentSupervisorOutput` instance and aggregates them into a list. |
| `__prepare_processors_chain(self)`                                 | Currently creates a **single‑element** chain: `[URLRegexProcessor(self._options)]`. The method can be extended to add more processors later.                  |

**Typical usage**

```python
from content_supervisor.supervisor import ContentSupervisor

options = {
    "use_executor": True,  # instruct URL processor to fetch page bodies
    # … other future options …
}
supervisor = ContentSupervisor(options)

text = "Check out https://example.com and http://test.org for more info."
results = supervisor.check_text(text)

for out in results:
    print(f"Processor: {out.content_type}")
    print(out.content_body)  # either list of URLs or {url: page_text}
```

The `check_text` call returns a list of `ContentSupervisorOutput` objects, each representing the output of a distinct
processor in the chain.

---

## Summary of the data flow

1. **Input** – a raw string (`text_str`).
2. **Processor chain** – each processor receives the same original text.
    * `URLRegexProcessor` extracts URLs.
    * If `use_executor` is `True`, the processor hands the URL list to `WWWContentGrabberExecutor`.
3. **Executor** – downloads each URL, parses the HTML, extracts paragraph text, and returns a dict.
4. **Supervisor** – wraps every non‑empty processor result into a `ContentSupervisorOutput` and returns the collection
   to the caller.

The design is deliberately lightweight: adding a new processor (e.g., a keyword extractor) only requires subclassing
`ProcessorI` and inserting the instance into `ContentSupervisor.__prepare_processors_chain`. No other part of the system
needs to change.