## Moduł`engine.controllers.search.DBSemanticSearchController`

Logika wyszukiwania jest w ogólnej mierze nastepująca:

1. W pierwszym kroku, na podstawie meta-informacji przygotowywane są dokumenty,
   w których przeszukiwana będzie fraza. Ten krok ogranicza przestrzeń w bazie semantycznej.
   Tylko przefiltrowane w tym kroku dokumenty będą uwzględniane podczas przeszukiwania
   semantycznego. W tym kroku bardzo dużo błędnych dokumentów odrzucamy na starcie,
   zanim zaczniemy przeszukiwanie semantyczne.
2. Do przeszukiwania semantycznego wykorzystywany jest oczywiście aspekt podobieństwa
   semantycznego, jednak zamiast przeszukiwać całą bazę semantyczną,
   przeszukiwane są tylko dokumenty, które wyszły z _Kroku 1_.

### Wyszukiwanie z opcjami, metoda `search_with_options`

```python
def search_with_options(
        self,
        question_str: str,
        search_params: dict,
        convert_to_pd: bool = False,
        reformat_to_display: bool = False,
        ignore_question_lang_detect: bool = False,
        organisation_user: OrganisationUser = None,
        collection: CollectionOfDocuments = None,
        user_query: UserQuery = None,
):
    ...
```

Ważnym elementem jest`search_params`, to słownik, który służy do filtrowania dokumentów przed
przeszukiwaniem wyszukiwarki semantycznej. Słownik ten posiada pola:

* `categories` -- jako lista kategorii (stringów), dokument musi posiadać co
  najmniej jedną z tych kategorii, dokument ma przypisaną jedną kategorię główną
  (`document.category`) i to ona jest sprawdzana z tą listą.
* `documents` -- lista nazw dokumentów, które mają być brane pod uwagę podczas
  przeszukiwania, tylko te wskazane dokumenty będą przeszukiwane
* `relative_path` -- podobnie jak dokumenty, ale wskazane ściezki relatywne.
  W dokumentach kilka może mieć taką samą nazwę, ale tylko jeden posiada
  daną ściezkę relatywną.
* `relative_path_contains` -- Lista fraz, którymi przefiltrować tylko takie dokumenty,
  które zawierają co najmniej jedną ze zdefiniowanych fraz. Może się przydać, kiedy np.
  sciezka relatywna, to _url_ strony. Wtedy tą listą można przefiltrować strony po domenie,
  a właściwie po liście domen -- można tym sposobem wybrać tylko teksty z określonych urli.
* `templates` -- lista identyfikatorów templatek, którymi maja być filtrowane dokumenty
* `metadata_filter` -- dowolny filtr oparty o przeszukiwanie metadanych
  (pole `metadata_json` w `Document`)
* `only_template_documents` -- znacznik _boolowski_, który domyślnie jest wyłaczony.
  Włączenie tego przełącznika powoduje, że tylko dokumenty spełniający założenia templatek
  będą wykorzystywane do wuszkiwania. Jeżeli ustawiona na _False_ wtedy również inne
  mechanizmy filtrujące będą ogrniczały wyniki.
* `hybrid_search` -- znacznik _boolowski_ (domyślnie _True_). Włącza wyszukiwanie hybrydowe,
  łączące wyniki z Milvusa (wektorowe) z wynikami z PostgreSQL (Full-Text Search).
* `max_results` -- maksymalna liczba wyników do zwrócenia (domyślnie 50).
* `min_similarity` -- próg odcięcia podobieństwa cosine (domyślnie `0.5`).
  Wyniki z podobieństwem poniżej tego progu są odrzucane, dlatego zapytanie
  może zwrócić pusty zbiór wyników.

W polu `stats` (statystyki na poziomie dokumentu) każdy wynik z chunka jest
najpierw normalizowany do przedziału `[0, 1]` -- w zależności od etapu
wyszukiwania jest to podobieństwo cosine, wartość RRF lub logit
cross-encodera -- a dopiero potem liczone są wagi dokumentów. Oprócz surowej
średniej (`score`) dokumentacja endpointu opisuje pola
`score_normalized`, `score_weighted`, `score_weighted_scaled` (udział w
całkowitej masie, suma = `1.0`) oraz `weight_normalized` (waga względem
najlepszego dokumentu).

Warto pamiętać, że `score` przy wynikach (chunkach) to **ocena, która
ustaliła kolejność**, a jej sens zależy od trybu -- odpowiedź zawiera pole
`score_scale`:
* `cosine` -- zwykłe wyszukiwanie wektorowe (`hybrid_search` wyłączone):
  `score` to cosine,
* `rrf` -- wyszukiwanie hybrydowe (`hybrid_search` włączone, **domyślnie**):
  `score` to wartość RRF, więc dobrze dobrany fragment może mieć np.
  `0.03` -- to świetna pozycja w obu rankingach, a nie małe podobieństwo,
* `reranker` -- zastosowany cross-encoder: `score` to logit.

Do porównywania wyników z różnych trybów służy `score_normalized`
(znormalizowane do `[0, 1]`), a `similarity` (cosine) pojawia się tylko
wtedy, gdy ranking opiera się wyłącznie na wyszukiwaniu wektorowym.

**Flow** filtrowania meta-informacjami jest nastepujący:

1. Jeżeli przekazane były kategorie, to z bazy wybierane są te dokumenty,
   które posiadają jedną z przekazanych kategorii. Pobierane są nazwy tych dokumentów.
   Dokumenty te odkładane są jako **pierwsza kupka dokumentów**.
2. Jeżeli podane zostały identyfikatory templatek, to niezależnie od punktu 1.
   tworzona jest **druga kupka dokumentów**. Wybierane są tylko takie dokumenty,
   które spełniają założenia przekazanych szablonów.
3. Tworzona jest **kupka dokumentów** `documents` (ze słownika `search_params`)
4. Tworzona jest **kupka relatywnych ścieżek** ze słownika `search_params`
6. Tworzona jest **kupka dokumentów** na podstawie `metadata_filters`
7. Jeżeli ustawiona jest flaga `only_template_documents` to wszystkie
   kupki są usuwane, poza kupką z dokumentów, które dopasowane zostały
   na podstawie templatek. Wszysatko inne jest ignorowane -- nie ma
   innych ograniczeń na dokumenty.
8. Jeżeli ta flaga jest ustawiona na _False_ kupki z nazwami dokumentów
   łączone są do jedenk kupki z _nazwami dokumentów_. Braną są wszyskie nazwy dokumentów.
   Jeżeli dokument wystepouje tylko na jednej z kupek również jest brany pod uwagę.
   Nie jest wymagane aby dokument spełnił wszystkie filtry, musi spełnic co najmniej
   jeden i wtedy przechodzi do wyszukiwarki semantycznej.

Przkładowe zapisy `metadata_filters`:

```json
[
  {
    "operator": "in",
    "field": {
      "deep_labels": {
        "0": [
          "kategoria"
        ]
      }
    }
  },
  {
    "operator": "eq",
    "field": {
      "main_category": "kategoria"
    }
  },
  {
    "operator": "lt",
    "field": {
      "kategoria": {
        "gdzie_wartosc": {
          "jest_bardzo_gleboko": 100
        }
      }
    }
  }
]
```

Dostępne operatory: `["in", "eq", "ne", "gt", "lt", "gte", "lte", "hse"]`.
Gdzie `hse` (_has same element_) to operacja, która służy do sprawdzania czy dwa
zbiory posiadają taki sam element. Np. posiadając dwie listy można sprawdzić
czy zawierają jakiś wspólny element. Przykłady działania `hse`
na dwóch listach `list_a` oraz `list_b`.

```
list_a = [1, 2] list_b = [2, 5, 6, 7] --> return True
list_a = [], list_b = [2, 5, 6, 7] --> return False
list_a = [2] list_b = [2, 5, 6, 7] --> return True
list_a = [1, 2, 3, 5] list_b = [6, 7] --> return False
list_a = [1, 2, 3, 5] list_b = [5] --> return True
```

Podczas porównywania pojedycznego operatora z metadanymi, możliwe jest dowolne
zagłębienie słowników.

Poniżej znajduje się przykładaowa definicja metadanych `md_dict` oraz kilku
wyrażeń filtrujących z różnymi opratorami.

```python
# %%
md_dict = {
    "deep_labels": {
        "0": [],
        "1": [
            "Konflikty i kryzysy"
        ],
        "2": [
            "Informacje o Ukrainie w czasie kryzysu."
        ],
        "3": [
            "Napady Naturalne i Konflikty Militarne"
        ],
        "4": [
            "Konflikty zbrojne na Ukrainie."
        ],
        "5": {
            "6": {
                "7": 125
            }
        }
    },
    "channel": "TCH_channel",
    "dataset_id": "Telegram UA",
    "main_label": "Konflikty zbrojne na Ukrainie.",
    "clear_texts": False,
}

# Filtering expressions with operators
# 1
e_dict_1 = {
    "deep_labels": {
        "1": "Konflikty i kryzysy"
    }
}
e_op_1 = "in"
# 2
e_dict_2 = {
    "main_label": "Konflikty zbrojne na Ukrainie."
}
e_op_2 = "gt"
# 3
e_dict_3 = {
    "deep_labels": {
        "5": {
            "6": {
                "7": 125
            }
        }
    }
}
e_op_3 = "eq"
```

Wynik działania/pokrycia eyrażeń na metadanych:

``` 
Konflikty i kryzysy in ['Konflikty i kryzysy'] -> True
Konflikty zbrojne na Ukrainie. gt Konflikty zbrojne na Ukrainie. -> False
125 eq 125 -> True
```

Po wyżej przedstawionej procedurze uruchamiane jest wyszukiwanie hybrydowe (lub tylko semantyczne, zależnie od flagi `hybrid_search`).

Wyszukiwanie hybrydowe łączy dwa podejścia:
1. **Wyszukiwanie semantyczne (Milvus)**: Znajduje fragmenty o podobnym znaczeniu wektorowym.
2. **Wyszukiwanie tekstowe (PostgreSQL FTS)**: Znajduje fragmenty zawierające dokładnie te same słowa lub ich odmiany (wykorzystując wagi SearchVector i SearchRank).

Wyniki z obu źródeł są łączone przy użyciu algorytmu **Reciprocal Rank Fusion (RRF)**, który pozwala na sprawiedliwe uszeregowanie wyników pochodzących z różnych systemów punktacji.

Główna metoda wywołująca wyszukiwanie w `DBSemanticSearchController` wygląda obecnie tak:

```python
if hybrid_search:
    # Pobranie wyników wektorowych z Milvus
    milvus_results = self.search(...)
    
    # Pobranie wyników tekstowych z Postgres FTS
    postgres_results = textual_controller.search_texts(...)
    
    # Połączenie wyników algorytmem RRF
    texts_ids, texts_scores = self._rrf_merge(
        milvus_results=milvus_results,
        postgres_results=postgres_results,
        max_results=max_results
    )
else:
    # Tylko wyszukiwanie semantyczne
    milvus_results = self.search(...)
    # ... ekstrakcja ID i score ...
```

W metodzie `search` do przeszukiwania semantycznego wykorzystywany jest handler do Milvusa. Pamiętajmy, że tutaj przeszukujemy bazę semantyczną
czyli określamy podobieństwo między dwoma _reprezentacjami embeddingowymi_
tekstów -- pytania i fragmentu dokumentu. Dla embeddingu pytania
odszukiwane są najbardziej podobne embeddingi z fragmentami tekstów.

Tworzony jest słownik opcji do filtrowania
w Milvusie (po metadanych). Tworzony jest słownik `metadata_filter`
a do niego wpisywane są trzy wartości:

1. `language` -- jeżeli podano, to tylko _embeddingi_ ze wskazanym językiem będą przeszukiwane;
2. `filenames` -- lista nazw dokumentów -- tylko embeddinigi z tymi nazwami będą przeszukiwane;
3. `relative_paths` -- lista ścieżek relatywnych -- podobnie jak `filenames` jednak ściezki relatywne;

Jeżeli podano więcej niż jeden z ww wartości, to w odróżnieniu od bazy relacyjnej,
w Milvusie wybierane są do przeszukiwania te embeddingi,
które spełniają warunek połączony `AND`, nie `OR`.

## Fragmentowanie (chunking) kolekcji

Sposób dzielenia tekstów na chunki jest **ustawieniem kolekcji**, deklarowanym
raz -- przy jej tworzeniu -- a nie przy każdym wywołaniu indeksowania.
Wartości trzymane są w `CollectionOfDocuments.metadata_json`:

```json
{
  "chunking": {"max_tokens_in_chunk": 500, "number_of_overlap_tokens": 64},
  "strict_chunking": false
}
```

Jak działa `resolve_chunking` (`data/controllers/chunking.py`) dla jednego
wywołania indeksowania:

1. Wartości z zapytania wygrywają (kolekcja nie jest nadpisywana).
2. Brak wartości w zapytaniu -- używane są ustawienia kolekcji.
3. Gdy ani zapytanie, ani kolekcja nic nie deklarują -- używane są wartości
   domyślne (500 tokenów, brak nakładki) **i są zapamiętywane na kolekcji**,
   więc kolejne wywołanie nie musi nic podawać.
4. Kolekcja ze `strict_chunking: true` odrzuca wartości inne niż własne --
   `CHUNKING_MISMATCH` (`000006_DATA`).
5. Chunk dłuższy niż wejście embedera kolekcji jest odrzucany zamiast być po
   cichu ucinanym przez model -- `CHUNKING_TOO_LONG` (`000005_DATA`).
   `number_of_overlap_tokens` zawsze musi mieścić się w `[0, max_tokens_in_chunk)`.

Różne długości chunków w jednej kolekcji są **dozwolone** (wektory pozostają
porównywalne, rozmywają się jedynie progi i okna kontekstu) -- dlatego
nadpisanie ustawień kolekcji w ogóle istnieje. Twardym inwariantem kolekcji
jest jej embedder, wymiar wektora i metryka odległości, a nie jedna długość
chunka.

### Endpointy

| Endpoint | Zmiana |
|----------|--------|
| `POST new_collection` | opcjonalne `chunking` (`max_tokens_in_chunk`, `number_of_overlap_tokens`) oraz `strict_chunking`; odpowiedź zwraca oba pola (`collection_id`, `chunking`, `strict_chunking`) |
| `POST upload_and_index_files`, `POST add_and_index_texts` | `max_tokens_in_chunk` i `number_of_overlap_tokens` w `indexing_options` są opcjonalne -- bez nich obowiązują ustawienia kolekcji (pozostałych pięć flag nadal jest wymaganych) |
| `GET collections` | każdy obiekt zawiera `metadata_json`, czyli `chunking` i `strict_chunking` |
| `GET filter_options` | `chunking_defaults` -- wartości do wstępnie wypełnionego formularza |

### `max_model_length` modeli

`configs/embedders.json` i `configs/rerankers.json` mogą deklarować
`max_model_length`, czyli maksymalną liczbę tokenów przyjmowaną przez model:
w przypadku embedera jest to **jeden tekst**, w przypadku rerankera **cała para
`query + text`**.

* chunk dłuższy niż wejście embedera byłby ucinany bez ostrzeżenia, więc takie
  fragmentowanie jest odrzucane (`CHUNKING_TOO_LONG`);
* chunk, który nie mieści się w budżecie pary rerankera, jest tylko logowany
  jako ostrzeżenie -- zapytanie też zajmuje część tego budżetu;
* wpis bez `max_model_length` nie ma limitu, więc konfiguracje sprzed dodania
  tego klucza działają bez zmian.

Wartości są dostępne w `GET embedders` i `GET rerankers` jako `models_info`:

```json
{
  "models": ["google/embeddinggemma-300m"],
  "models_info": [
    {
      "name": "google/embeddinggemma-300m",
      "vector_size": 768,
      "max_model_length": 2048,
      "device": "cpu"
    }
  ]
}
```

### Audyt i naprawa ustawień

```shell script
python -m sse_api.manage check_collection_chunking
python -m sse_api.manage check_collection_chunking --collection my_docs --verify
python -m sse_api.manage check_collection_chunking --fix --dry-run
```

`--verify` mierzy realne chunki tokenizerem embedera kolekcji i wskazuje te,
które są dłuższe niż deklaracja. `--fix` poprawia **deklaracje** (brak
chunkingu -> wartości domyślne, chunk ponad wejściem embedera -> przycięcie do
limitu, niemożliwa nakładka -> wartość domyślna), a `--dry-run` pokazuje ten
plan bez zapisu. Żaden tekst nie jest dzielony ponownie -- istniejące chunki
można poprawić wyłącznie przez ponowne zaindeksowanie.

### Klient `sse_lib`

```python
client.create_collection(
    "my_docs", "My Documents", "Opis",
    embedder="google/embeddinggemma-300m",
    reranker="radlab/polish-cross-encoder",
    chunking={"max_tokens_in_chunk": 500, "number_of_overlap_tokens": 0},
    strict_chunking=False,
)
# chunkingu nie trzeba już podawać przy indeksowaniu
client.index_documents("my_docs", ["Tekst do zaindeksowania."])
client.embedders_info()   # vector_size, device, max_model_length
```

`IndexingOptions` wysyła klucze fragmentowania tylko wtedy, gdy zostały ustawione,
a `Collection.chunking` / `Collection.strict_chunking` odczytują ustawienia
kolekcji z `metadata_json`.

### Liczniki dokumentów kolekcji

`GET collections` zwraca przy każdej kolekcji dwie liczby, dzięki czemu klient
nie musi wołać `GET documents` raz na kolekcję:

| Pole | Co liczy |
------|----------|
| `document_count` | Wszystkie dokumenty kolekcji, również wyłączone z wyszukiwania |
| `active_document_count` | Dokumenty bierzące udział w wyszukiwaniu (`use_in_search=True`) |

Oba liczniki powstają **w bazie** -- jedno zapytanie na całą odpowiedź
(`RelationalDBController.with_document_counts`) -- i dotyczą kolekcji widocznych
dla użytkownika. Różnica `document_count - active_document_count` to dokumenty
wyłączone, a `active_document_count` to dokładnie tyle, ile domyślnie
przelicza `GET documents`.

---

## Kolejka indeksacji (Celery + RabbitMQ)

Indeksację można uruchomić w wątku żądania (tak jak dotąd) albo w kolejce
**Celery** z brokerem **RabbitMQ**. Decyduje o tym `ENV_USE_CELERY=1` (lub
`"use_celery": true` w `django-config.json`). Niezależnie od trybu wykonywany
jest ten sam pipeline, a postęp wygląda identycznie.

Jednostką pracy jest **dokument**: zadanie `prepare_indexing_job` zapisuje
dokumenty zgłoszenia i wrzuca do kolejki po jednym `index_document`.
Uszkodzony dokument jest ponawiany trzy razy, a potem oznaczany jako `failed`
bez przerywania reszty zgłoszenia; oba zadania potwierdzają wiadomości po
wykonaniu (`acks_late`), więc zabity worker niczego nie gubi.

Postęp mieszka w bazie relacyjnej (nie w result backendzie Celery), dlatego
przeżywa restart workera i jest widoczny dla wielu workerów naraz:

| Pole | Znaczenie |
|------|-----------|
| `IndexingJob.status` | `queued` → `preparing` → `indexing` → `completed` / `partial` / `failed` |
| `IndexingJob.total_documents`, `documents_indexed`, `documents_failed`, `documents_skipped` | liczniki zgłoszenia (aktualizowane przez `F()`) |
| `IndexingJob.current_document` | dokument aktualnie indeksowany |
| `Document.indexing_status` | `pending` / `indexing` / `indexed` / `failed` / `skipped`; `NULL` dla dokumentów zaindeksowanych wcześniej |

Czyta go `GET api/indexing_status` -- po `job_id` albo po `collection_name`
(najnowsze zadanie kolekcji). Odpowiedź zawiera zadanie, jego dokumenty ze
statusami oraz liczbę dokumentów kolekcji wciąż oczekujących na indeksację.
Endpointy zgłaszające pliki zwracają to samo zadanie w polu `indexing_job`,
a dodatkowo `"async": true`, gdy praca tylko trafiła do kolejki.

```shell
curl -H "Authorization: Token $TOKEN" \
  "$HOST/api/indexing_status?collection_name=MyCollection"
```

```json
{"status": true, "body": {
  "job": {"id": 12, "status": "indexing", "total_documents": 30,
           "documents_indexed": 12, "documents_failed": 1,
           "documents_pending": 17, "percent": 43,
           "current_document": "regulation.pdf", "finished": false},
  "documents": [{"id": 41, "name": "regulation.pdf", "indexing_status": "indexed"}],
  "pending_documents": 17}}
```

**Uruchomienie** (docker compose chowa kolejkę za profilem, więc stack bez
`ENV_USE_CELERY=1` pozostaje nietknięty):

```bash
# .env: ENV_USE_CELERY=1 — broker, użytkownik i hasło mają wartości lokalne
docker compose --profile indexing up -d     # api + worker + rabbitmq

# ... albo lokalnie przez run.sh, bez żadnych dodatkowych konfigów:
./run.sh infra    # PostgreSQL + Milvus + RabbitMQ (127.0.0.1:5672, UI 15672)
./run.sh worker   # worker indeksacji w foreground
./run.sh start    # API (gunicorn); przy ENV_USE_CELERY=1 worker startuje razem z nim
./run.sh status   # kontenery, worker, porty 5471/19530/19121/5672/8271
./run.sh down     # zatrzymuje WSZYSTKIE procesy SSE (worker, serwer API) i usuwa kontenery;
                    # wolumeny z danymi zostają
./run.sh clean    # kontenery + wolumeny (postgres_data, milvus_data, rabbitmq_data) — kasuje dane

# ... albo worker odpalony ręcznie, po zainstalowaniu zależności API:
celery -A sse_api.config worker -Q indexing -l INFO
```

**Basen workera to `solo`** (`CELERY_WORKER_POOL`, wartość domyślna `run.sh` i
`docker compose`).  Odczyt wgranych plików działa w własnym basenie
`multiprocessing.Pool`, a zadania basenu `prefork` są procesami-demona, którym
Python zabrania tworzenia własnych dzieci: taki worker odpowiada
`AssertionError: daemonic processes are not allowed to have children` zamiast
cokolwiek zaindeksować.  Zadanie, które mimo to trafi do procesu-demona,
przeczyta swoje pliki w jednym procesie (z ostrzeżeniem) zamiast paść.

Embedder kolekcji jest ładowany raz na proces workera, więc `CELERY_CONCURRENCY`
to pytanie o pamięć; zostaw go na 1, o ile każdy proces nie uniesie własnego
modelu. Worker zabity na zawsze zostawia zadanie jako uruchomione --
`python -m sse_api.manage reset_stale_indexing_jobs` domyka wszystko, co wciąż
się "wykonuje" po godzinie (`--minutes`, `--dry-run`).


## Pliki wgranych dokumentów

Pliki, z których powstały dokumenty, leżą pod katalogiem uploadu
(`upload_sse/<organizacja>/<user>/<kolekcja>/<wgranie>`), a ich ścieżki to
kolumny `Document.path`.  Wgranie może poprosić o nie z powrotem, skoro ich
wektory są już zapisane -- wystarczy jedna opcja `indexing_options`:

```json
{"clear_text": true, "check_text_lang": true, "...": "...",
 "delete_files_after_indexing": true}
```

| Opcja | Domyślnie | Co robi |
|-------|-----------|---------|
| `delete_files_after_indexing` | brak (pliki zostają) | Plik każdego dokumentu tego wywołania znika z dysku, gdy jego dokument skończy się indeksować |

* Opcja trafia do `IndexingJob.indexing_options` razem z resztą opcji
  wywołania -- stąd dowiaduje się o niej worker kolejki, który o żądaniu nie
  wie nic.
* **`indexed` i `skipped` usuwają plik**, **`failed` go zostawia**: plik
  dokumentu, którego wektorów nie ma w bazie, to jedyna rzecz, na której da
  się jeszcze coś zrobić przy ponowieniu.
* Usunięcie dzieje się per dokument, w `run_index_document`, w obu trybach
  (w żądaniu i w kolejce), a katalog znika dopiero wtedy, gdy nie ma w nim
  już żadnego pliku.
* Problem z dyskiem idzie do loga i przerywa najwyżej usuwanie tego jednego
  pliku -- nigdy całej indeksacji.
* W przeciwieństwie do pięciu przełączników czyszczenia tekstu
  `delete_files_after_indexing` jest w `indexing_options` opcjonalna, więc
  klient, który jej nie zna, działa dalej.

## Usuwanie treści a dysk

`delete_documents` i `delete_collection` usuwają razem z wierszami i wektorami
także pliki z dysku (pliki jako ostatnie, żeby dokument, który wciąż jest w
bazach, nigdy nie stracił swojego pliku).  Odpowiedź mówi, ile plików
poszło, w polu `deleted_files`, a pliku, którego usunąć się nie dało (np.
tylko do odczytu), po prostu nie udaje się usunąć -- jest wypisany w
`file_errors`.  Problem z dyskiem nie powoduje błędu całego usuwania.

* Usuwane są wyłącznie ścieżki z katalogu uploadu -- ścieżki spoza niego
  nigdy nie są dotykane, więc wpis do bazy nie wyłączy niczego obok.
* Katalog znika tylko wtedy, gdy jest już pusty, więc katalog innej kolekcji
  nigdy nie ucierpi.
* `delete_texts` plików nie usuwa: reszta dokumentu wciąż siedzi w bazach.
