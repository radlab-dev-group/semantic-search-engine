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

Po wyżej przedstawionej procedurze uruchamiane jest wyszukiwanie:

```python
query_results = self.search(
    search_text=question_str,
    max_results=search_params.get("max_results", 50),
    rerank_results=search_params.get("rerank_results", False),
    language=lang_str,
    return_with_factored_fields=search_params.get(
        "return_with_factored_fields", False
    ),
    search_in_documents=docs_to_search,
    relative_paths=relative_paths,
)[0]
```

W tej metodzie do przeszukiwania semantycznego wykorzystywany jest
handler do Milvusa. Pamiętajmy, że tutaj przeszukujemy bazę semantyczną
czyli określamy podobieństwo między dwoma _reprezentacjami embeddingowymi_
tekstów -- pytania i fragmentu dokumentu. Dla embeddingu pytania
odszukiwane są najbardziej podobne embeddingi z fragmentami tekstów.

Tworzonyt jest słownik opcji do filtrowania
w Milvusie (po metadanych). Tworzony jest słownik `metadata_filter`
a do niego wpisywane są trzy wartości:

1. `language` -- jeżeli podano, to tylko _embeddingi_ ze wskazanym językiem będą przeszukiwane;
2. `filenames` -- lista nazw dokumentów -- tylko embeddinigi z tymi nazwami będą przeszukiwane;
3. `relative_paths` -- lista ścieżek relatywnych -- podobnie jak `filenames` jednak ściezki relatywne;

Jeżeli podano więcej niż jeden z ww wartości, to w odróżnieniu od bazy relacyjnej,
w Milvusie wybierane są do przeszukiwania te embeddingi,
które spełniają warunek połączomy `AND`, nie `OR 
