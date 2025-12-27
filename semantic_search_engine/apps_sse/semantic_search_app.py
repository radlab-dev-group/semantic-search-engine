import json
import math
import os
import django
from radlab_data.text.utils import TextUtils

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from data.controllers import DBSemanticSearchController, DBTextSearchController


def prepare_stats(postgres_docs):
    doc_stats = dict()
    for result in postgres_docs:
        result = result["result"]
        doc_name = result["text"]["document_name"]
        doc_score = result["text"]["score"]
        doc_page_number = result["text"]["page_number"]
        if doc_name not in doc_stats:
            doc_stats[doc_name] = {
                "score": 0.0,
                "hits": 0,
                "pages": [],
                "pages_count": 0,
            }
        doc_stats[doc_name]["hits"] += 1
        doc_stats[doc_name]["score"] += doc_score
        doc_stats[doc_name]["pages"].append(doc_page_number)

    for doc, res in doc_stats.items():
        res["score"] = res["score"] / res["hits"]
        res["pages"] = sorted(set(res["pages"]))
        res["pages_count"] = len(set(res["pages"]))
        res["score_weight"] = math.log(
            float(res["score"] * res["hits"] * res["pages_count"])
        )
    return doc_stats


def filter_stats(doc_stats, remove_under_hits=5, remove_under_pages=3):
    filter_result = dict()
    for doc, res in doc_stats.items():
        if (
            res["hits"] >= remove_under_hits
            and res["pages_count"] >= remove_under_pages
        ):
            filter_result[doc] = res
    return filter_result


def main(argv=None):
    index_name = "HNSW"
    ts = DBTextSearchController()
    se = DBSemanticSearchController(
        collection_name=f"AiA2023_{index_name}", batch_size=10, index_name=index_name
    )

    print(
        "Jak logika wsparcia utrzymania infrastruktury B+R (Panda) "
        "odpowiada na problemy sektora nauki w Polsce?"
    )

    while True:
        query_str = input("Input query: ")
        query_str = query_str.strip()

        lang_str = TextUtils.text_language(query_str)

        print(f"<<<<<[{lang_str}] {query_str}")
        query_results = se.search(
            search_text=query_str,
            rerank_results=True,
            max_results=100,
            language=lang_str,
            additional_output_fields=[],
            return_with_factored_fields=False,
        )[0]

        texts_ids = []
        texts_scores = []
        for text_res in query_results:
            texts_ids.append(int(text_res["external_text_pk"]))
            texts_scores.append(text_res["score"])

        postgres_docs = ts.get_texts(
            texts_ids=texts_ids, texts_scores=texts_scores, surrounding_chunks=2
        )

        results = {
            "query": query_str,
            "stats": filter_stats(
                doc_stats=prepare_stats(postgres_docs),
                remove_under_hits=5,
                remove_under_pages=3,
            ),
            "answers": postgres_docs,
        }
        ofile = input("Nazwa pliku do zapisania: ")
        with open(ofile, "wt", encoding="utf8") as fout:
            fout.write(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
