### RAG Evaluation Documentation

This document describes the evaluation process for the Semantic Search Engine (SSE) RAG system. Evaluation is crucial
for ensuring the quality, accuracy, and reliability of the generated responses and the underlying retrieval mechanism.

---

### 1. Evaluation Overview

The project supports two main evaluation strategies:

- **LLM-as-a-Judge**: Uses a high-quality LLM to evaluate RAG responses across semantic dimensions (Faithfulness,
  Relevance, Precision).
- **Retrieval & Traditional Metrics**: Evaluates the search component by comparing retrieved documents against ground
  truth and calculates traditional NLP metrics (ROUGE, BLEU) for generated answers.

---

### 2. LLM-as-a-Judge Evaluation

This method uses an automated "judge" (another LLM) to score the RAG system's output. It is particularly effective for
assessing the quality of answers when natural language varies.

#### Script: `evaluate_rag_llm.py`

Located at: `sse_apps/not_tested/evaluator/evaluate_rag_llm.py`

#### Metrics Measured:

1. **Faithfulness**: Measures if the answer is derived solely from the retrieved context (avoids hallucinations).
2. **Answer Relevance**: Measures how well the answer addresses the user's question.
3. **Context Precision**: Evaluates whether the retrieved context contains the necessary information to answer the
   query.

#### Input Data Format (JSON):

A list of objects containing at least a `question`. `ground_truth` is optional but recommended.

```json
[
  {
    "question": "What are the company's rules regarding remote work?",
    "ground_truth": "Remote work is allowed up to 3 days a week with manager approval."
  }
]
```

#### How to Run:

```bash
python sse_apps/not_tested/evaluator/evaluate_rag_llm.py \
  -u <username> \
  -c <collection_name> \
  -i <input_questions.json> \
  -o <output_results.json> \
  --model <judge_model_name>
```

---

### 3. Retrieval & Traditional Metrics Evaluation

This method focuses on the performance of the search engine and uses statistical metrics to compare generated answers
with human-provided references.

#### Script: `eavaluate_embedder_search.py`

Located at: `sse_apps/not_tested/evaluator/eavaluate_embedder_search.py`

#### Metrics Measured:

- **Retrieval Performance**: % of expected files found in top-K results.
- **ROUGE (1, 2, L)**: Overlap of n-grams between generated and human answers.
- **BLEU (1-4)**: Precision of n-grams in generated answers.

#### Test Configuration Format:

This script requires a more complex JSON configuration defining test cases, collections, and expected results.

Example structure:

```json
{
  "test_configuration": {
    "config": {
      "generative_models": [
        "google/gemini-2.5-flash-lite"
      ],
      "search_options": {
        "max_results": [
          5,
          10
        ],
        "percentage_rank_mass": [
          0.8
        ]
      },
      "generate_options": {
        "answers_count": 1
      },
      "evaluators": {
        "generative": [
          "rouge",
          "bleu"
        ]
      }
    },
    "what_to_test": [
      "semantic_search",
      "generative_response"
    ]
  },
  "semantic_search_rag": {
    "collections": [
      "MyCollection"
    ],
    "run_examples": [
      "Example1"
    ],
    "test_examples": {
      "Example1": {
        "questions": [
          "How to reset password?"
        ],
        "human_answers": {
          "1": {
            "text_answer": "Go to settings and click reset.",
            "files": [
              {
                "file_name": "manual.pdf",
                "exact_match_file_name": true
              }
            ]
          }
        }
      }
    }
  }
}
```

#### How to Run:

```bash
python sse_apps/not_tested/evaluator/eavaluate_embedder_search.py \
  -u <username> \
  --test-configuration <test_config.json> \
  -o <output_results.xlsx>
```

The script generates several XLSX files:

- `<output>.xlsx`: Main results with metrics.
- `<output>-chunks.xlsx`: Detailed list of chunks retrieved for each question.
- `<output>-chunks-unique.xlsx`: Deduplicated chunks for manual review.

---

### 4. Prerequisites

Before running evaluation scripts, ensure:

1. **Environment is set up**: The script must be run from the project root with the appropriate `PYTHONPATH` or within
   the installed environment.
2. **Django Settings**: `DJANGO_SETTINGS_MODULE` must be set (usually `main.settings`).
3. **Milvus & LLM Access**: The Milvus database and the generative model API must be reachable as configured in
   `configs/`.

---

### 5. Recommendation for Regular Testing

For continuous improvement:

1. **Curate a Golden Dataset**: Maintain a set of ~50-100 diverse question-answer pairs with ground truth documents.
2. **Benchmark Changes**: Run the `evaluate_rag_llm.py` before and after significant changes to retrieval parameters
   (e.g., changing Reranker or RRF weights).
3. **Analyze Failures**: Use the generated XLSX files from `eavaluate_embedder_search.py` to identify which documents
   are frequently missed by the search engine.
