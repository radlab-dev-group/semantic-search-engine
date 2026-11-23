"""
schemas.py
----------

Pydantic request/response models for the model service HTTP API.

Two operations, two endpoints:

* ``POST /v1/embed``   — turn texts into vectors (indexing + query embedding)
* ``POST /v1/rerank``  — score ``query`` against ``documents`` (cross-encoder)

Keeping them separate (rather than one "pipeline" endpoint) lets the engine
call only what it needs: the reranker is the expensive part and is scaled and
deployed independently of the embedder.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# /v1/embed
# ----------------------------------------------------------------------
class EmbedRequest(BaseModel):
    """Embed a batch of texts with a named embedder model."""

    model: str = Field(..., description="Embedder name from configs/embedders.json")
    texts: List[str] = Field(
        ..., min_length=1, description="Texts to embed (at least one)"
    )
    normalize: Optional[bool] = Field(
        None,
        description=(
            "L2-normalise the vectors. Omit to use the service default "
            "(normalised, which is what cosine search in Milvus expects)."
        ),
    )
    batch_size: Optional[int] = Field(
        None, ge=1, description="Inference batch size (service default when omitted)"
    )


class EmbedResponse(BaseModel):
    model: str
    dim: int = Field(..., description="Dimensionality of the returned vectors")
    embeddings: List[List[float]]
    count: int
    took_ms: float


# ----------------------------------------------------------------------
# /v1/rerank
# ----------------------------------------------------------------------
class RerankRequest(BaseModel):
    """Score documents against a query with a named reranker model."""

    model: str = Field(..., description="Reranker name from configs/rerankers.json")
    query: str = Field(..., description="Query text")
    documents: List[str] = Field(
        ..., min_length=1, description="Candidate texts to score"
    )
    top_n: Optional[int] = Field(
        None,
        ge=1,
        description="Return only this many best-scoring documents (all when omitted)",
    )
    return_documents: bool = Field(
        False,
        description="Include the document text in the response (off by default; "
        "the engine correlates results by index and does not need it)",
    )
    batch_size: Optional[int] = Field(
        None, ge=1, description="Inference batch size (service default when omitted)"
    )


class RerankHit(BaseModel):
    index: int = Field(..., description="Position of the text in the request list")
    score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    model: str
    results: List[RerankHit]
    count: int = Field(..., description="Number of hits returned")
    scored: int = Field(..., description="Number of documents actually scored")
    took_ms: float


# ----------------------------------------------------------------------
# discovery / health
# ----------------------------------------------------------------------
class ModelInfo(BaseModel):
    name: str
    kind: str
    device: str
    max_model_length: Optional[int] = None
    vector_size: Optional[int] = None
    loaded: bool = False


class ModelsResponse(BaseModel):
    role: str
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    role: str
    loaded: List[str]
    errors: Dict[str, str] = Field(default_factory=dict)
    details: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ----------------------------------------------------------------------
# /v1/tokenize, /v1/truncate
# ----------------------------------------------------------------------
class TokenizeRequest(BaseModel):
    """Count the tokens of a batch of texts with a named model tokenizer."""

    model: str = Field(
        ..., description="Model name from the catalog (embedder or reranker)"
    )
    texts: List[str] = Field(
        ..., min_length=1, description="Texts to count (at least one)"
    )


class TokenizeResponse(BaseModel):
    model: str
    token_counts: List[int]
    max_length: int = Field(
        ...,
        description="Maximum model input length in tokens; 0 when unknown",
    )
    took_ms: float


class TruncateRequest(BaseModel):
    """Truncate texts to a token budget with a named model tokenizer."""

    model: str = Field(
        ..., description="Model name from the catalog (embedder or reranker)"
    )
    texts: List[str] = Field(
        ..., min_length=1, description="Texts to truncate (at least one)"
    )
    max_tokens: int = Field(..., ge=1, description="Token budget for every text")


class TruncateResponse(BaseModel):
    model: str
    truncated_texts: List[str]
    was_truncated: List[bool]
    took_ms: float


# ----------------------------------------------------------------------
# /v1/denoise
# ----------------------------------------------------------------------
class DenoiseRequest(BaseModel):
    """Clean a noisy text (OCR artefacts, broken sentences)."""

    text: str = Field(..., min_length=1, description="Text to denoise")
    max_length: Optional[int] = Field(
        None,
        ge=1,
        description="Model input window in tokens (model default when omitted)",
    )
    batch_size: Optional[int] = Field(
        None,
        ge=1,
        description="Inference batch size (service default when omitted)",
    )


class DenoiseResponse(BaseModel):
    cleaned_text: str
    correctness_score: float = Field(
        ...,
        description="Similarity of the cleaned text to the input, in [0, 1]",
    )
    took_ms: float


# ----------------------------------------------------------------------
# /v1/extractive_qa
# ----------------------------------------------------------------------
class ExtractiveQARequest(BaseModel):
    """Answer a question with spans of the retrieved passages."""

    question: str = Field(..., min_length=1, description="The question to answer")
    contexts: List[str] = Field(
        ..., min_length=1, description="Retrieved passages (at least one)"
    )


class ExtractiveQAAnswer(BaseModel):
    """The span one passage offers, with its confidence."""

    answer: str = Field(..., description="Extracted span, verbatim from the passage")
    score: float = Field(..., description="Model confidence, in [0, 1]")


class ExtractiveQAResponse(BaseModel):
    """One answer per input passage, in input order."""

    answers: List[ExtractiveQAAnswer]
    count: int
    took_ms: float


AnyModel = Any
