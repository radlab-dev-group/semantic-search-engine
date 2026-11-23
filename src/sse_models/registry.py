"""
registry.py
-----------

Model catalog and loader for the model service.

The catalog is read from the same ``configs/embedders.json`` and
``configs/rerankers.json`` the main engine uses, so a model registered for the
engine is automatically served here.  Models are addressed by their catalog
*name*; a client can never ask the service to load an arbitrary filesystem
path.

Loading is lazy and thread-safe: the first request for a model pays the load
cost, every later request reuses the cached instance.  ``warm()`` preloads
everything active (opt-in via ``SSE_MODELS_WARM_MODELS``).
"""

import json
import os
import threading
from typing import Any, Dict, List

from sse_models.settings import ModelServiceSettings


class ModelNotFoundError(Exception):
    """Requested model name is not present in the catalog."""


class ModelConfigError(Exception):
    """The catalog entry for a model is unusable (missing path, bad json)."""


def _read_catalog(path: str, kind: str) -> Dict[str, Dict[str, Any]]:
    """
    Read one model catalog file.

    Returns a mapping ``name -> entry`` limited to the names listed in
    ``active_models`` (when that key is present and non-empty), so a model
    that was deactivated for the engine is not served here either.
    """
    if not os.path.isfile(path):
        raise ModelConfigError(f"{kind} configuration not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigError(f"cannot read {kind} config {path}: {exc}") from exc

    entries = raw.get("models") or []
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        name = entry.get("name")
        if not name:
            continue
        by_name[name] = entry

    active = raw.get("active_models") or []
    if active:
        by_name = {name: by_name[name] for name in active if name in by_name}

    return by_name


class ModelRegistry:
    """
    Lazy, cached registry of embedder and reranker models.

    A model is loaded once and reused.  The loader is guarded by a per-model
    lock so concurrent first requests do not load the same weights twice.
    """

    def __init__(self, settings: ModelServiceSettings | None = None) -> None:
        self._settings = settings or ModelServiceSettings()
        self._lock = threading.Lock()
        self._model_locks: Dict[str, threading.Lock] = {}
        # One inference lock per loaded model, never one for the whole process:
        # two models pinned to different GPUs (see the ``device`` key in the
        # catalog) have nothing to serialize against each other, while a single
        # model still serves one batch at a time so its latency stays flat.
        self._inference_locks: Dict[str, threading.Lock] = {}
        self._embedders: Dict[str, Dict[str, Any]] | None = None
        self._rerankers: Dict[str, Dict[str, Any]] | None = None
        self._denoiser: Dict[str, Any] | None = None
        self._loaded: Dict[str, Any] = {}
        self._load_errors: Dict[str, str] = {}
        self._tokenizer_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # catalog
    # ------------------------------------------------------------------
    def _catalog(self, kind: str) -> Dict[str, Dict[str, Any]]:
        if kind == "embedder":
            if self._embedders is None:
                self._embedders = _read_catalog(
                    self._settings.embedders_config_path(), "embedders"
                )
            return self._embedders

        if kind == "reranker":
            if self._rerankers is None:
                self._rerankers = _read_catalog(
                    self._settings.rerankers_config_path(), "rerankers"
                )
            return self._rerankers

        raise ValueError(f"unknown model kind: {kind}")

    def list_models(self, kind: str | None = None) -> List[Dict[str, Any]]:
        """
        Describe the catalog for one kind (or both when ``kind`` is ``None``).

        The payload is deliberately free of absolute paths — callers only need
        to know which names exist and whether they are already loaded.
        """
        kinds = [kind] if kind else ["embedder", "reranker"]
        info: List[Dict[str, Any]] = []
        for one_kind in kinds:
            for name, entry in self._catalog(one_kind).items():
                info.append(
                    {
                        "name": name,
                        "kind": one_kind,
                        "device": entry.get("device", "cpu"),
                        "max_model_length": entry.get("max_model_length"),
                        "vector_size": entry.get("vector_size"),
                        "loaded": f"{one_kind}:{name}" in self._loaded,
                    }
                )
        return info

    def has_model(self, kind: str, name: str) -> bool:
        return name in self._catalog(kind)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def get(self, kind: str, name: str) -> Any:
        """
        Return the model instance for ``kind``/``name``, loading it if needed.

        Raises :class:`ModelNotFoundError` when the name is not in the catalog
        and :class:`ModelConfigError` when the entry or the load fails.
        """
        catalog = self._catalog(kind)
        if name not in catalog:
            raise ModelNotFoundError(
                f"{kind} model '{name}' is not configured; "
                f"available: {', '.join(sorted(catalog)) or 'none'}"
            )

        key = f"{kind}:{name}"
        cached = self._loaded.get(key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(key)
            if cached is not None:
                return cached
            model_lock = self._model_locks.setdefault(key, threading.Lock())

        with model_lock:
            cached = self._loaded.get(key)
            if cached is not None:
                return cached
            model = self._load(kind, name, catalog[name])
            self._loaded[key] = model
            self._load_errors.pop(key, None)
            return model

    def _load(self, kind: str, name: str, entry: Dict[str, Any]) -> Any:
        path = entry.get("path")
        if not path:
            raise ModelConfigError(f"{kind} model '{name}' has no path configured")
        if not os.path.isdir(path):
            raise ModelConfigError(f"{kind} model '{name}' path not found: {path}")

        # ``device`` is passed through to the loader untouched, so any torch
        # device string works per model: "cpu", "cuda", "cuda:1", "cuda:2".
        # Nothing validates it — a typo fails at load with the torch error,
        # which is more useful than a second, narrower vocabulary here.
        device = entry.get("device") or "cpu"
        load_kwargs = {"device": device, "trust_remote_code": True}

        if kind == "embedder":
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(path, **load_kwargs)

        if kind == "reranker":
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(path, **load_kwargs)
            _ensure_pad_token(model)
            return model

        raise ValueError(f"unknown model kind: {kind}")

    # ------------------------------------------------------------------
    # tokenizers (shared by /v1/tokenize and /v1/truncate)
    # ------------------------------------------------------------------
    def tokenizer_for(self, kind: str, name: str):
        """
        Fast tokenizer of a catalog model, loaded once and cached.

        Tokenizers are read from the same catalog path as the weights and
        cached separately from the model itself: counting tokens must not
        force the multi-gigabyte weights into memory.
        """
        key = f"tokenizer:{kind}:{name}"
        cached = self._tokenizer_cache.get(key)
        if cached is not None:
            return cached

        catalog = self._catalog(kind)
        if name not in catalog:
            raise ModelNotFoundError(
                f"{kind} model '{name}' is not configured; "
                f"available: {', '.join(sorted(catalog)) or 'none'}"
            )
        path = catalog[name].get("path")
        if not path or not os.path.isdir(path):
            raise ModelConfigError(
                f"{kind} model '{name}' has no usable path for its tokenizer"
            )

        with self._lock:
            cached = self._tokenizer_cache.get(key)
            if cached is not None:
                return cached
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self._tokenizer_cache[key] = tokenizer
            return tokenizer

    def max_length_of(self, kind: str, name: str) -> int:
        """Declared ``max_model_length`` of a catalog model, ``0`` when unknown."""
        catalog = self._catalog(kind)
        entry = catalog.get(name) or {}
        try:
            return int(entry.get("max_model_length") or 0)
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # denoiser (T5 text-cleaning model, optional)
    # ------------------------------------------------------------------
    def denoiser(self):
        """
        The text denoiser (T5), loaded once and cached.

        The denoiser is not part of the embedder/reranker catalogs: it is a
        separate generative model configured in ``configs/models.json`` (the
        ``denoiser`` section, with ``model_path`` and ``device``).  Loading it
        is entirely lazy — a service that never answers ``/v1/denoise`` never
        pays for its weights.
        """
        key = "denoiser"
        cached = self._loaded.get(key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(key)
            if cached is not None:
                return cached
            model = self._load_denoiser()
            self._loaded[key] = model
            self._load_errors.pop(key, None)
            return model

    def _load_denoiser(self):
        config_path = os.path.join(self._settings.config_dir, "models.json")
        if not os.path.isfile(config_path):
            raise ModelConfigError(
                f"denoiser configuration not found: {config_path}"
            )
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelConfigError(
                f"cannot read denoiser config {config_path}: {exc}"
            ) from exc

        section = config.get("denoiser") or {}
        path = section.get("model_path")
        if not path:
            raise ModelConfigError("denoiser section has no model_path")
        device = section.get("device") or "cpu"

        from transformers import T5ForConditionalGeneration, T5Tokenizer

        model = T5ForConditionalGeneration.from_pretrained(path)
        try:
            import torch

            model.to(
                device
                if torch.cuda.is_available() or not device.startswith("cuda")
                else "cpu"
            )
        except Exception:  # torch missing / no device: keep the CPU copy
            pass
        tokenizer = T5Tokenizer.from_pretrained(path)
        return {"model": model, "tokenizer": tokenizer, "path": path}

    # ------------------------------------------------------------------
    # extractive QA (question-answering pipeline, optional)
    # ------------------------------------------------------------------
    def qa_model(self):
        """
        The extractive QA pipeline, loaded once and cached.

        Like the denoiser it is not part of the embedder/reranker catalogs:
        the ``extractive_qa`` section of ``configs/models.json`` carries its
        ``model_path`` and ``device``.  Loading is entirely lazy — a service
        that never answers ``/v1/extractive_qa`` never pays for its weights.
        """
        key = "qa_model"
        cached = self._loaded.get(key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(key)
            if cached is not None:
                return cached
            model = self._load_qa_model()
            self._loaded[key] = model
            self._load_errors.pop(key, None)
            return model

    def _load_qa_model(self):
        from sse_models.extractive_qa import load_qa_pipeline

        config_path = os.path.join(self._settings.config_dir, "models.json")
        if not os.path.isfile(config_path):
            raise ModelConfigError(
                f"extractive_qa configuration not found: {config_path}"
            )
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelConfigError(
                f"cannot read extractive_qa config {config_path}: {exc}"
            ) from exc

        section = config.get("extractive_qa") or {}
        path = section.get("model_path") or section.get("model_name")
        if not path:
            raise ModelConfigError(
                "extractive_qa section has no model_path (or model_name)"
            )
        device = section.get("device") or "cpu"

        return load_qa_pipeline(path, device=device)

    def inference_lock(self, kind: str, name: str) -> threading.Lock:
        """
        Lock guarding inference on one model.

        Deliberately per model. A catalog may pin the embedder to ``cuda:1``
        and the reranker to ``cuda:2``; a process-wide lock would run those two
        device queues one after the other and throw away the separation the
        ``device`` values were chosen for.
        """
        key = f"{kind}:{name}"
        with self._lock:
            lock = self._inference_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._inference_locks[key] = lock
            return lock

    def warm(self, kind: str | None = None) -> Dict[str, str]:
        """
        Preload every active model for ``kind`` (both kinds by default).

        A model that fails to load does not abort the warm-up: the error is
        recorded per model so the health endpoint can report it and the other
        models stay usable.
        """
        kinds = [kind] if kind else ["embedder", "reranker"]
        status: Dict[str, str] = {}
        for one_kind in kinds:
            for name in self._catalog(one_kind):
                key = f"{one_kind}:{name}"
                try:
                    self.get(one_kind, name)
                    status[key] = "loaded"
                except Exception as exc:  # noqa: BLE001 - warm-up must not raise
                    self._load_errors[key] = str(exc)
                    status[key] = f"error: {exc}"
        return status

    def loaded(self) -> List[str]:
        return sorted(self._loaded)

    def errors(self) -> Dict[str, str]:
        return dict(self._load_errors)

    def stats(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded(),
            "errors": self.errors(),
        }


def _ensure_pad_token(model) -> None:
    """
    Give a reranker a pad token when it has none.

    Rerankers derived from causal LLMs (Qwen3-Reranker is one) ship a tokenizer
    whose vocabulary has no padding symbol, and their ``config.json`` has no
    ``pad_token_id`` either.  Scoring more than one pair at a time then fails
    twice over: the tokenizer cannot pad the batch, and the sequence-classifier
    head refuses ``batch_size > 1`` outright ("Cannot handle batch sizes > 1 if
    no padding token is defined") because it needs the pad id to find the last
    real token to pool from.

    ``eos_token`` is the conventional substitute.  The attention mask keeps
    padded positions out of the computation and the head pools the rightmost
    non-pad token, so a pair's score does not depend on the length of the
    pairs it happens to be batched with.

    Models that already define a pad token are left untouched.
    """
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        substitute = tokenizer.eos_token
        if substitute is None and len(tokenizer) > 0:
            substitute = tokenizer.convert_ids_to_tokens(len(tokenizer) - 1)
        if substitute is not None:
            tokenizer.pad_token = substitute

    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        # Left padding keeps every real token at a fixed offset from the end of
        # the row, which is what a causal model expects.
        tokenizer.padding_side = "left"

    # The scoring head reads the id off a model config, not off the tokenizer,
    # so a tokenizer-only fix would still be rejected.  sentence-transformers
    # keeps its own config copy next to the one baked into the wrapped HF
    # model, and the head uses the latter, so both have to be set.
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        return

    candidates = [model, getattr(model, "model", None)]
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None and getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = pad_id
