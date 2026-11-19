"""
Tests for the model registry: catalog handling, per-model device placement and
the per-model inference lock.

The two loaders are replaced with spies, so nothing is downloaded or placed on
a device — what is under test is what the registry *asks for*.
"""

import json
import threading

import pytest

from sse_models.registry import ModelConfigError, ModelNotFoundError, ModelRegistry
from sse_models.settings import ModelServiceSettings


@pytest.fixture
def make_registry(tmp_path, monkeypatch):
    """
    Give tests a catalog writer plus a record of what was loaded.

    Returns ``(make, loads)``: ``make(embedders, rerankers)`` writes a config
    directory and hands back a registry over it; ``loads`` collects every
    ``(kind, directory_name, device)`` the registry asked to load.
    """
    loads = []

    class SpyST:
        def __init__(self, path, **kwargs):
            loads.append(("embedder", path, kwargs.get("device")))

    class SpyCE:
        def __init__(self, path, **kwargs):
            loads.append(("reranker", path, kwargs.get("device")))

    # the registry imports both inside _load, so patch them at their source
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", SpyST)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", SpyCE)

    def make(embedders, rerankers):
        cfg = tmp_path / "configs"
        cfg.mkdir(exist_ok=True)
        for entry in list(embedders) + list(rerankers):
            model_dir = tmp_path / entry["name"].replace("/", "_")
            model_dir.mkdir(exist_ok=True)
            entry["path"] = str(model_dir)
        (cfg / "embedders.json").write_text(
            json.dumps(
                {
                    "models": embedders,
                    "active_models": [e["name"] for e in embedders],
                }
            )
        )
        (cfg / "rerankers.json").write_text(
            json.dumps(
                {
                    "models": rerankers,
                    "active_models": [r["name"] for r in rerankers],
                }
            )
        )
        return ModelRegistry(ModelServiceSettings(env={"SSE_CONFIG_DIR": str(cfg)}))

    return make, loads


# ----------------------------------------------------------------------
# device placement
# ----------------------------------------------------------------------
def test_device_is_passed_through_per_model(make_registry):
    make, loads = make_registry
    registry = make(
        embedders=[
            {"name": "emb/one", "device": "cuda:1"},
            {"name": "emb/two", "device": "cuda:2"},
            {"name": "emb/plain"},
        ],
        rerankers=[{"name": "rrk/one", "device": "cuda:1"}],
    )

    for name in ("emb/one", "emb/two", "emb/plain"):
        registry.get("embedder", name)
    registry.get("reranker", "rrk/one")

    devices = {
        (kind, path.rsplit("/", 1)[-1]): device for kind, path, device in loads
    }
    assert devices == {
        ("embedder", "emb_one"): "cuda:1",
        ("embedder", "emb_two"): "cuda:2",
        ("embedder", "emb_plain"): "cpu",
        ("reranker", "rrk_one"): "cuda:1",
    }


def test_device_is_reported_in_the_catalog(make_registry):
    make, _ = make_registry
    registry = make(
        embedders=[{"name": "emb/a", "device": "cuda:1"}],
        rerankers=[{"name": "rrk/a", "device": "cuda:2"}],
    )
    info = {m["name"]: m["device"] for m in registry.list_models()}
    assert info == {"emb/a": "cuda:1", "rrk/a": "cuda:2"}


def test_blank_device_falls_back_to_cpu(make_registry):
    make, loads = make_registry
    registry = make(embedders=[{"name": "emb/a", "device": ""}], rerankers=[])
    registry.get("embedder", "emb/a")
    assert loads[0][2] == "cpu"


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------
def test_unknown_model_lists_what_exists(make_registry):
    make, _ = make_registry
    registry = make(embedders=[{"name": "emb/real"}], rerankers=[])
    with pytest.raises(ModelNotFoundError) as excinfo:
        registry.get("embedder", "emb/missing")
    assert "emb/real" in str(excinfo.value)


def test_missing_path_is_a_config_error(tmp_path):
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "embedders.json").write_text(
        json.dumps({"models": [{"name": "emb/a"}], "active_models": ["emb/a"]})
    )
    (cfg / "rerankers.json").write_text(json.dumps({"models": []}))

    registry = ModelRegistry(ModelServiceSettings(env={"SSE_CONFIG_DIR": str(cfg)}))
    with pytest.raises(ModelConfigError):
        registry.get("embedder", "emb/a")


def test_unknown_kind_is_rejected(make_registry):
    make, _ = make_registry
    registry = make(embedders=[], rerankers=[])
    with pytest.raises(ValueError):
        registry.list_models("summarizer")


# ----------------------------------------------------------------------
# per-model inference lock
# ----------------------------------------------------------------------
def test_same_model_shares_one_lock(make_registry):
    make, _ = make_registry
    registry = make(embedders=[{"name": "emb/a"}], rerankers=[])
    assert registry.inference_lock("embedder", "emb/a") is registry.inference_lock(
        "embedder", "emb/a"
    )


def test_different_models_do_not_share_a_lock(make_registry):
    """The multi-GPU guarantee: cuda:1 and cuda:2 must not serialize together."""
    make, _ = make_registry
    registry = make(embedders=[{"name": "emb/a"}], rerankers=[{"name": "rrk/a"}])

    embedder_lock = registry.inference_lock("embedder", "emb/a")
    reranker_lock = registry.inference_lock("reranker", "rrk/a")
    assert embedder_lock is not reranker_lock

    # holding one must not block the other
    with embedder_lock:
        assert reranker_lock.acquire(blocking=False)
        reranker_lock.release()


def test_lock_is_keyed_by_kind_and_name(make_registry):
    make, _ = make_registry
    registry = make(embedders=[{"name": "same"}], rerankers=[{"name": "same"}])
    assert registry.inference_lock(
        "embedder", "same"
    ) is not registry.inference_lock("reranker", "same")
    assert isinstance(
        registry.inference_lock("embedder", "same"), type(threading.Lock())
    )
