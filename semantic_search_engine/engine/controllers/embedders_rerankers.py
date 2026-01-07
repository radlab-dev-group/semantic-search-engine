"""
embedders_rerankers.py
---------------------------------
Utility module for loading and accessing configuration of embedding and
reranking models.

The module defines two global registries:

* ``ALL_AVAILABLE_EMBEDDERS_MODELS`` – a mapping from embedder name to its
  configuration dictionary (containing ``path``, ``vector_size`` and ``device``).

* ``ALL_AVAILABLE_RERANKERS_MODELS`` – a mapping from reranker name to its
  configuration dictionary (containing ``path`` and ``device``).

A helper class :class:`EmbeddingModelsConfig` reads JSON configuration files
and populates these registries.  The class also provides static lookup helpers
to retrieve model‑specific information and convenience methods to list the
currently active models.
"""

import json

ALL_AVAILABLE_EMBEDDERS_MODELS = {}
ALL_AVAILABLE_RERANKERS_MODELS = {}


class EmbeddingModelsConfig:
    """
    Configuration loader for embedder and reranker models.

        Parameters
        ----------
        embedders_config : str, optional
            Path to the JSON file describing embedder models. Defaults to
            ``"configs/embedders.json"``.
        rerankers_config : str, optional
            Path to the JSON file describing reranker models. Defaults to
            ``"configs/rerankers.json"``.

        The constructor loads the configuration files (if the supplied strings are
        non‑empty) and populates the global ``ALL_AVAILABLE_EMBEDDERS_MODELS`` and
        ``ALL_AVAILABLE_RERANKERS_MODELS`` dictionaries with the *active* model
        entries.
    """

    def __init__(
        self,
        embedders_config: str = "configs/embedders.json",
        rerankers_config: str = "configs/rerankers.json",
    ):
        """
        Create a new configuration loader.

                Parameters
                ----------
                embedders_config : str
                    Path to the embedder configuration JSON file.
                rerankers_config : str
                    Path to the reranker configuration JSON file.

                The method stores the provided paths and immediately loads the files
                (if the supplied strings are non‑empty) by delegating to the private
                ``_load_embedders_cfg`` and ``_load_rerankers_cfg`` helpers.
        """
        self._embedders_config = embedders_config
        self._rerankers_config = rerankers_config

        if len(embedders_config):
            self._load_embedders_cfg()

        if len(rerankers_config):
            self._load_rerankers_cfg()

    @staticmethod
    def get_embedder_path(model_name):
        """
        Return the filesystem path for the given embedder model.

               Parameters
               ----------
               model_name : str
                   Name of the embedder model as listed in ``ALL_AVAILABLE_EMBEDDERS_MODELS``.

               Returns
               -------
               str
                   Path to the model files.
        """
        return ALL_AVAILABLE_EMBEDDERS_MODELS[model_name]["path"]

    @staticmethod
    def get_embedder_vector_size(model_name):
        """
        Return the dimensionality of the vectors produced by an embedder.

                Parameters
                ----------
                model_name : str
                    Name of the embedder model.

                Returns
                -------
                int
                    Vector size (dimensionality) for the specified model.
        """
        return ALL_AVAILABLE_EMBEDDERS_MODELS[model_name]["vector_size"]

    @staticmethod
    def get_embedder_device(model_name):
        """
        Return the compute device (e.g., ``cpu`` or ``cuda``) for an embedder.

        Parameters
        ----------
        model_name : str
            Name of the embedder model.

        Returns
        -------
        str
            Device identifier used by the model.
        """
        return ALL_AVAILABLE_EMBEDDERS_MODELS[model_name]["device"]

    @staticmethod
    def get_reranker_path(model_name):
        """
        Return the filesystem path for the given reranker model.

                Parameters
                ----------
                model_name : str
                    Name of the reranker model as listed in ``ALL_AVAILABLE_RERANKERS_MODELS``.

                Returns
                -------
                str
                    Path to the reranker model files.
        """
        return ALL_AVAILABLE_RERANKERS_MODELS[model_name]["path"]

    @staticmethod
    def get_reranker_device(model_name):
        """
        Return the compute device for a reranker model.

                Parameters
                ----------
                model_name : str
                    Name of the reranker model.

                Returns
                -------
                str
                    Device identifier (e.g., ``cpu`` or ``cuda``) for the reranker.
        """
        return ALL_AVAILABLE_RERANKERS_MODELS[model_name]["device"]

    @staticmethod
    def embedders():
        """
        List the names of all currently loaded embedder models.

                Returns
                -------
                list[str]
                    Keys of ``ALL_AVAILABLE_EMBEDDERS_MODELS``.
        """
        return list(ALL_AVAILABLE_EMBEDDERS_MODELS.keys())

    @staticmethod
    def rerankers():
        """
        List the names of all currently loaded reranker models.

                Returns
                -------
                list[str]
                    Keys of ``ALL_AVAILABLE_RERANKERS_MODELS``.
        """
        return list(ALL_AVAILABLE_RERANKERS_MODELS.keys())

    def _load_embedders_cfg(self):
        """
        Load embedder configuration from ``self._embedders_config``.

                The method parses the JSON file, builds a temporary ``name → config``
                map, and then copies only the models listed under ``active_models`` into
                the global ``ALL_AVAILABLE_EMBEDDERS_MODELS`` dictionary.
        """
        with open(self._embedders_config, "rt") as models_in:
            whole_emb = json.load(models_in)

        m2config = {}
        for m in whole_emb["models"]:
            m2config[m["name"]] = m

        for a_model in whole_emb["active_models"]:
            ALL_AVAILABLE_EMBEDDERS_MODELS[a_model] = m2config[a_model]

    def _load_rerankers_cfg(self):
        """
        Load reranker configuration from ``self._rerankers_config``.

                Works analogously to :meth:`_load_embedders_cfg`, but populates the
                ``ALL_AVAILABLE_RERANKERS_MODELS`` dictionary only if a model is not
                already present (prevents accidental overwriting).
        """
        with open(self._rerankers_config, "rt") as models_in:
            whole_emb = json.load(models_in)

        m2config = {}
        for m in whole_emb["models"]:
            m2config[m["name"]] = m

        for a_model in whole_emb["active_models"]:
            if a_model not in ALL_AVAILABLE_RERANKERS_MODELS:
                ALL_AVAILABLE_RERANKERS_MODELS[a_model] = m2config[a_model]
