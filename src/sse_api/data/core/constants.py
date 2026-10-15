"""
Defaults of the chunking of a collection.

A collection decides once how its texts are split; the values below are used
when a collection is created without saying otherwise.  They are also what a
UI can prefill its form with, see the ``filtering_options`` endpoint.
"""

#: Key of the chunking settings inside ``CollectionOfDocuments.metadata_json``.
CHUNKING_METADATA_KEY = "chunking"

#: Key of the strict-chunking flag inside the same dictionary.
STRICT_CHUNKING_METADATA_KEY = "strict_chunking"

#: Sub-keys of the chunking settings, named exactly like in ``indexing_options``.
MAX_TOKENS_IN_CHUNK_KEY = "max_tokens_in_chunk"
NUMBER_OF_OVERLAP_TOKENS_KEY = "number_of_overlap_tokens"

#: Default size of a text chunk, in tokens.
DEFAULT_MAX_TOKENS_IN_CHUNK = 500

#: Default number of tokens repeated between two neighbouring chunks.
DEFAULT_NUMBER_OF_OVERLAP_TOKENS = 0

#: Upper bound of a chunk when the embedder does not declare its own limit.
#: It only guards against typos such as ``500000``; a model which declares
#: ``max_model_length`` in ``configs/embedders.json`` is always stricter.
MAX_TOKENS_IN_CHUNK_LIMIT = 100000
