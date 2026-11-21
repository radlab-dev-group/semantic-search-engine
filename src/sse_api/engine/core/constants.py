"""
constants.py
------------

Defaults of the search engine which are shared by the protocol, the database
adapters and the indexing paths.  The module stays free of any Django import,
so ``engine.core`` keeps working outside of a running site.
"""

#: Minimum length, in characters, a chunk must have to reach the index.
#:
#: A chunk shorter than this carries no meaning of its own (a page number,
#: an OCR artefact, a table cell).  The limit used to be 300 characters,
#: which threw away short but essential entries: dictionary definitions,
#: single legal articles, price-list positions.  40 characters keeps those
#: while still filtering out the noise.
MIN_CHUNK_CHAR_LEN = 40
