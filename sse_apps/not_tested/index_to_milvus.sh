#!/bin/bash

cp apps_sse/index_collection_to_milvus.py .

# Available indexes: {IVF_FLAT, HNSW}
INDEX_TYPE=HNSW

python index_collection_to_milvus.py \
  --index-name ${INDEX_TYPE} \
	--from-collection AiA2023 \
	--to-collection AiA2023_${INDEX_TYPE} \
	--chunk-type clear_texts_proper_page_chunk_max_tokens_200_overlap_20

rm index_collection_to_milvus.py
