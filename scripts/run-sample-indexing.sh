#!/bin/bash

if [ "${1}" == "sample" ]
then
  SLEEP_TIME=3
  DATA_DIR=/mnt/data/radlab-projekty/EGO-GPT4-20230425/data/etap1/I_tura___do_16.06/07_NEW/Łódzkie
else
  SLEEP_TIME=7
  DATA_DIR=/mnt/data/radlab-projekty/EGO-GPT4-20230425/data/etap1/I_tura___do_16.06
fi

echo ""
echo "Running indexing directory ${DATA_DIR} in ${SLEEP_TIME} seconds"
echo "Press CTR+C to abort indexing..."
echo ""
sleep ${SLEEP_TIME}

cp apps_sse/add_files_from_dir.py .

python3 add_files_from_dir.py \
	-d ${DATA_DIR} \
	-c AiA2023 \
	-o /dev/null \
	--clear-texts \
	--check-text-language \
	--proper-pages \
	--merge-document-pages \
	--split-to-max-tokens-in-chunk=200 \
	--overlap-tokens-between-chunks=20 \
	--processes=10

rm add_files_from_dir.py 
