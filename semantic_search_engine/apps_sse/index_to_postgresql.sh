#!/bin/bash

cp apps_sse/add_files_from_dir.py . 

python3 add_files_from_dir.py \
	-d /mnt/data/radlab-projekty/EGO-GPT4-20230425/data/etap1/I_tura___do_16.06/07_NEW/Łódzkie \
	-o ego-converted-full-dataset-etap1.json \
	--clear-texts \
	--merge \
	--check-text-language \
	--proper-pages \
	--split-to-max-tokens-in-chunk=200 \
	--overlap-tokens-between-chunks=20 \
	--processes=10 \
	-c AiA2023

rm add_files_from_dir.py

