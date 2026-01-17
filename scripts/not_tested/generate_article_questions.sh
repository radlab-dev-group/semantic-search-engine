#!/bin/bash

MIN_TEXT_LEN=200
COLLECTION=playground_news_stream_articles
OUT_FILE_PATH=article-questions-pLLama3_1-8B-DPO-L-30k-lora32_16_8bit.jsonl

APP_NAME=article_question_generator.py
APP_DIR=../sse_apps/dataset/generator

cp ${APP_DIR}/${APP_NAME} .

python3 ${APP_NAME} \
  --collection-name ${COLLECTION} \
  --min-text-len ${MIN_TEXT_LEN} \
  --output-file-path ${OUT_FILE_PATH}

rm ${APP_NAME}
