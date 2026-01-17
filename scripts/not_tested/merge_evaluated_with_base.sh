#!/bin/bash

APP_NAME=merge_evaluated_results.py
APP_DIR=apps_sse/evaluator
cp ${APP_DIR}/${APP_NAME} .

EVAL_RES_DIR=/mnt/data2/dev/radlab-projekty/Aia-2024/eval-datasets/<DATASET>>

python3 ${APP_NAME} \
  --base-file ${EVAL_RES_DIR}/DATASET.xlsx \
  --evaluated-file ${EVAL_RES_DIR}/DATASET_DONE.xlsx \
  --merged-out-file ${EVAL_RES_DIR}/DATASET_MERGED.xlsx

rm ${APP_NAME}

