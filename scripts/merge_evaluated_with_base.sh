#!/bin/bash

APP_NAME=merge_evaluated_results.py
APP_DIR=apps_sse/evaluator
cp ${APP_DIR}/${APP_NAME} .

EVAL_RES_DIR=/mnt/data2/dev/radlab-projekty/Aia-2024/eval-datasets/akozak-ocena-reczna-skala_0-6/20240827

python3 ${APP_NAME} \
  --base-file ${EVAL_RES_DIR}/eval_aia_20240927_113150.xlsx \
  --evaluated-file ${EVAL_RES_DIR}/eval_aia_20240927_113150_akozak_done.xlsx \
  --merged-out-file ${EVAL_RES_DIR}/eval_aia_20240927_113150_akozak_done_merged.xlsx

rm ${APP_NAME}

