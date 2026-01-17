#!/bin/bash

USERNAME="default_user"
CONFIGS_TO_TEST=(
  ../tmp/tests/aia.json
)

DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_XLSX_FILE="eval_aia_${DATE_STR}.xlsx"

# Python app -- evaluator
APP_NAME=eavaluate_embedder_search.py
APP_DIR=../sse_apps/evaluator
cp ${APP_DIR}/${APP_NAME} .

# Run evaluator on each config
for config in ${CONFIGS_TO_TEST[*]}
do
  python3 "${APP_NAME}" \
    --username "${USERNAME}" \
    --test-configuration "${config}" \
    --out-xlsx-file "${OUT_XLSX_FILE}"
done

rm ${APP_NAME}
