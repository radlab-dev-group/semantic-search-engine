#!/bin/bash

COLLECTION_ID=1

DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_XLSX_FILE="collection_${COLLECTION_ID}_${DATE_STR}.xlsx"

APP_NAME=dump_collection_to_xlsx.py
APP_DIR=apps_sse/dataset
cp ${APP_DIR}/${APP_NAME} .

python3 "${APP_NAME}" \
    --collection-id "${COLLECTION_ID}" \
    --out-xlsx-file "${OUT_XLSX_FILE}"

rm ${APP_NAME}
