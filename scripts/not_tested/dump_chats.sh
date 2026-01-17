#!/bin/bash

DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_DIR="chat_dumps/${DATE_STR}"

APP_NAME=dump_user_chats.py
APP_DIR=../sse_apps/admin

cp ${APP_DIR}/${APP_NAME} .
python3 "${APP_NAME}" \
  --out-dir "${OUT_DIR}"
rm ${APP_NAME}
