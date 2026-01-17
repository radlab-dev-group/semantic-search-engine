#!/bin/bash

AUTH_CONFIG="configs/auth-config.json"
QUERY_TEMPL_CONFIG="configs/query-templates.json"

cp ../sse_apps/admin/add_query_template_to_org.py .

python3 add_query_template_to_org.py \
  --auth-config ${AUTH_CONFIG} \
  --query-config ${QUERY_TEMPL_CONFIG}

rm add_query_template_to_org.py
