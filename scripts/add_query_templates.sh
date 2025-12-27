#!/bin/bash

Q_TEMPL_CONFIG="configs/query-templates.json"
KEYCLOAK_CONFIG="configs/auth-config.json"

cp apps_sse/admin/add_query_template_to_org.py .

python3 add_query_template_to_org.py \
  --keycloak-config ${KEYCLOAK_CONFIG} \
  --query-config ${Q_TEMPL_CONFIG}

rm add_query_template_to_org.py
