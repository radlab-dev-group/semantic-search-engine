#!/bin/bash

INSTALLATION_MODE_NAME=${1}


if [[ "$INSTALLATION_MODE_NAME" == "dep" || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "📦 Installing dependencies"
  echo "   🔧 installing radlab-data"
  pip install git+https://github.com/radlab-dev-group/radlab-data.git
  echo "   🔧 installing llm-router"
  pip install git+https://github.com/radlab-dev-group/llm-router.git
fi

if [[ "$INSTALLATION_MODE_NAME" == "migrate"  || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "🚀 Running migrations..."
  python3 -m sse_api.manage migrate
fi

if [[ "$INSTALLATION_MODE_NAME" == "semantic"  || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "📚 Preparing semantic database"
  python3 ../sse_tools/admin/prepare_semantic_db.py
fi


if [[ "$INSTALLATION_MODE_NAME" == "add_user"  || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "👤 Adding default user"
  python3 ../sse_tools/admin/add_org_group_user.py -u ../../configs/user-group-organisation.json
fi

if [[ "$INSTALLATION_MODE_NAME" == "add_query_templates"  || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "📄 Adding query templates"
  python3 ../sse_tools/admin/add_query_template_to_org.py
fi
