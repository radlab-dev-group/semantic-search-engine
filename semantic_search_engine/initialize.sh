#!/bin/bash

INSTALLATION_MODE_NAME=${1}

if [[ "$INSTALLATION_MODE_NAME" == "clear" ]]; then
  echo "🔧 Clearing Django migration files..."
  find . -type f -path "*/migrations/*.py" ! -name "__init__.py" -delete
  find . -type f -path "*/migrations/*.pyc" -delete
fi


if [[ "$INSTALLATION_MODE_NAME" == "dep" || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "📦 Installing dependencies"
  echo "   🔧 installing radlab-data"
  pip install git+https://github.com/radlab-dev-group/radlab-data.git
  echo "   🔧 installing llm-router"
  pip install git+https://github.com/radlab-dev-group/llm-router.git
fi

if [[ "$INSTALLATION_MODE_NAME" == "migrate"  || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "🚀 Running migrations..."
  python3 manage.py makemigrations
  python3 manage.py migrate
fi

if [[ "$INSTALLATION_MODE_NAME" == "semantic"  || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "📚 Preparing semantic database"
  cp apps_sse/admin/prepare_semantic_db.py .
  python3 prepare_semantic_db.py
  rm -f prepare_semantic_db.py
fi


if [[ "$INSTALLATION_MODE_NAME" == "add_user"  || "$INSTALLATION_MODE_NAME" == "all" ]]; then
  echo "👤 Adding default user"
  cp ../scripts/admin/add_user.sh .
  bash add_user.sh
  rm -f add_user.sh
fi


#then
#  add_question_templates
#  end_installation_message
#  remove_bash_dependencies
#fi
