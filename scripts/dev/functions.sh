# Install bash dependencies

function end_installation_message() {
  echo ""
  echo ""
  echo "Copying ../scripts/run-api.sh to application dir $(pwd)"
  cp ../scripts/run-api.sh .
  chmod +x ./run-api.sh

  echo ""
  echo "Semantic search engine is successfully installed!"
  echo "If you wan to run api, use command:"
  echo ""
  echo ">> ./run-api.sh"
  echo ""
  echo "This command will run api using bash script from ../scripts/run-api.sh"
  echo ""
  cat ../scripts/run-api.sh
  echo ""
}


function add_default_organisation_and_user() {
  echo "Adding default organisation, group and user."

  cp ../scripts/add_user.sh .
  bash add_user.sh
  rm -f add_user.sh
}

function prepare_semantic_db() {
    echo "Preparing semantic database"

    APP_NAME="prepare_semantic_db.py"

    cp ../semantic_search_engine/apps_sse/admin/${APP_NAME} .
    python3 ${APP_NAME} --config configs/milvus_config.json
    rm -f ${APP_NAME}
}


function add_question_templates() {
    echo "Adding question templates"

    cp ../scripts/add_query_templates.sh ./
    bash add_query_templates.sh
    rm -f add_query_templates.sh
}
