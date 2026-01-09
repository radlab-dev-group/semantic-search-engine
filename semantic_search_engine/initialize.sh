#!/bin/bash

INSTALLATION_MODE_NAME=${1}


python3 manage.py makemigrations

#
#export PIP_BREAK_SYSTEM_PACKAGES=1
#
#source ../scripts/dev/functions.sh
#
#check_args_branch "${INSTALLATION_MODE_NAME}"
#BRANCH_NAME=$(resolve_branch_name "${INSTALLATION_MODE_NAME}")
#
#clear_django_core
#
#install_radlab_data_and_copy_apps "${BRANCH_NAME}" "apps_sse/installed"
#install_radlab_semantic_search_db "${BRANCH_NAME}"
#install_radlab_text_cleaner "${BRANCH_NAME}"
#install_radlab_content_supervisor "${BRANCH_NAME}"
#install_radlab_django_core "${BRANCH_NAME}"
#
#install_urls ../tmp/urls
#copy_configs ../tmp/configs
#copy_resources ../tmp/resources
#
#if [ $# -eq 1 ]
#then
#  make_and_migrate
#  prepare_semantic_db
#  add_default_organisation_and_user
#  add_question_templates
#  end_installation_message
#  remove_bash_dependencies
#fi
