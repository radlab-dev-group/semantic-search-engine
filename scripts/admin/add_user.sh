#!/bin/bash

USER_CONFIG="configs/user-group-organisation.json"

cp ../apps_sse/admin/add_org_group_user.py .

python3 add_org_group_user.py --user-config ${USER_CONFIG}

rm -f add_org_group_user.py
