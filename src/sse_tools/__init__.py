"""
sse_tools
---------

Standalone command-line tools for the Semantic Search Engine.

Each subpackage holds one family of tools:

* ``sse_tools.admin`` — administration (users, organisations, collections).
* ``sse_tools.installed`` — file/dataset converters.
* ``sse_tools.evaluation`` — dataset generators and search evaluators.
* ``sse_tools.indexing`` — bulk indexing helpers.

Every tool is a plain script with an ``if __name__ == "__main__"`` entry
point.  They are run from the repository root so that ``configs/`` and the
installed ``sse_api`` package resolve naturally:

    python3 src/sse_tools/admin/add_org_group_user.py -u configs/user-group-organisation.json
"""
