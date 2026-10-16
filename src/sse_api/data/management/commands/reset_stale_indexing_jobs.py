"""
reset_stale_indexing_jobs
-------------------------

Closes indexing jobs which no worker is working on anymore.

A worker killed in the middle of a document leaves its job in a running status
and its documents in ``pending`` / ``indexing``.  Nothing would ever change
that, so a client would wait forever.  This command marks everything which is
still running after ``--minutes`` as failed; the documents can be indexed again
by uploading them once more.

Examples::

    python manage.py reset_stale_indexing_jobs
    python manage.py reset_stale_indexing_jobs --minutes 240
    python manage.py reset_stale_indexing_jobs --dry-run
"""

from django.core.management.base import BaseCommand

from sse_api.data.core import indexing_progress
from sse_api.data.core.constants import DEFAULT_INDEXING_STALE_MINUTES


class Command(BaseCommand):
    help = (
        "Mark indexing jobs and documents which are still running after too "
        "long as failed."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--minutes",
            type=int,
            default=DEFAULT_INDEXING_STALE_MINUTES,
            help="How long a job may run before it is treated as lost",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be marked, write nothing",
        )

    def handle(self, *args, **options):
        minutes = options["minutes"]
        if minutes <= 0:
            self.stderr.write(self.style.ERROR("--minutes has to be positive"))
            return

        if options["dry_run"]:
            self.stdout.write(
                "Would mark as failed: {} jobs, {} documents "
                "(older than {} minutes)".format(
                    indexing_progress.stale_jobs(minutes).count(),
                    indexing_progress.stale_documents(minutes).count(),
                    minutes,
                )
            )
            return

        reset = indexing_progress.reset_stale_jobs(minutes=minutes)
        self.stdout.write(
            self.style.SUCCESS(
                f"Marked as failed: {reset['jobs']} jobs, {reset['documents']} "
                f"documents (older than {minutes} minutes)"
            )
        )
