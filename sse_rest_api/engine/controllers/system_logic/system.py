"""
system.py
---------

Controller that encapsulates operations on ``UserQueryResponseAnswer``
objects which are not directly tied to a Django view.  Currently, it
provides a method for persisting a user rating (value, maximum possible
value, and optional comment) for a given answer.
"""

from engine.models import UserQueryResponseAnswer


class EngineSystemController:
    """
    Service class responsible for handling system‑level actions on
    ``UserQueryResponseAnswer`` instances, such as storing user feedback.
    """

    def __init__(self, store_to_db: bool = True):
        """
        Create a controller.

        Parameters
        ----------
        store_to_db : bool, default True
            When ``True`` the controller will persist changes to the
            database; when ``False`` the method will only modify the
            instance in memory (useful for tests).
        """
        self.store_to_db = store_to_db

    def set_rating(
        self,
        query_response_answer: UserQueryResponseAnswer,
        rating_value: int,
        rating_value_max: int,
        comment: str | None,
    ) -> bool:
        """
        Record a rating for a ``UserQueryResponseAnswer``.

        The method updates the ``rate_value``, ``rate_nax_value`` (maximum
        allowed rating) and optional ``rate_comment`` fields.  If the
        controller was instantiated with ``store_to_db=True`` the changes
        are saved to the database.

        Parameters
        ----------
        query_response_answer : UserQueryResponseAnswer
            The answer object to be rated.
        rating_value : int
            Numerical rating provided by the user.
        rating_value_max : int
            Upper bound of the rating scale.
        comment : str | None
            Optional free‑form comment accompanying the rating.

        Returns
        -------
        bool
            ``True`` if the operation succeeded.
        """
        query_response_answer.rate_value = rating_value
        query_response_answer.rate_nax_value = rating_value_max
        if comment is not None:
            query_response_answer.rate_comment = comment
        if self.store_to_db:
            query_response_answer.save()
        return True
