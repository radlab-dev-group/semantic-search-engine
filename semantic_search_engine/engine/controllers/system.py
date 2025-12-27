from engine.models import UserQueryResponseAnswer


class EngineSystemController:
    def __init__(self, store_to_db: bool = True):
        self.store_to_db = store_to_db

    def set_rating(
        self,
        query_response_answer: UserQueryResponseAnswer,
        rating_value: int,
        rating_value_max: int,
        comment: str | None,
    ) -> bool:
        query_response_answer.rate_value = rating_value
        query_response_answer.rate_nax_value = rating_value_max
        if comment is not None:
            query_response_answer.rate_comment = comment
        if self.store_to_db:
            query_response_answer.save()
        return True
