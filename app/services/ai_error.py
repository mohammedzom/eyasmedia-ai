from app.schemas import AiErrorCode


class AiServiceError(Exception):
    def __init__(
        self,
        code: AiErrorCode,
        message: str,
        http_status: int,
        retryable: bool,
    ) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        super().__init__(message)
