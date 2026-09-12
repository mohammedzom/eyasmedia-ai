import httpx
from google import genai
from google.genai import errors
from google.genai import types
from pydantic import ValidationError

from app.config import settings
from app.schemas import AiErrorCode, ProjectTasksData
from app.services.ai_error import AiServiceError


SYSTEM_PROMPT = """You are a senior software engineer and project manager.

Analyze the provided software project and break it into clear, actionable implementation tasks.

Rules:
- Generate only tasks required to implement the project.
- Keep each task focused on one meaningful unit of work.
- Use concise task names.
- Write clear task descriptions explaining what should be implemented.
- Avoid vague tasks.
- Avoid unnecessary architecture and over-engineering.
- Return tasks in a logical implementation order.
- Preserve the provided project name.
"""


class GeminiService:
    def __init__(self) -> None:
        self.client: genai.Client | None = None

    def generate_tasks(
        self,
        project_name: str,
        project_description: str,
    ) -> ProjectTasksData:
        if not settings.gemini_api_key:
            raise AiServiceError(
                code=AiErrorCode.CONFIGURATION_ERROR,
                message="AI task generation is not configured.",
                http_status=503,
                retryable=False,
            )

        if self.client is None:
            self.client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=settings.gemini_timeout_seconds * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )

        user_prompt = f"""Project Name:
{project_name}

Project Description:
{project_description}

Generate the implementation tasks for this project."""

        try:
            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=ProjectTasksData,
                ),
            )

            parsed = getattr(response, "parsed", None)
            if not isinstance(parsed, ProjectTasksData):
                parsed = ProjectTasksData.model_validate(parsed)
        except errors.ClientError as exception:
            raise self._map_client_error(exception) from None
        except errors.ServerError:
            raise AiServiceError(
                code=AiErrorCode.PROVIDER_UNAVAILABLE,
                message="AI task generation is temporarily unavailable.",
                http_status=503,
                retryable=True,
            ) from None
        except httpx.TimeoutException:
            raise AiServiceError(
                code=AiErrorCode.TIMEOUT,
                message="AI task generation timed out.",
                http_status=504,
                retryable=True,
            ) from None
        except httpx.RequestError:
            raise AiServiceError(
                code=AiErrorCode.PROVIDER_UNAVAILABLE,
                message="AI task generation is temporarily unavailable.",
                http_status=503,
                retryable=True,
            ) from None
        except (ValidationError, errors.UnknownApiResponseError):
            raise AiServiceError(
                code=AiErrorCode.INVALID_RESPONSE,
                message="AI provider returned an invalid response.",
                http_status=502,
                retryable=False,
            ) from None

        return parsed.model_copy(update={"projectName": project_name})

    def _map_client_error(self, exception: errors.ClientError) -> AiServiceError:
        if exception.code == 429:
            if self._contains_quota_failure(exception.details):
                return AiServiceError(
                    code=AiErrorCode.QUOTA_EXCEEDED,
                    message="AI quota has been exhausted.",
                    http_status=429,
                    retryable=True,
                )

            return AiServiceError(
                code=AiErrorCode.RATE_LIMITED,
                message="AI provider rate limit reached.",
                http_status=429,
                retryable=True,
            )

        if exception.code in {401, 403} or self._contains_error_reason(
            exception.details,
            {"API_KEY_INVALID", "UNAUTHENTICATED"},
        ):
            return AiServiceError(
                code=AiErrorCode.AUTHENTICATION_FAILED,
                message="AI provider authentication failed.",
                http_status=503,
                retryable=False,
            )

        return AiServiceError(
            code=AiErrorCode.CONFIGURATION_ERROR,
            message="AI task generation is not configured correctly.",
            http_status=503,
            retryable=False,
        )

    def _contains_quota_failure(self, value: object) -> bool:
        if isinstance(value, dict):
            if str(value.get("@type", "")).endswith("google.rpc.QuotaFailure"):
                return True

            if "quotaId" in value:
                return True

            return any(self._contains_quota_failure(item) for item in value.values())

        if isinstance(value, list):
            return any(self._contains_quota_failure(item) for item in value)

        return False

    def _contains_error_reason(
        self,
        value: object,
        reasons: set[str],
    ) -> bool:
        if isinstance(value, dict):
            if value.get("reason") in reasons or value.get("status") in reasons:
                return True

            return any(
                self._contains_error_reason(item, reasons)
                for item in value.values()
            )

        if isinstance(value, list):
            return any(self._contains_error_reason(item, reasons) for item in value)

        return False


gemini_service = GeminiService()
