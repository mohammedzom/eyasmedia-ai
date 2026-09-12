from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import errors

from app.config import settings
from app.schemas import AiErrorCode, ProjectTasksData
from app.services.ai_error import AiServiceError
from app.services.gemini_service import GeminiService


def test_gemini_service_uses_structured_output() -> None:
    service = GeminiService()
    service.client = MagicMock()
    service.client.models.generate_content.return_value = SimpleNamespace(
        parsed={
            "projectName": "Test Project",
            "tasks": [
                {
                    "taskName": "Build API",
                    "taskDescription": "Implement the project API.",
                }
            ],
        }
    )

    with patch.object(settings, "gemini_api_key", "offline-test-key"):
        result = service.generate_tasks(
            project_name="Test Project",
            project_description="Build a test project.",
        )

    request = service.client.models.generate_content.call_args.kwargs
    config = request["config"]

    assert request["model"] == "gemini-2.5-flash"
    assert config.temperature == 0.2
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ProjectTasksData
    assert config.system_instruction
    assert result == ProjectTasksData.model_validate(
        service.client.models.generate_content.return_value.parsed
    )


def test_gemini_service_configures_one_bounded_provider_attempt() -> None:
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={"projectName": "Test Project", "tasks": []}
    )

    with (
        patch.object(settings, "gemini_api_key", "offline-test-key"),
        patch.object(settings, "gemini_timeout_seconds", 25),
        patch("app.services.gemini_service.genai.Client", return_value=client) as client_class,
    ):
        GeminiService().generate_tasks("Test Project", "Build a test project.")

    http_options = client_class.call_args.kwargs["http_options"]
    assert http_options.timeout == 25_000
    assert http_options.retry_options.attempts == 1


def test_missing_api_key_maps_to_configuration_error() -> None:
    with patch.object(settings, "gemini_api_key", ""):
        with pytest.raises(AiServiceError) as raised:
            GeminiService().generate_tasks("Test Project", "Build a test project.")

    assert raised.value.code is AiErrorCode.CONFIGURATION_ERROR
    assert raised.value.http_status == 503
    assert raised.value.retryable is False


def test_quota_failure_maps_from_structured_provider_details() -> None:
    provider_error = errors.ClientError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerDay"}],
                    }
                ],
            }
        },
    )

    mapped = generate_with_error(provider_error)

    assert mapped.code is AiErrorCode.QUOTA_EXCEEDED
    assert mapped.http_status == 429
    assert mapped.retryable is True


def test_generic_429_maps_to_rate_limited() -> None:
    provider_error = errors.ClientError(
        429,
        {"error": {"status": "RESOURCE_EXHAUSTED"}},
    )

    mapped = generate_with_error(provider_error)

    assert mapped.code is AiErrorCode.RATE_LIMITED
    assert mapped.http_status == 429
    assert mapped.retryable is True


@pytest.mark.parametrize(
    "provider_error",
    [
        errors.ClientError(401, {"error": {"status": "UNAUTHENTICATED"}}),
        errors.ClientError(
            400,
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
        ),
    ],
)
def test_authentication_failure_maps_from_status_or_structured_reason(
    provider_error: errors.ClientError,
) -> None:
    mapped = generate_with_error(provider_error)

    assert mapped.code is AiErrorCode.AUTHENTICATION_FAILED
    assert mapped.http_status == 503
    assert mapped.retryable is False


def test_other_provider_client_error_maps_to_configuration_error() -> None:
    mapped = generate_with_error(
        errors.ClientError(400, {"error": {"status": "INVALID_ARGUMENT"}})
    )

    assert mapped.code is AiErrorCode.CONFIGURATION_ERROR
    assert mapped.http_status == 503
    assert mapped.retryable is False


def test_provider_server_error_maps_to_unavailable() -> None:
    mapped = generate_with_error(
        errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}})
    )

    assert mapped.code is AiErrorCode.PROVIDER_UNAVAILABLE
    assert mapped.http_status == 503
    assert mapped.retryable is True


def test_provider_timeout_maps_to_timeout() -> None:
    mapped = generate_with_error(httpx.ReadTimeout("provider timed out"))

    assert mapped.code is AiErrorCode.TIMEOUT
    assert mapped.http_status == 504
    assert mapped.retryable is True


def test_provider_connection_failure_maps_to_unavailable() -> None:
    mapped = generate_with_error(httpx.ConnectError("provider refused connection"))

    assert mapped.code is AiErrorCode.PROVIDER_UNAVAILABLE
    assert mapped.http_status == 503
    assert mapped.retryable is True


def test_invalid_structured_response_maps_to_invalid_response() -> None:
    service = configured_service()
    service.client.models.generate_content.return_value = SimpleNamespace(
        parsed={"projectName": "Test Project", "tasks": [{"unexpected": "value"}]}
    )

    with (
        patch.object(settings, "gemini_api_key", "offline-test-key"),
        pytest.raises(AiServiceError) as raised,
    ):
        service.generate_tasks("Test Project", "Build a test project.")

    assert raised.value.code is AiErrorCode.INVALID_RESPONSE
    assert raised.value.http_status == 502
    assert raised.value.retryable is False


def test_missing_parsed_response_maps_to_invalid_response() -> None:
    service = configured_service()
    service.client.models.generate_content.return_value = SimpleNamespace()

    with (
        patch.object(settings, "gemini_api_key", "offline-test-key"),
        pytest.raises(AiServiceError) as raised,
    ):
        service.generate_tasks("Test Project", "Build a test project.")

    assert raised.value.code is AiErrorCode.INVALID_RESPONSE
    assert raised.value.http_status == 502


def configured_service() -> GeminiService:
    service = GeminiService()
    service.client = MagicMock()
    return service


def generate_with_error(provider_error: Exception) -> AiServiceError:
    service = configured_service()
    service.client.models.generate_content.side_effect = provider_error

    with (
        patch.object(settings, "gemini_api_key", "offline-test-key"),
        pytest.raises(AiServiceError) as raised,
    ):
        service.generate_tasks("Test Project", "Build a test project.")

    return raised.value
