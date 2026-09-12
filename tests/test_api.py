import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AiErrorCode, ProjectTasksData, Task
from app.services.ai_error import AiServiceError


client = TestClient(app)

PROJECT_NAME = "E-commerce Website"
PROJECT_DESCRIPTION = (
    "Build an e-commerce application where users can browse products, "
    "add products to a cart, and place orders."
)
VALID_REQUEST = {
    "projectName": PROJECT_NAME,
    "projectDescription": PROJECT_DESCRIPTION,
}


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": True,
        "messages": "Service is healthy.",
        "data": {},
    }


def test_generate_tasks_success() -> None:
    generated_tasks = ProjectTasksData(
        projectName=PROJECT_NAME,
        tasks=[
            Task(
                taskName="Implement Product Catalog",
                taskDescription="Create the product browsing functionality.",
            ),
            Task(
                taskName="Implement Shopping Cart",
                taskDescription=(
                    "Allow users to add and remove products from the cart."
                ),
            ),
        ],
    )

    with patch(
        "app.main.gemini_service.generate_tasks",
        return_value=generated_tasks,
    ) as mock_generate_tasks:
        response = client.post("/api/v1/generate-tasks", json=VALID_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] is True
    assert body["messages"] == "Project tasks generated successfully."
    assert body["data"]["projectName"] == PROJECT_NAME
    assert len(body["data"]["tasks"]) == 2
    assert set(body) == {"status", "messages", "data"}
    assert set(body["data"]) == {"projectName", "tasks"}
    for task in body["data"]["tasks"]:
        assert set(task) == {"taskName", "taskDescription"}

    mock_generate_tasks.assert_called_once_with(
        project_name=PROJECT_NAME,
        project_description=PROJECT_DESCRIPTION,
    )


def test_missing_project_name_is_rejected() -> None:
    response = client.post(
        "/api/v1/generate-tasks",
        json={"projectDescription": "Some project description."},
    )

    assert response.status_code == 422


def test_missing_project_description_is_rejected() -> None:
    response = client.post(
        "/api/v1/generate-tasks",
        json={"projectName": "Test Project"},
    )

    assert response.status_code == 422


def test_empty_project_name_is_rejected() -> None:
    response = client.post(
        "/api/v1/generate-tasks",
        json={
            "projectName": "",
            "projectDescription": "Some description.",
        },
    )

    assert response.status_code == 422


def test_empty_project_description_is_rejected() -> None:
    response = client.post(
        "/api/v1/generate-tasks",
        json={
            "projectName": "Test Project",
            "projectDescription": "",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("service_error", "expected_status", "expected_message"),
    [
        (
            AiServiceError(
                AiErrorCode.QUOTA_EXCEEDED,
                "AI quota has been exhausted.",
                429,
                True,
            ),
            429,
            "AI quota has been exhausted.",
        ),
        (
            AiServiceError(
                AiErrorCode.RATE_LIMITED,
                "AI provider rate limit reached.",
                429,
                True,
            ),
            429,
            "AI provider rate limit reached.",
        ),
        (
            AiServiceError(
                AiErrorCode.TIMEOUT,
                "AI task generation timed out.",
                504,
                True,
            ),
            504,
            "AI task generation timed out.",
        ),
        (
            AiServiceError(
                AiErrorCode.PROVIDER_UNAVAILABLE,
                "AI task generation is temporarily unavailable.",
                503,
                True,
            ),
            503,
            "AI task generation is temporarily unavailable.",
        ),
        (
            AiServiceError(
                AiErrorCode.AUTHENTICATION_FAILED,
                "AI provider authentication failed.",
                503,
                False,
            ),
            503,
            "AI provider authentication failed.",
        ),
        (
            AiServiceError(
                AiErrorCode.INVALID_RESPONSE,
                "AI provider returned an invalid response.",
                502,
                False,
            ),
            502,
            "AI provider returned an invalid response.",
        ),
        (
            AiServiceError(
                AiErrorCode.CONFIGURATION_ERROR,
                "AI task generation is not configured.",
                503,
                False,
            ),
            503,
            "AI task generation is not configured.",
        ),
    ],
)
def test_expected_ai_failure_returns_stable_safe_response(
    service_error: AiServiceError,
    expected_status: int,
    expected_message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="app.main"),
        patch(
            "app.main.gemini_service.generate_tasks",
            side_effect=service_error,
        ),
    ):
        response = client.post("/api/v1/generate-tasks", json=VALID_REQUEST)

    assert response.status_code == expected_status
    assert response.json() == {
        "status": False,
        "messages": expected_message,
        "data": {},
        "error": {
            "code": service_error.code.value,
            "retryable": service_error.retryable,
        },
    }
    assert "traceback" not in response.text.lower()
    assert "GEMINI_API_KEY" not in response.text
    assert "AI task generation failed." in caplog.text


def test_unexpected_failure_returns_internal_error_without_exception_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_provider_message = "provider-secret-body"

    with (
        caplog.at_level(logging.ERROR, logger="app.main"),
        patch(
            "app.main.gemini_service.generate_tasks",
            side_effect=RuntimeError(secret_provider_message),
        ),
    ):
        response = client.post("/api/v1/generate-tasks", json=VALID_REQUEST)

    assert response.status_code == 500
    assert response.json() == {
        "status": False,
        "messages": "AI task generation failed unexpectedly.",
        "data": {},
        "error": {
            "code": "AI_INTERNAL_ERROR",
            "retryable": False,
        },
    }
    assert secret_provider_message not in response.text
    assert secret_provider_message not in caplog.text
    assert "Traceback" not in caplog.text
