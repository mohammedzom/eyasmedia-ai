import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ProjectTasksData, Task


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


def test_gemini_failure_returns_safe_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.ERROR, logger="app.main"),
        patch(
            "app.main.gemini_service.generate_tasks",
            side_effect=RuntimeError("Gemini unavailable"),
        ),
    ):
        response = client.post("/api/v1/generate-tasks", json=VALID_REQUEST)

    assert response.status_code == 502
    assert response.json() == {
        "status": False,
        "messages": "Failed to generate project tasks.",
        "data": {},
    }
    assert "Gemini unavailable" not in response.text
    assert "traceback" not in response.text.lower()
    assert "GEMINI_API_KEY" not in response.text
    assert "Gemini task generation failed." in caplog.text
