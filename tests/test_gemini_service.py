from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import settings
from app.schemas import ProjectTasksData
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
