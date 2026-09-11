from google import genai
from google.genai import types

from app.config import settings
from app.schemas import ProjectTasksData


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
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        if self.client is None:
            self.client = genai.Client(api_key=settings.gemini_api_key)

        user_prompt = f"""Project Name:
{project_name}

Project Description:
{project_description}

Generate the implementation tasks for this project."""

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

        parsed = response.parsed
        if not isinstance(parsed, ProjectTasksData):
            parsed = ProjectTasksData.model_validate(parsed)

        return parsed.model_copy(update={"projectName": project_name})


gemini_service = GeminiService()
