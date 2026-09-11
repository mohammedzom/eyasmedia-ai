import logging

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.schemas import ProjectRequest, ProjectTasksResponse
from app.services.gemini_service import gemini_service


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Eyas Project Management Assistant",
    description="Generate project implementation tasks using AI.",
    version="1.0.0",
)


@app.get("/health", response_model=ProjectTasksResponse)
def health() -> ProjectTasksResponse:
    return ProjectTasksResponse(
        status=True,
        messages="Service is healthy.",
        data={},
    )


@app.post("/api/v1/generate-tasks", response_model=ProjectTasksResponse)
def generate_tasks(
    project: ProjectRequest,
) -> ProjectTasksResponse | JSONResponse:
    try:
        project_tasks = gemini_service.generate_tasks(
            project_name=project.projectName,
            project_description=project.projectDescription,
        )
        return ProjectTasksResponse(
            status=True,
            messages="Project tasks generated successfully.",
            data=project_tasks,
        )
    except Exception:
        logger.exception("Gemini task generation failed.")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ProjectTasksResponse(
                status=False,
                messages="Failed to generate project tasks.",
                data={},
            ).model_dump(),
        )
