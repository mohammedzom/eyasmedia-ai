import logging
from time import monotonic

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.schemas import AiError, AiErrorCode, ProjectRequest, ProjectTasksResponse
from app.services.ai_error import AiServiceError
from app.services.gemini_service import gemini_service


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Eyas Project Management Assistant",
    description="Generate project implementation tasks using AI.",
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=ProjectTasksResponse,
    response_model_exclude_none=True,
)
def health() -> ProjectTasksResponse:
    return ProjectTasksResponse(
        status=True,
        messages="Service is healthy.",
        data={},
    )


@app.post(
    "/api/v1/generate-tasks",
    response_model=ProjectTasksResponse,
    response_model_exclude_none=True,
)
def generate_tasks(
    project: ProjectRequest,
) -> ProjectTasksResponse | JSONResponse:
    started_at = monotonic()

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
    except AiServiceError as exception:
        logger.warning(
            "AI task generation failed.",
            extra={
                "error_code": exception.code.value,
                "http_status": exception.http_status,
                "provider_operation": "generate_tasks",
                "duration_ms": round((monotonic() - started_at) * 1000),
            },
        )
        return JSONResponse(
            status_code=exception.http_status,
            content=ProjectTasksResponse(
                status=False,
                messages=exception.message,
                data={},
                error=AiError(
                    code=exception.code,
                    retryable=exception.retryable,
                ),
            ).model_dump(mode="json", exclude_none=True),
        )
    except Exception as exception:
        logger.error(
            "Unexpected AI task generation failure.",
            extra={
                "error_code": AiErrorCode.INTERNAL_ERROR.value,
                "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "provider_operation": "generate_tasks",
                "exception_type": type(exception).__name__,
                "duration_ms": round((monotonic() - started_at) * 1000),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ProjectTasksResponse(
                status=False,
                messages="AI task generation failed unexpectedly.",
                data={},
                error=AiError(
                    code=AiErrorCode.INTERNAL_ERROR,
                    retryable=False,
                ),
            ).model_dump(mode="json", exclude_none=True),
        )
