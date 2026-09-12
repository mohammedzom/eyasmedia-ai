from enum import StrEnum

from pydantic import BaseModel, Field


class AiErrorCode(StrEnum):
    QUOTA_EXCEEDED = "AI_QUOTA_EXCEEDED"
    RATE_LIMITED = "AI_RATE_LIMITED"
    AUTHENTICATION_FAILED = "AI_AUTHENTICATION_FAILED"
    PROVIDER_UNAVAILABLE = "AI_PROVIDER_UNAVAILABLE"
    TIMEOUT = "AI_TIMEOUT"
    INVALID_RESPONSE = "AI_INVALID_RESPONSE"
    CONFIGURATION_ERROR = "AI_CONFIGURATION_ERROR"
    INTERNAL_ERROR = "AI_INTERNAL_ERROR"


class ProjectRequest(BaseModel):
    projectName: str = Field(min_length=1)
    projectDescription: str = Field(min_length=1)


class Task(BaseModel):
    taskName: str
    taskDescription: str


class ProjectTasksData(BaseModel):
    projectName: str
    tasks: list[Task]


class AiError(BaseModel):
    code: AiErrorCode
    retryable: bool


class ProjectTasksResponse(BaseModel):
    status: bool
    messages: str
    data: ProjectTasksData | dict
    error: AiError | None = None
