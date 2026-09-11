from pydantic import BaseModel, Field


class ProjectRequest(BaseModel):
    projectName: str = Field(min_length=1)
    projectDescription: str = Field(min_length=1)


class Task(BaseModel):
    taskName: str
    taskDescription: str


class ProjectTasksData(BaseModel):
    projectName: str
    tasks: list[Task]


class ProjectTasksResponse(BaseModel):
    status: bool
    messages: str
    data: ProjectTasksData | dict
