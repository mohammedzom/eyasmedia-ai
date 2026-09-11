# Eyas Project Management Assistant

A FastAPI service that uses Gemini 2.5 Flash to generate structured implementation tasks for software projects.

## Run the API

```bash
uvicorn app.main:app --reload
```

## Run tests

```bash
pytest
```

## Run with Docker

```bash
docker build -t eyas-project-management-assistant .
docker run --rm -p 8000:8000 --env-file .env eyas-project-management-assistant
```

## Run with Docker Compose

```bash
docker compose up --build
```

Stop the service with:

```bash
docker compose down
```

## Environment variables

```env
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

## Main endpoint

```http
POST /api/v1/generate-tasks
```
