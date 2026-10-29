import uvicorn
from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logger import log

settings = get_settings()

app = FastAPI(
    title="ProspectIQ",
    description="AI Sales Intelligence & Outreach Agent API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "ProspectIQ"}


if __name__ == "__main__":
    log.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
