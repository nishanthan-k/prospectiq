import uvicorn
from fastapi import FastAPI, HTTPException

from app.core.config import get_settings
from app.core.logger import log

from app.llm.client import LLMRequest, LLMResponse, LLMTimeoutError, call_llm

settings = get_settings()

app = FastAPI(
    title="ProspectIQ",
    description="AI Sales Intelligence & Outreach Agent API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "ProspectIQ"}


@app.post("/chat", response_model=LLMResponse)
async def chat(request: LLMRequest):
    try:
        return await call_llm(request)
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc


if __name__ == "__main__":
    log.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
