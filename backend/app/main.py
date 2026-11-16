from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logger import log
from app.llm import (
    LLMRequest,
    LLMTimeoutError,
    call_llm,
    close_groq_client,
    get_groq_client,
)
from app.research import ResearchRequest, ResearchResponse, conduct_research

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_groq_client()
    yield
    await close_groq_client()


app = FastAPI(
    title="ProspectIQ",
    description="AI Sales Intelligence & Outreach Agent API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(LLMTimeoutError)
async def llm_timeout_handler(_request: Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": str(exc)})


api_router = APIRouter(prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "ProspectIQ"}


@api_router.post("/chat")
async def chat(request: LLMRequest):
    return await call_llm(request)


@api_router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    return await conduct_research(request)


app.include_router(api_router)


if __name__ == "__main__":
    log.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
