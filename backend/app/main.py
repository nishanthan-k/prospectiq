from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent import (
    AgentInvestigateRequest,
    AgentInvestigateResponse,
    AgentQualifyRequest,
    AgentQualifyResponse,
    AgentRunRequest,
    AgentRunResponse,
    investigate_company,
    qualify_company,
    run_sales_agent,
)
from app.core.config import get_settings
from app.core.logger import log
from app.llm import (
    LLMRequest,
    LLMTimeoutError,
    call_llm,
    close_gemini_client,
    get_gemini_client,
)
from app.research import ResearchRequest, ResearchResponse, conduct_research

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_gemini_client()
    yield
    await close_gemini_client()


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


@api_router.post("/agent/investigate", response_model=AgentInvestigateResponse)
async def agent_investigate(request: AgentInvestigateRequest):
    return await investigate_company(request)


@api_router.post("/agent/qualify", response_model=AgentQualifyResponse)
async def agent_qualify(request: AgentQualifyRequest):
    return await qualify_company(request)


@api_router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(request: AgentRunRequest):
    return await run_sales_agent(request)


app.include_router(api_router)


if __name__ == "__main__":
    log.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
