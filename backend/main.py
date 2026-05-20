"""AURA FastAPI backend with RAG-powered context streaming."""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.rag_chain import build_rag_chain, run_rag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global RAG components (initialized at startup)
_retriever = None
_llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _llm
    log.info("Initializing RAG pipeline...")
    _retriever, _llm = build_rag_chain()
    log.info("RAG pipeline ready.")
    yield
    log.info("Shutting down AURA backend.")


app = FastAPI(
    title="AURA API",
    description="Audio Understanding & RAG for Auto — context-aware vehicle control",
    version="1.0.0",
    lifespan=lifespan,
)


class ContextRequest(BaseModel):
    audio_event: str = Field(..., example="yawning")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.9)
    driving_hours: float = Field(..., ge=0.0, example=2.5)
    hvac_status: str = Field(..., example="internal")


class ContextResponse(BaseModel):
    audio_event: str
    confidence: float
    query: str
    retrieved_context: list[str]
    advice: str
    command_json: str
    raw_llm_response: str


@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": _retriever is not None}


@app.post("/api/v1/context-stream", response_model=ContextResponse)
async def context_stream(req: ContextRequest):
    result = run_rag(
        retriever=_retriever,
        llm=_llm,
        audio_event=req.audio_event,
        confidence=req.confidence,
        driving_hours=req.driving_hours,
        hvac_status=req.hvac_status,
    )

    raw = result["llm_response"]
    # Parse ADVICE / COMMAND from response
    advice = raw
    command_json = "{}"
    if "COMMAND:" in raw:
        parts = raw.split("COMMAND:", 1)
        advice = parts[0].replace("ADVICE:", "").strip()
        command_json = parts[1].strip()
    elif "ADVICE:" in raw:
        advice = raw.replace("ADVICE:", "").strip()

    return ContextResponse(
        audio_event=req.audio_event,
        confidence=req.confidence,
        query=result["query"],
        retrieved_context=result["retrieved_context"],
        advice=advice,
        command_json=command_json,
        raw_llm_response=raw,
    )
