"""
FastAPI wrapper around the chat/review/revise/final graph.

Endpoints:
    POST /chat             start a new conversation -> {thread_id, draft, done}
    POST /chat/{thread_id} resume it with an approve/revise decision

Install:
    pip install fastapi uvicorn langgraph langgraph-checkpoint-sqlite

Run:
    uvicorn app:app --reload

Example:
    curl -X POST localhost:8000/chat -d '{"question": "Explain recursion"}' \\
         -H 'content-type: application/json'
    # -> {"thread_id": "chat-ab12cd34", "draft": "...", "done": false}

    curl -X POST localhost:8000/chat/chat-ab12cd34 \\
         -d '{"decision": "revise", "feedback": "shorter please"}' \\
         -H 'content-type: application/json'
    # -> {"thread_id": "chat-ab12cd34", "draft": "...", "done": false}

    curl -X POST localhost:8000/chat/chat-ab12cd34 \\
         -d '{"decision": "approve"}' -H 'content-type: application/json'
    # -> {"thread_id": "chat-ab12cd34", "draft": null, "done": true, "status": "approved"}
"""

import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from workflow import build_graph_builder

graph = None  # set during lifespan startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    async with AsyncSqliteSaver.from_conn_string("chat.db") as checkpointer:
        graph = build_graph_builder().compile(checkpointer=checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


class StartRequest(BaseModel):
    question: str


class ReviewRequest(BaseModel):
    decision: Literal["approve", "revise"]
    feedback: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    draft: Optional[str] = None
    done: bool
    status: Optional[str] = None


def _to_response(thread_id: str, state: dict) -> ChatResponse:
    if "__interrupt__" in state:
        draft = state["__interrupt__"][0].value["draft"]
        return ChatResponse(thread_id=thread_id, draft=draft, done=False)
    return ChatResponse(thread_id=thread_id, done=True, status=state.get("status"))


@app.post("/chat", response_model=ChatResponse)
async def start_chat(body: StartRequest):
    thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    state = await graph.ainvoke(
        {"question": body.question, "draft": "", "feedback": None, "status": "drafting"},
        config=config,
    )
    return _to_response(thread_id, state)


@app.post("/chat/{thread_id}", response_model=ChatResponse)
async def review_chat(thread_id: str, body: ReviewRequest):
    config = {"configurable": {"thread_id": thread_id}}

    # Make sure this thread actually exists / is paused before resuming.
    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.next:
        raise HTTPException(status_code=404, detail="No pending review for this thread_id")

    resume_value = {"decision": body.decision}
    if body.decision == "revise":
        resume_value["feedback"] = body.feedback or ""

    state = await graph.ainvoke(Command(resume=resume_value), config=config)
    return _to_response(thread_id, state)