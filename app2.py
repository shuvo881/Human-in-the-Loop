"""
FastAPI wrapper around the chat/review/revise/final graph.

Single endpoint:
    POST /chat

    thread_id is always required, supplied by the caller (not generated
    here). Whether it's treated as a new conversation or a resume depends
    on whether that thread_id already has state:

    - Unknown thread_id -> starts a new conversation (question required).
    - Existing, paused thread_id -> resumes it (decision required).

Install:
    pip install fastapi uvicorn langgraph langgraph-checkpoint-sqlite

Run:
    uvicorn app:app --reload

Example:
    curl -X POST localhost:8000/chat \\
         -d '{"thread_id": "my-thread-1", "question": "Explain recursion"}' \\
         -H 'content-type: application/json'
    # -> {"thread_id": "my-thread-1", "draft": "...", "done": false}

    curl -X POST localhost:8000/chat \\
         -d '{"thread_id": "my-thread-1", "decision": "revise", "feedback": "shorter please"}' \\
         -H 'content-type: application/json'
    # -> {"thread_id": "my-thread-1", "draft": "...", "done": false}

    curl -X POST localhost:8000/chat \\
         -d '{"thread_id": "my-thread-1", "decision": "approve"}' \\
         -H 'content-type: application/json'
    # -> {"thread_id": "my-thread-1", "draft": null, "done": true, "status": "approved"}
"""

from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, model_validator

from workflow import build_graph_builder

graph = None  # set during lifespan startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    async with AsyncSqliteSaver.from_conn_string("chat.db") as checkpointer:
        graph = build_graph_builder().compile(checkpointer=checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str
    # required to start a new conversation
    question: Optional[str] = None
    # required to resume an existing one
    decision: Optional[Literal["approve", "revise"]] = None
    feedback: Optional[str] = None

    @model_validator(mode="after")
    def _check_required_fields(self):
        if not self.question and self.decision is None:
            raise ValueError("provide `question` (new conversation) or `decision` (resume)")
        return self


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
async def chat(body: ChatRequest):
    thread_id = body.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    # print(f"snapshot for thread_id={thread_id}: {snapshot}")
    is_new = snapshot is None or not snapshot.values

    # --- start a new conversation on this thread_id ---
    if is_new:
        if not body.question:
            raise HTTPException(
                status_code=400,
                detail="`question` is required to start a new conversation on this thread_id",
            )
        state = await graph.ainvoke(
            {"question": body.question, "draft": "", "feedback": None, "status": "drafting"},
            config=config,
        )
        return _to_response(thread_id, state)

    # --- resume an existing, paused conversation ---
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="This thread_id has already reached a final state")
    if body.decision is None:
        raise HTTPException(
            status_code=400,
            detail="`decision` is required to resume an in-progress thread_id",
        )

    resume_value = {"decision": body.decision}
    if body.decision == "revise":
        resume_value["feedback"] = body.feedback or ""

    state = await graph.ainvoke(Command(resume=resume_value), config=config)
    return _to_response(thread_id, state)