"""
Chat -> draft -> review/revise loop -> approve -> final.

Flow:
    START -> generate -> review -> (approve)  -> final -> END
                            ^          |
                            |     (feedback)
                            +---- revise <-------+

Same logic as before, just with the dead/duplicate `client` imports removed
(neither `http.client` nor `xmlrpc.client` was actually used — the code
calls `llm.invoke(...)`).
"""

from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from llm import llm


class ChatState(TypedDict):
    question: str
    draft: str
    feedback: Optional[str]
    status: Literal["drafting", "in_review", "approved"]


def generate_node(state: ChatState) -> dict:
    """First pass: ask the model to answer the question."""
    resp = llm.invoke([{"role": "user", "content": state["question"]}])
    return {"draft": resp.content, "status": "in_review"}


def review_node(state: ChatState) -> Command[Literal["revise", "final"]]:
    """Pause and show the draft to the human for approval or feedback."""
    result = interrupt(
        {
            "instruction": "Approve this answer, or send feedback to revise it.",
            "draft": state["draft"],
        }
    )
    # print(f"review_node: result={result}")
    # Expect result like {"decision": "approve"} or
    # {"decision": "revise", "feedback": "make it shorter"}
    if result.get("decision") == "approve":
        return Command(goto="final")
    return Command(goto="revise", update={"feedback": result.get("feedback", "")})


def revise_node(state: ChatState) -> dict:
    """Send the previous draft + human feedback back to the model."""
    resp = llm.invoke(
        [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": state["draft"]},
            {
                "role": "user",
                "content": f"Please revise your previous answer. Feedback: {state['feedback']}",
            },
        ]
    )
    return {"draft": resp.content, "status": "in_review"}


def final_node(state: ChatState) -> dict:
    return {"status": "approved"}


def build_graph_builder() -> StateGraph:
    """Returns an uncompiled builder so the caller can attach whatever
    checkpointer it wants (sync SqliteSaver for a script, AsyncSqliteSaver
    for FastAPI, etc.)."""
    builder = StateGraph(ChatState)
    builder.add_node("generate", generate_node)
    builder.add_node("review", review_node)
    builder.add_node("revise", revise_node)
    builder.add_node("final", final_node)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", "review")
    # review routes to "revise" or "final" itself via Command(goto=...)
    builder.add_edge("revise", "review")
    builder.add_edge("final", END)
    return builder