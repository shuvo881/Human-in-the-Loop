from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Dict, Literal, Optional, TypedDict
from langchain.messages import AIMessage, AnyMessage, HumanMessage
from typing_extensions import Annotated
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pathlib import Path

from llm import llm

def file_paths_reducer(left: list[str], right: list[str]) -> list[str]:
    """Validate new file paths and merge them with the existing state."""
    valid_paths = []
    for path in right:
        p = Path(path)
        if p.exists() and p.is_file():
            valid_paths.append(str(p))
    return left + valid_paths if valid_paths else left

class FileInfo(TypedDict):
    path: Annotated[str, file_paths_reducer]
    file_type: Optional[str]
    company_name: Optional[str]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    status: Literal["draft", "review", "approved", "rejected"]
    file_paths: Annotated[list[str], file_paths_reducer]
    related_files_info: list[FileInfo]
    features: Optional[dict]



def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "status": "draft", "file_paths": ['test2.txt'], "related_files_info": [{"path": "test2.txt", "file_type": "text", "company_name": "Example Corp"}]}

builder = StateGraph(State).add_node(node).add_edge(START, "node")
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))
# store = SqliteSaver(sqlite3.connect("store.db", check_same_thread=False))
graph = builder.compile(checkpointer=checkpointer,)


config = {"configurable": {"thread_id": '12345', }}
result = graph.invoke({"messages": [HumanMessage("Hi")], "file_paths": ["./test.txt"], "related_files_info": [{"path": "./related.txt"}]}, config=config)

print("Result:", result)

for message in result["messages"]:
    message.pretty_print()

print(graph.get_state(config))