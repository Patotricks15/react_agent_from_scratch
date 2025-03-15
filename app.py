from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage, add_messages]]

