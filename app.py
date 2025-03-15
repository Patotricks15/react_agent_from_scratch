from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
import requests
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_weather(location:str):

    """
    Gets the weather for a given location.

    Args:
        location (str): The city or zip code of the location

    Returns:
        str: The weather for the location
    """
    url = f"http://wttr.in/{location}?format=3"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return f"Não consegui obter o clima para {location}."
    

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
tools = [get_weather]
model = model.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(
    state: AgentState,
    config: RunnableConfig,
):

    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    return {"messages": [response]}

def should_continue(state: AgentState):

    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)


workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", should_continue, {'continue':'tools', 'end':END})

workflow.add_edge("tools", "agent")

graph = workflow.compile()


inputs = {"messages": [("user", "what is the weather in Inhoaíba RJ?")]}
print(graph.invoke(inputs)['messages'][-1].content)