# React Agent from scratch

## Overview
This project demonstrates a conversational agent workflow built using LangChain and LangGraph. The agent leverages an external tool to retrieve weather data from wttr.in and integrates a state graph to manage tool calls and agent responses in a loop until the conversation ends.

        +-----------+         
        | __start__ |         
        +-----------+         
              *               
              *               
              *               
          +-------+           
          | agent |           
          +-------+           
         *         .          
       **           ..        
      *               .       
+-------+         +---------+ 
| tools |         | __end__ | 
+-------+         +---------+ 

## Key Features
- **Agent Integration:** Utilizes the ChatOpenAI model (gpt-3.5-turbo-1106) bound with custom tool functions.
- **Weather Retrieval Tool:** Provides a `get_weather` tool that fetches weather information based on a location input.
- **State Graph Workflow:** Implements a state graph using LangGraph's `StateGraph` to manage interactions between the agent and tool calls.
- **Conditional Flow:** The workflow conditionally routes responses—if the agent's message includes tool calls, it transitions to execute those tools, then resumes the agent conversation.

## Project Structure
- **Tool Definition:**
  - `get_weather(location: str)`: A tool function that calls the wttr.in API to fetch weather info. Returns a weather string if successful or an error message if not.
  
- **Agent Setup:**
  - The `ChatOpenAI` model from LangChain is instantiated with a specified model and temperature.
  - The model is then bound with the `get_weather` tool using the tool binding mechanism.

- **Workflow Functions:**
  - `tool_node(state: AgentState)`: Processes tool calls from the latest message, invokes the corresponding tool(s), and wraps the result in a `ToolMessage`.
  - `call_model(state: AgentState, config: RunnableConfig)`: Combines a system prompt with the conversation history and invokes the chat model, returning the model's response.
  - `should_continue(state: AgentState)`: Examines the latest message to decide whether to route control to the tool node (if there are tool calls) or to end the workflow.

- **Workflow Graph:**
  - A state graph is built using `StateGraph` with nodes for the agent and tool functions.
  - The entry point is set to the agent node.
  - Conditional edges direct the flow: if there are tool calls, the graph transitions to the tools node; otherwise, it concludes the conversation.
  - After executing the tool, control flows back to the agent for further processing if needed.

## How to Run
1. **Setup Dependencies:**  
   Ensure you have the required libraries installed (e.g., LangChain, LangGraph, requests). You can install them via pip if necessary.
   
2. **Execute the Script:**  
   Run the Python script. The script compiles the workflow graph, prints an ASCII representation of the state graph, and executes a sample query by sending an input message such as: