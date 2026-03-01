from pydantic import BaseModel
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def make_default_graph():
    builder = StateGraph(State)
    
    def call_model(State: State):
        return {"messages": [model.invoke(State.messages)]}

    # add node
    builder.add_node("agent", call_model)

    # add edges
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    return builder.compile()

def make_alternative_graph():
    """
    Make a tool-calling agent
    """
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def call_model_with_tools(state: State):
        return {"messages": [model_with_tools.invoke(state.messages)]}

    def should_continue(state: State):
        last_message = state.messages[-1]
        if last_message.tool_calls:
            return "tools"
        else:
            return END
    

    builder = StateGraph(State)
    builder.add_node("agent", call_model_with_tools)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        { "tools": "tools", END: END }
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("tools", END)
    return builder.compile()

agent = make_alternative_graph()