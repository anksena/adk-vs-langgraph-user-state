from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def get_name(tool_context: ToolContext) -> str:
    """Returns the name of the user."""
    name = tool_context.state.get("user:name")
    if name:
        return name
    else:
        return "I don't know your name."

def remember_name(name: str, tool_context: ToolContext) -> str:
    """Remembers the user's name."""
    tool_context.state["user:name"] = name
    return f"I will remember your name as {name}."

root_agent = Agent(
    name="StatefulAgent",
    model="gemini-2.5-flash",
    description="A simple agent that remembers your name.",

    instruction=(
        "You are a helpful assistant that remembers the user's name. "
        
        "If the user tells you their name, you MUST use the `remember_name` tool. "
        
        "If the user asks for their name, check the state. If 'user:name' exists, "
        "tell them their name. Otherwise, you MUST use the `get_name` tool to retrieve it."
    ),    
    tools=[
        remember_name,
        get_name,
    ],
)
