import sqlite3
import os
from typing import TypedDict, Annotated, List, Dict, Any
from absl import app, flags
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from pathlib import Path
from db_utils import create_user_database, get_user_name, cleanup_db, persist_user_name_to_db, CHECKPOINT_DB, ALL_DBS

FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False, "Enable debug logging.")

# Load environment variables
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- 1. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], lambda x, y: x + y if y else x]
    user_name: str
    user_id: str

# --- 2. Define Tools for External Interaction ---
# Rationale for this tool:
# Tools are the exclusive mechanism for the agent to interact with the outside world.
# They are stateless and cannot modify the agent's internal state directly.
# This function, `remember_user_name_external`, serves as the agent's only touchpoint
# with the persistent external database. Its arguments (`user_id`, `name`) are critical,
# as they are the only data passed from the agent's reasoning process to the external system.
@tool
def remember_user_name_external(user_id: str, name: str) -> str:
    """Use this to remember the user's name in their permanent profile."""
    persist_user_name_to_db(user_id, name)
    return f"OK, I've saved {name} to your main profile."

tools = [remember_user_name_external]
tool_executor = ToolNode(tools)


# Rationale for global initialization:
# Initializing the LLM and binding tools are expensive operations.
# By defining `llm` and `agent_runnable` here, we create a single,
# reusable instance that persists for the application's lifecycle.
# This avoids the severe performance overhead of re-initializing
# the model on every invocation of the `call_model` node.
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
agent_runnable = llm.bind_tools(tools)


def get_system_prompt(name):
    """Generates the system prompt based on whether the user's name is known."""
    
    # 1. Start with a clear base role and personality.
    system_prompt = ("You are a helpful assistant. "
    "Your goal is to learn about the user's name and use it in the converation.")

    if name:
        # 2. Instructions for when the name IS known.
        system_prompt += (f" You know the user's name is {name}. Address them by their name {name}")
    else:
        # 3. Instructions for when the name IS NOT known.
        # The prompt needs to be more proactive and explicit about the tool-call trigger.
        system_prompt += (f" You do not know the user's name yet. "
            "\n\n**CRITICAL INSTRUCTION:** When the user provides their name "
            "(e.g., 'Hi, my name is Bob'), "
            "you MUST immediately call the 'remember_user_name_external' tool "
            "with the exact name they provided. This is your highest priority "
            "task when a name is given. Do not answer their other queries "
            "in the same turn; just call the tool."
        )
        
    return system_prompt

def call_model(state: AgentState):
    """The agent node, which now reads the manually injected state."""
    messages = state["messages"]
    name = state.get("user_name")
    if FLAGS.debug:
        print(f"DEBUG: call_model received name='{name}'")
   
    system_prompt = get_system_prompt(name=name)

    all_messages = [SystemMessage(content=system_prompt)] + messages
    if FLAGS.debug:
        print(f"DEBUG LLM Input Messages:\n{all_messages}\n")
    
    response = agent_runnable.invoke(all_messages)
    if FLAGS.debug:
        print(f"DEBUG LLM Response Messages:\n{response}\n")
    
    # Note: The following snippet demonstrates how to inject the 'user_id' into the tool call.
    # However, this approach is commented out because it violates the Single Responsibility Principle.
    # The 'call_model' node's primary responsibility is to interface with the language model.
    # Modifying the tool call arguments here would give it the additional responsibility of state manipulation,
    # which is poor design. A dedicated node ('update_tool_call') is used instead to handle this task.
    # if response.tool_calls:
    #     response.tool_calls[0]['args']['user_id'] = state["user_id"]
        
    return {"messages": [response]}


# Rationale for this node:
# The LLM, in its tool-use request, only provides the arguments it can infer from the
# user's message (e.g., the user's name). It is unaware of application-level context
# like the `user_id`, which is essential for the tool to interact with the correct
# user's data in the external database. This node acts as a bridge, intercepting the
# LLM's tool call and injecting the necessary `user_id` from the agent's state
# before the tool is actually executed.
def update_tool_call_with_user_id(state: AgentState):
    """Adds the user_id to the tool call arguments."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_call['args']['user_id'] = state["user_id"]
        # Create a new AIMessage with the updated tool call
        new_message = AIMessage(
            content=last_message.content,
            tool_calls=[tool_call],
            id=last_message.id,
        )
        return {"messages": [new_message]}
    return {}

# Rationale for this node:
# Tools are designed to be stateless and cannot modify the agent's internal state directly.
# This node ensures that the agent's state is updated only *after* the tool has successfully
# persisted the information to the external database. It acts as a synchronization point,
# reflecting the external change in the agent's memory for subsequent turns in the conversation.
def update_user_name_in_state(state: AgentState):
    """Updates the user_name in the state based on the last tool call."""
    if FLAGS.debug:
        print(f"\n--- DEBUG: State before explicit update: User Name in State: {state.get('user_name')} ---")
    
    last_message = state["messages"][-2] # The message before the tool result
    if last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        if tool_call['name'] == 'remember_user_name_external':
            name = tool_call['args']['name']
            return {"user_name": name}
    return {}

# --- 3. Define Graph Nodes ---
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("update_tool_call", update_tool_call_with_user_id)
builder.add_node("tools", tool_executor)
builder.add_node("update_state_with_name", update_user_name_in_state)

# --- 4. Build the Graph ---
builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "update_tool_call", END: END}
)
builder.add_edge("update_tool_call", "tools")

# This commented-out edge represents the simpler graph structure that would be used
# if the 'user_id' were injected directly within the 'call_model' node. However,
# to adhere to the Single Responsibility Principle, we use a dedicated 'update_tool_call'
# node, which requires the more complex routing defined above.
# builder.add_conditional_edges(
#     "agent",
#     tools_condition,
#     {"tools": "tools", END: END}
# )

builder.add_edge("tools", "update_state_with_name")
builder.add_edge("update_state_with_name", "agent")

def generate_agent_state(user_id: str, user_message: str) -> dict:
    """
    Generates the agent state for invoking the graph and prints user preferences.

    Args:
        user_id: The ID of the user, supplied by the invoking application.
        user_message: The message from the user.

    Returns:
        A dictionary representing the agent's state, including the user message,
        user preferences loaded from a persistent database (keyed by user_id),
        and the user_id itself.
    """
    user_name = get_user_name(user_id)
    print(f"AppServer: Retrieved user_name for user_id='{user_id}' from external database: '{user_name}'")
    return {
        "messages": [HumanMessage(content=user_message)],
        "user_name": user_name,
        "user_id": user_id,
    }

def print_agent_response(response: dict, run_label: str):
    """Prints the agent's response, handling different content formats."""
    prefix = f"{run_label} Agent Response:"
    agent_response = response['messages'][-1]
    if isinstance(agent_response.content, list) and agent_response.content:
        print(f"{prefix} {agent_response.content[0]['text']}")
    else:
        print(f"{prefix} {agent_response.content}")
    if FLAGS.debug:
        print(f"DEBUG {prefix} (Full):\n{agent_response}\n")

def main(_):
    try:
        create_user_database()

        with SqliteSaver.from_conn_string(CHECKPOINT_DB) as memory:
            app = builder.compile(checkpointer=memory)

            print(f"\n[Run 1: User shares name in NEW session]")
            USER_ID, USER_NAME, SESSION_ID_1 = "user_John", "John", "session_789"
            config_1 = {"configurable": {"thread_id": SESSION_ID_1}}
            initial_state_1 = generate_agent_state(USER_ID, f"Hi, my name is {USER_NAME}.")
            response = app.invoke(initial_state_1, config=config_1)
            print_agent_response(response, "Run 1")
            print(f"AppServer: (Write 1) External DB is now: {get_user_name(USER_ID)}")

            print(f"\n[Run 2: User asks name in ANOTHER NEW session]")
            USER_ID, SESSION_ID_2 = "user_John", "session_101"
            config_2 = {"configurable": {"thread_id": SESSION_ID_2}}
            initial_state_2 = generate_agent_state(USER_ID, "What is my name?")
            response = app.invoke(initial_state_2, config=config_2)
            print_agent_response(response, "Run 2")

            print(f"\n[Run 3: A different user in a NEW session]")
            NEW_USER_ID, SESSION_ID_3 = "user_Jane", "session_202"
            config_3 = {"configurable": {"thread_id": SESSION_ID_3}}
            initial_state_3 = generate_agent_state(NEW_USER_ID, "Do you know my name?")
            response = app.invoke(initial_state_3, config=config_3)
            print_agent_response(response, "Run 3")
    finally:
        cleanup_db(ALL_DBS)

if __name__ == "__main__":
    app.run(main)
