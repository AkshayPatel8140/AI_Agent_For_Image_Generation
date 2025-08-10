# agent_simple.py
# A minimal LangGraph agent that can decide to call tools:
# - DuckDuckGo web search
# - Safe calculator

# Run this command in the terminal to run this code
# python -m venv .venv
# For mac => source .venv/bin/activate
# For Windows => .venv\Scripts\activate
# pip install -U langgraph langchain langchain-community "langchain[openai]" duckduckgo-search pydantic
# export OPENAI_API_KEY=sk-...   # Power a stronger chat model if desired
# python agent_simple.py


from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict


from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression like 2*(3+4)/5.
    Only +, -, *, /, **, parentheses, and integers/floats are allowed."""

    import ast, operator as op

    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers allowed")
        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Disallowed expression")

    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


search_tool = DuckDuckGoSearchRun()  # no API key required

# List of the tools
tools = [search_tool, calculator]


# ---------- State ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- Model ----------
# Pick a chat model provider in one line. Works with OpenAI, Anthropic, Gemini, Bedrock, or local providers via LangChain.
# Examples:
#   llm = init_chat_model("openai:gpt-4.1-mini")
#   llm = init_chat_model("google_genai:gemini-2.0-flash")
#   llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
# For a no-key local option (if Ollama is installed), try:
#   pip install langchain-ollama && llm = init_chat_model("ollama:llama3.1")
llm = init_chat_model("openai:gpt-4.1-mini")


# Bind the tools with the llm
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ---------- Graph ----------
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


if __name__ == "__main__":
    print(
        "Try questions like:\n - search: What is LangGraph?\n - math: What is (2.5+7)/3?"
    )
    while True:
        try:
            q = input("\n User:").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            for event in graph.stream(
                {"messages": [HumanMessage(content=q)]}, stream_mode="values"
            ):
                last = event["messages"][-1]
                print("Assistant:", getattr(last, "content", last))
        except KeyboardInterrupt:
            break
