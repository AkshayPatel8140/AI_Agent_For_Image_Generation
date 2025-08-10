# multi_agent_image_generation.py
# Two-agent LangGraph:
# 1) Prompt Engineer agent: turns a vague idea into a polished image prompt
# 2) Image Generator agent: calls OpenAI Images API and saves a PNG locally


# Run this command in the terminal to run this code
# python -m venv .venv
# For mac => source .venv/bin/activate
# For Windows => .venv\Scripts\activate
# pip install -U langgraph langchain langchain-community "langchain[openai]" duckduckgo-search pydantic openai
# export OPENAI_API_KEY=sk-...   # Power a stronger chat model if desired
from __future__ import annotations

import base64
import os
from datetime import datetime
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command


# Optional: model name can be any latest model
CHAT_MODEL = "openai:gpt-4.1-mini"
IMAGE_MODEL = "gpt-image-1"


# ============================
# Image generation tool (OpenAI Images API)
# ============================
# This is written as a simple function. The Image agent will call it directly.
def generate_image_with_openai(prompt: str, size: str = "1024x1024") -> str:
    """Calls OpenAI Image API, writes PNG to the local folder, and return the files path."""

    from openai import OpenAI

    client = OpenAI()

    result = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=size,
        n=1,
        # response_format="b64_json",
    )

    b64 = result.data[0].b64_json
    img_bytes = base64.b64decode(b64)

    os.makedirs("generated_image", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("generated_image", f"image_{ts}.png")
    with open(file_path, "wb") as f:
        f.write(img_bytes)
    return file_path


# ============================
# Shared Graph State
# ============================
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================
# System Prompt
# ============================
PROMPT_ENGINEER_SYS = (
    "Act as a world-class prompt engineer for text-to-image models. "
    "Given a user idea, produce a single polished prompt that is concrete, visual, and style-aware. "
    "Include subject, scene, composition, lighting, lens, color palette, and quality tags. "
    "Avoid copyrighted terms and disallowed content. "
    "Output exactly one line starting with: FINAL PROMPT: "
)

IMAGE_AGENT_SYS = (
    "Act as an image-generation operator. "
    "Input will be a single line starting with 'FINAL PROMPT:'. "
    "Call the image tool with that prompt. "
    "When done, respond with exactly one line starting with 'FINAL ANSWER:' followed by the local file path."
)

# ============================
# Define LLM
# ============================
llm = init_chat_model(CHAT_MODEL)


# ============================
# Helper Route
# ============================
def _route(last_message: BaseMessage, default_next: str) -> str:
    content = getattr(last_message, "content", "")
    if isinstance(content, str) and content.startswith("FINAL ANSWER:"):
        return END
    return default_next


# ============================
# Nodes
# ============================
def prompt_engineer_node(state: MessagesState) -> Command[Literal["image_node", END]]:
    """Uses the chat mode to transform the use's idea into a single engineered prompt line."""

    msgs = [SystemMessage(content=PROMPT_ENGINEER_SYS), *state["messages"]]
    ai = llm.invoke(msgs)

    text = ai.content.strip()
    if "FINAL PROMPT:" not in text:
        text = "FINAL PROMPT: " + text.replace("\n", " ").strip()

    new_messages = state["messages"] + [AIMessage(content=text, name="prompt_engineer")]
    # goto = _route(AIMessage(content=text), "image_node")
    return Command(update={"messages": new_messages}, goto="image_node")


def image_node(state: MessagesState) -> Command[Literal[END, "prompt_node"]]:
    """Reads the engineered prompt and generate an image using OpenAI Images API."""

    final_prompt = None
    for m in reversed(state["messages"]):
        c = getattr(m, "content", "")
        if isinstance(c, str) and c.startswith("FINAL PROMPT:"):
            final_prompt = c.replace("FINAL PROMPT:", "").strip()
            break
    if not final_prompt:
        return Command(update={}, goto="prompt_node")

    file_path = generate_image_with_openai(final_prompt, size="1024x1024")
    out_line = f"FINAL ANSWER: {file_path}"

    new_message = state["messages"] + [AIMessage(content=out_line, name="image_agent")]
    # goto = _route(AIMessage(content=out_line), END)
    return Command(update={"messages": new_message}, goto=END)


# ============================
# Graph
# ============================
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("prompt_node", prompt_engineer_node)
graph_builder.add_node("image_node", image_node)

graph_builder.add_edge(START, "prompt_node")

graph = graph_builder.compile()


# ============================
# Main Entry
# ============================
if __name__ == "__main__":
    print("Example idea:")
    print(
        "A cozy steampunk library interior with warm light and brass, isometric view, octane render, ultra-detailed"
    )
    idea = input("\nIdea: ").strip()

    final_path = None
    for event in graph.stream(
        {"messages": [HumanMessage(content=idea)]}, stream_mode="values"
    ):
        last = event["messages"][-1]
        txt = getattr(last, "content", "")
        print("Agent:", txt)
        if isinstance(txt, str) and txt.startswith("FINAL ANSWER:"):
            final_path = txt.replace("FINAL ANSWER:", "").strip()
            break

    if final_path:
        print(f"\nSaved image -> {final_path}")
