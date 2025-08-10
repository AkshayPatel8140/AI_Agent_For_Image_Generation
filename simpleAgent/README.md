# Simple Agent — Tool-Using Chatbot

A single-agent chatbot built with LangGraph that can:
1. Perform web searches using DuckDuckGo.
2. Safely evaluate math expressions.

---

## 📂 Structure
```
simpleAgent/
└── simple_agent.py
```

---

## 🔧 Requirements
```bash
pip install -U langgraph langchain langchain-community duckduckgo-search
```
Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...   # macOS/Linux
setx OPENAI_API_KEY "sk-..."   # Windows PowerShell
```

---

## ▶️ Run
```bash
cd simpleAgent
python simple_agent.py
```

---

## 💬 Example
```
User: What is LangGraph?
Assistant: LangGraph is...

User: What is (2.5 + 7) / 3?
Assistant: 3.17
```

Ctrl+C or type `quit` to exit.

---

## 📜 Notes
- `calculator` tool uses Python's AST for **safe** arithmetic parsing.
- `DuckDuckGoSearchRun` requires no API key.
