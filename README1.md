# AI Agent for Image Generation

This repository contains **three AI agent projects** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://www.langchain.com/) to demonstrate different patterns for image generation.

## 📂 Project Structure
```
AI_Agent_for_Image_Generation/
│
├── simpleAgent/                   # Single-agent example
│   └── simple_agent.py
│
├── imageGeneretor/                 # Multi-agent image generator
│   ├── mulit_agent_image_generation.py
│   └── generated_image/            # Stores generated images
│
└── CrestCompose/                   # Brand-aware image generator
    ├── brand_image_agent.py
    ├── brand_kb/
    │   ├── brand_kb.json            # Brand details (colors, dos/donts, logo paths)
    │   └── logos/
    │       ├── acme/               # Example company logos
    │       └── redwhite/
    └── generated_brand_image/       # Stores brand-composited images
```

---

## 🔧 Setup
All projects require Python 3.10+.

```bash
# Clone repo
git clone https://github.com/<your-username>/AI_Agent_for_Image_Generation.git
cd AI_Agent_for_Image_Generation

# Create and activate venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install shared requirements
pip install -U langgraph langchain langchain-community openai pillow numpy pydantic duckduckgo-search
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...   # macOS/Linux
setx OPENAI_API_KEY "sk-..."   # Windows PowerShell
```

---

## ▶️ Running the Projects

### 1. Simple Agent (Tool-based chatbot)
```bash
cd simpleAgent
python simple_agent.py
```
- Supports DuckDuckGo search and a safe calculator tool.
- Type natural queries like:
  ```
  What is LangGraph?
  What is (2.5 + 7) / 3?
  ```
- Ctrl+C or type `quit` to exit.

---

### 2. Multi-Agent Image Generator
```bash
cd imageGeneretor
python mulit_agent_image_generation.py
```
- First agent: engineers an image prompt from your idea.
- Second agent: calls OpenAI Images API to create the image.
- Final PNG is saved to `generated_image/`.

Example run:
```
Idea: A cozy steampunk library with warm light and brass
```

---

### 3. Brand-Aware Image Generator (CrestCompose)
```bash
cd CrestCompose
python brand_image_agent.py   --company acme   --idea "Hero banner with sleek robotics theme, metallic textures, minimal layout"   --size 1024x1024   --position top-right
```
- Uses company brand KB from `brand_kb.json` and logo from `brand_kb/logos/`.
- Ensures **exact logo** placement without distortion or recolor.
- Output saved to `generated_brand_image/`.

---

## 📜 License
MIT License
