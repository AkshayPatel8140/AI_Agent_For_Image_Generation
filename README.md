# AI Agent for Image Generation

This repository contains **three AI agent projects** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://www.langchain.com/) to demonstrate different patterns for image generation.

## 📂 Project Structure
```
AI_Agent_for_Image_Generation/
│
├── simpleAgent/                        # Single-agent example
│   └── simple_agent.py
│
├── imageGeneretor/                     # Multi-agent image generator
│   ├── mulit_agent_image_generation.py
│   └── generated_image/                # Stores generated images
│
├── CrestCompose/                       # Brand image generator
│   ├── brand_image_agent.py
│   ├── brand_kb/
│   │   ├── brand_kb.json               # Brand details (colors, dos/donts, logo paths)
│   │   └── logos/
│   │       ├── acme/                   # company logos
│   │       │   ├── acme_logo_dark.png                   
│   │       │   └── acme_logo_white.png     
│   │       │              
│   │       └── redwhite/
│   │           ├── redwhite_logo_dark.png                   
│   │           └── redwhite_logo_white.png  
│   └── generated_brand_image/          # Stores brand-composited images    
│
└── brandAwareImageGenerator/           # Brand-aware image generator
    ├── brand_aware_image.py
    ├── brand.json                      # Details of the brand
    └── output/                         # Stores generated images  

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

## Set your OpenAI API key:
```
export OPENAI_API_KEY=sk-...   # macOS/Linux
setx OPENAI_API_KEY "sk-..."   # Windows PowerShell
```

## Running the Projects

### 1. Simple Agent (Tool-based chatbot)
- Supports DuckDuckGo search and a safe calculator tool.
- Run this in the terminal

    ```bash
    cd simpleAgent
    python simple_agent.py
    ```
- Type natural queries like:
    ```bash
    What is LangGraph?
    What is (2.5 + 7) / 3?
    ```
- Ctrl+C or type quit to exit.


### 2. Multi-Agent Image Generator
- First agent: engineers an image prompt from your idea.
- Second agent: calls OpenAI Images API to create the image.
- Final PNG is saved to generated_image/.
- Run this in the terminal.

    ```bash
    cd imageGeneretor
    python mulit_agent_image_generation.py
    ```
- Type natural queries like:
    ```bash
    Idea: A cozy steampunk library with warm light and brass
    ```
- Ctrl+C or type quit to exit.

### 3. Brand Image Generator (CrestCompose)
- Uses company brand KB from brand_kb.json and logo from brand_kb/logos/.
- Ensures exact logo placement without distortion or recolor.
- Output saved to generated_brand_image/.
- Run this in the terminal.

    ```bash
    cd CrestCompose
    python brand_image_agent.py \
    --company acme \
    --idea "Hero banner with sleek robotics theme, metallic textures, minimal layout" \
    --size 1024x1024 \
    --position top-right
    ```

### 4. Brand-Aware Image Generator
- A single-pass, brand-aware pipeline that turns an idea into an on-brand image, performs quick brand checks, and writes a credentials sidecar for provenance.
**What it does**
- Prompt engineering using brand tone and palette
- Image generation using either OpenAI Images API or a local Pillow stub
- Brand checks: palette proximity and forbidden terms
- Writes a simple Content Credentials sidecar (JSON) with asset hash and brand schema hash

**Location**
```
brandAwareImageGenerator/
  └── brand_aware_image.py
```

**Run**
```bash
# From the repo root
cd brandAwareImageGenerator

# Option A: local stub (no external services)
python brand_aware_image.py \
  --idea "Back to school banner with friendly tone" \
  --size 1024x1024

# Option B: OpenAI Images API
export OPENAI_API_KEY=sk-...   # macOS/Linux
# setx OPENAI_API_KEY "sk-..." # Windows PowerShell
python brand_aware_image.py \
  --idea "Back to school banner with friendly tone" \
  --size 1024x1024 \
  --use-openai
```

**CLI options**
- `--idea`  Short idea text for the asset
- `--size`  WxH like `1024x1024` (default `1024x1024`)
- `--use-openai`  Boolean flag to call OpenAI Images API

**Outputs**
Files are written to `output/`:
- `base_<timestamp>.png`  Base image
- `final_<timestamp>.png`  Final image (same as base in this exercise)
- `credentials_<timestamp>.json`  Sidecar with minimal provenance
- `brand.json`  Auto-created sample brand config if missing

**Troubleshooting**
- Missing fonts: the stub falls back to Pillow default font
- OpenAI errors: confirm `OPENAI_API_KEY` and network access
- Invalid `brand.json`: the script prints a Pydantic validation error and exits
