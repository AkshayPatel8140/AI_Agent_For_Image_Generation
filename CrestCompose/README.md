# CrestCompose — Brand-Aware Image Generator

Multi-agent pipeline for **on-brand image generation**:
1. Loads brand KB (colors, dos/donts, logos).
2. Engineers a prompt consistent with brand style.
3. Generates a base image.
4. Composites exact brand logo without recolor or distortion.

---

## 📂 Structure
```
CrestCompose/
├── brand_image_agent.py
├── brand_kb/
│   ├── brand_kb.json         # Brand details
│   └── logos/                # Company logo PNGs
│       ├── acme/
│       └── redwhite/
└── generated_brand_image/    # Output folder
```

---

## 🔧 Requirements
```bash
pip install -U langgraph langchain openai pillow numpy pydantic
```
Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...   # macOS/Linux
setx OPENAI_API_KEY "sk-..."   # Windows PowerShell
```

---

## 🛠 Prepare Brand KB
Example `brand_kb.json`:
```json
{
  "acme": {
    "display_name": "ACME Robotics",
    "brand_colors_hex": ["#FF3B3B", "#111111", "#FFFFFF"],
    "voice": "innovative, confident, minimal",
    "visual_style": "modern, high contrast, light-on-dark",
    "dos": ["use red as accent only", "prefer dark backgrounds", "keep clutter low"],
    "donts": ["never stretch logo", "never recolor logo", "no drop shadows"],
    "logo_variants": {
      "light": "logos/acme/acme_logo_light.png",
      "dark": "logos/acme/acme_logo_dark.png"
    },
    "default_logo_preference": "light",
    "default_logo_position": "top-right"
  }
}
```

---

## ▶️ Run
```bash
cd CrestCompose
python brand_image_agent.py   
--company acme   
--idea "Hero banner with sleek robotics theme, metallic textures, minimal layout"   
--size 1024x1024   
--position top-right
```

---

## 📜 Notes
- Logos must be **transparent PNGs**.
- No recoloring, scaling preserves aspect ratio.
- Picks light/dark logo variant automatically for contrast unless overridden.
