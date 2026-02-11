# Ref2Font V3 (FLUX.2 Klein 9B LoRA) — Contextual LoRA for Font Atlases

This repository contains the **V3** of the **contextual LoRA** trained for **black-forest-labs/FLUX.2-klein-9B**. It generates **1280×1280 font atlases** from a single reference image "Aa".

**Update V3 (Latest):** 
- **Cyrillic Support:** Full support for Russian, Ukrainian, and other Cyrillic-based scripts.
- **Expanded Charset:** Added `"` (double quote) and `&` (ampersand) to all atlases.
- **Updated Prompts:** New specific prompts for different charsets to ensure mapping stability.

**Update V2:** Fixed dataset generation issues, increased resolution to 1280px, and improved vectorization scripts.

## What’s Inside
- **LoRA weights**: `Ref2FontV3.safetensors`
- **ComfyUI workflow**: `Example Workflow/` (see notes inside the workflow nodes)
- **Examples**: `Example/` (input images + generated atlases)
- **Post-processing scripts**: `flux_pipeline.py`, `flux_grid_to_ttf.py`, `flux_upscale.py`

> Disclaimer: it works **well**, but **not perfectly**. Expect occasional artifacts.

## Examples


## Requirements
The post-processing scripts require Python 3.10+ and these packages:

```
numpy
pillow
fonttools
scikit-image
tqdm
```

**For `--no-upscale` workflow this is enough (recommended).**

**`flux_upscale.py` is currently experimental and may not improve quality yet.**

## Setup

```powershell
git clone https://github.com/SnJake/Ref2Font.git
cd Ref2Font
# from the repo root
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ComfyUI Workflow
The workflow is in `Example Workflow/`. It already contains detailed notes inside the nodes.

### Required models
1) Base model (FLUX.2 Klein 9B):
```
https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/flux-2-klein-base-9b.safetensors
```
Place in: `ComfyUI/models/diffusion_models`

2) Text encoder (Qwen):
```
https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors
```
Place in: `ComfyUI/models/text_encoders`

3) VAE:
```
https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/vae/flux2-vae.safetensors
```
Place in: `ComfyUI/models/vae`

### LoRA
Download the LoRA (V3):

[HF Repo](https://huggingface.co/SnJake/Ref2Font)

Or from [CivitAI](https://civitai.com/models/2361340).

Place in: `ComfyUI/models/loras`

## ⚠️ IMPORTANT: V3 Required Prompts
To get the correct grid layout and character sequence, you **must** use these specific prompts depending on your target language:

### For Latin (English, etc.):
**Reference image must contain "Aa"**
> A technical font atlas grid of the Latin charset: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"&". The style is strictly derived from the reference image "Aa".

### For Cyrillic (Russian, etc.):
**Reference image must contain "Аа"**
> A technical font atlas grid of the Cyrillic charset: "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!?.,;:-"&". The style is strictly derived from the reference image "Аа".

### Input image rules
- **Strict black & white only** (no gray, no shadows, no volume)
- **1280×1280** (recommended) or 1024x1024
- Follow the examples in `Example/`

## Post-processing: Atlas → TTF
After you generate the atlas, use the pipeline script to convert the atlas into a TTF font.

### Example commands (Windows)
`--align-mode geometric` keeps old bbox-based alignment. `--align-mode visual` centers glyphs by foreground centroid (recommended).

```
python flux_pipeline.py ^
  --input "path\to\your_atlas.png" ^
  --output-dir "output\folder" ^
  --no-upscale ^
  --use-grid ^
  --simplify 0.5 ^
  --canvas 1280 ^
  --contour-level 0.5 ^
  --trace-scale 4 ^
  --trace-blur 1.0 ^
  --smooth-iters 2 ^
  --baseline-mode auto ^
  --align-mode visual ^
  --keep-components 4 ^
  --min-component-area 3 ^
  --component-center-bias 0.65 ^
  --cell-bleed 0.4 ^
  --cell-bleed-max 10 ^
  --core-overlap-min 0.35 ^
  --charset "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-""&" ^
  --no-auto-invert
```

```
python flux_pipeline.py ^
  --input "path\to\your_atlas.png" ^
  --output-dir "output\folder" ^
  --no-upscale ^
  --use-grid ^
  --simplify 0.5 ^
  --canvas 1280 ^
  --contour-level 0.5 ^
  --trace-scale 4 ^
  --trace-blur 1.0 ^
  --smooth-iters 2 ^
  --baseline-mode auto ^
  --align-mode visual ^
  --keep-components 4 ^
  --min-component-area 3 ^
  --component-center-bias 0.65 ^
  --cell-bleed 0.4 ^
  --cell-bleed-max 10 ^
  --core-overlap-min 0.35 ^
  --charset "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!?.,;:-""&" ^
  --no-auto-invert
```

## Recommended Workflow (Step-by-step)
1) Download base models (see links above) and place them in ComfyUI folders.
2) Download LoRA and put it in `ComfyUI/models/loras`.
3) Create the input image (1280×1280 preferred, pure black/white).
4) Run the ComfyUI workflow (`Example Workflow/`) and generate the atlas. 
5) Create and activate a venv, then install dependencies.
6) Run `flux_pipeline.py` with your atlas path to generate the TTF.

## License
MIT



