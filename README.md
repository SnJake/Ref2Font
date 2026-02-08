# Ref2Font V2 (FLUX.2 Klein 9B LoRA) — Contextual LoRA for Font Atlases

This repository contains the **V2** of the **contextual LoRA** trained for **black-forest-labs/FLUX.2-klein-9B**. It generates **1280×1280 font atlases** from a single reference image "Aa".

**Update V2:** Fixed dataset generation issues, increased resolution to 1280px, and improved vectorization scripts.

## What’s Inside
- **LoRA weights**: `Ref2FontV2.safetensors`
- **ComfyUI workflow**: `Example Workflow/` (see notes inside the workflow nodes)
- **Examples**: `Example/` (input images + generated atlases)
- **Post-processing scripts**: `flux_pipeline.py`, `flux_grid_to_ttf.py`, `flux_upscale.py`

> Disclaimer: it works **well**, but **not perfectly**. Expect occasional artifacts.

## Examples
<img width="2560" height="1280" alt="Example_1_C" src="https://github.com/user-attachments/assets/d9064bbf-f3e9-4753-abbe-af2e83493746" />

<img width="2560" height="1280" alt="Example_2_C" src="https://github.com/user-attachments/assets/f5a97678-862f-4930-b8ce-cad3da7cef77" />

<img width="2560" height="1280" alt="Example_3_C" src="https://github.com/user-attachments/assets/583ac8e9-04df-4df0-bb0e-89929a482400" />

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
Download the LoRA (V2):

[HF Repo](https://huggingface.co/SnJake/Ref2Font)

Or from [CivitAI](https://civitai.com/models/2361340).

Place in: `ComfyUI/models/loras`

### Input image rules
- **Strict black & white only** (no gray, no shadows, no volume)
- **1280×1280** (recommended) or 1024x1024
- Follow the examples in `Example/`

## Post-processing: Atlas → TTF
After you generate the atlas, use the pipeline script to convert the atlas into a TTF font.

### Example commands (Windows)

```powershell
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
  --keep-components 4 ^
  --min-component-area 3 ^
  --component-center-bias 0.65 ^
  --cell-bleed 0.4 ^
  --cell-bleed-max 10 ^
  --core-overlap-min 0.35 ^
  --no-auto-invert
```

```powershell
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
  --keep-components 3 ^
  --min-component-area 10 ^
  --component-center-bias 0.35 ^
  --cell-bleed 0.12 ^
  --cell-bleed-max 32 ^
  --no-auto-invert
```

## Recommended Workflow (Step-by-step)
1) Download base models (see links above) and place them in ComfyUI folders.
2) Download LoRA and put it in `ComfyUI/models/loras`.
3) Create the input image (1280×1280 preferred, pure black/white).
4) Run the ComfyUI workflow (`Example Workflow/`) and generate the atlas. 
   
**⚠️ IMPORTANT:** To get the correct grid layout and character sequence, you **must** use this prompt:
> Generate letters and symbols "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-" in the style of the letters given to you as a reference.

5) Create and activate a venv, then install dependencies.
6) Run `flux_pipeline.py` with your atlas path to generate the TTF.

## License
MIT



