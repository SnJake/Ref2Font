# Ref2Font (FLUX.2 Klein 9B LoRA) — Contextual LoRA for Font Atlases

This repository contains a **contextual LoRA** trained for **black-forest-labs/FLUX.2-klein-9B**. It generates **1024×1024 font atlases** from a single reference image (in the same style/format as the provided examples).

> Disclaimer: it works **well**, but **not perfectly**. Expect occasional artifacts and minor alignment issues.

## What’s Inside
- **LoRA weights**: `Ref2FontV1.safetensors`
- **ComfyUI workflow**: `Example Workflow/` (see notes inside the workflow nodes)
- **Examples**: `Example/` (input images + generated atlases)
- **Post-processing scripts**: `flux_pipeline.py`, `flux_grid_to_ttf.py`, `flux_upscale.py`

## Examples

<img width="2048" height="1024" alt="Example1_C" src="https://github.com/user-attachments/assets/c09b7f5e-93e9-4889-a8af-9eb9c69013ca" />

<img width="2048" height="1024" alt="Example2_C" src="https://github.com/user-attachments/assets/f1a711c5-119f-4b0c-9d09-4a5f4aefa0b2" />

<img width="2048" height="1024" alt="Example3_C" src="https://github.com/user-attachments/assets/db9eb2b8-7c7c-4afa-b3b1-69c0ae9bf53e" />

## Requirements
The post-processing scripts require Python 3.10+ and these packages:

```
numpy
pillow
fonttools
scikit-image
tqdm
```

**Upscaler is optional.** If you want to use `flux_upscale.py`, install PyTorch separately (CPU or CUDA build):

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```



## Setup

```powershell
git clone https://github.com/SnJake/Ref2Font.git
cd Ref2Font
# from the repo root
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# optional for upscaler
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## ComfyUI Workflow
The workflow is in `Example Workflow/`. It already contains detailed notes inside the nodes.

### Required models
1) Base model (FLUX.2 Klein 9B):
```
https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/flux-2-klein-9b.safetensors
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
Download the LoRA:

[HF Repo](https://huggingface.co/SnJake/Ref2Font)

Or from [CivitAI](https://civitai.com/models/2361340).

Place in: `ComfyUI/models/loras`

### Input image rules
- **Strict black & white only** (no gray, no shadows, no volume)
- **1024×1024** exactly
- Follow the examples in `Example/`

## Post-processing: Atlas → TTF
After you generate the atlas, use the pipeline script to convert the atlas into a TTF font.

### Example command (Windows)
```powershell
python flux_pipeline.py ^
  --input "D:\ComfyUI_temp_nckhb_00003_.png" ^
  --output-dir "G:\Flux 2 Klein 9B\test" ^
  --no-upscale ^
  --use-grid ^
  --vectorize contours ^
  --simplify 0.5 ^
  --canvas 1024 ^
  --contour-level 0.5 ^
  --trace-scale 4 ^
  --trace-blur 1.0 ^
  --smooth-iters 2 ^
  --baseline-mode auto ^
  --baseline-quantile 0.9 ^
  --baseline-min-pixels 20 ^
  --cols 8 ^
  --rows 9 ^
  --no-auto-invert
```

### Generic command template
```powershell
python flux_pipeline.py ^
  --input "path/to/atlas_name.png" ^
  --output-dir "output/dir" ^
  --no-upscale ^
  --use-grid ^
  --vectorize contours ^
  --simplify 0.5 ^
  --canvas 1024 ^
  --contour-level 0.5 ^
  --trace-scale 4 ^
  --trace-blur 1.0 ^
  --smooth-iters 2 ^
  --baseline-mode auto ^
  --baseline-quantile 0.9 ^
  --baseline-min-pixels 20 ^
  --cols 8 ^
  --rows 9 ^
  --no-auto-invert
```

## Recommended Workflow (Step-by-step)
1) Download base models (see links above) and place them in ComfyUI folders.
2) Download LoRA and put it in `ComfyUI/models/loras`.
3) Create the input image (1024×1024, pure black/white, like the examples). You can create input image in Nano Banana Pro or other similar models.
4) Run the ComfyUI workflow (`Example Workflow/`) and generate the atlas. **Important: To get the correct grid layout and character sequence, you must use this prompt:**
Generate letters and symbols "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-" in the style of the letters given to you as a reference.
5) Create and activate a venv, then install dependencies.
6) Run `flux_pipeline.py` with your atlas path to generate the TTF.

## Notes
- `flux_upscale.py` is optional. You can skip upscaling with `--no-upscale`.
- If you see odd inversion, try removing `--no-auto-invert` or add `--invert`.
- If you see vertical jitter, use `--baseline-mode auto` (enabled in the example above).

## License
MIT
