# Ref2Font V3 (FLUX.2 Klein 9B LoRA) — Contextual LoRA for Font Atlases

This repository contains **V3** of the contextual LoRA for **FLUX.2-klein-9B**. It generates **1280×1280 font atlases** from a single reference image (**"Aa"** for Latin or **"Аа"** for Cyrillic scripts).

**Update V3 (Latest):** 
- **Cyrillic Support:** Full support for Russian.
- **Expanded Charset:** Added `"` (double quote) and `&` (ampersand) to all atlases.
- **Updated Prompts:** New specific prompts for different charsets to ensure mapping stability.
- **Straighter Letters:** Improved alignment and reduced "jitter" in atlas generation.

**Update V2:** Fixed dataset generation issues, increased resolution to 1280px, and improved vectorization scripts.

## What’s Inside
- **LoRA weights**: `Ref2FontV3.safetensors`
- **ComfyUI workflow**: `Example Workflow/` (see notes inside the workflow nodes)
- **Examples**: `Example/` (input images + generated atlases)
- **Post-processing scripts**: `flux_pipeline.py`, `flux_grid_to_ttf.py`, `flux_upscale.py`

> Disclaimer: it works **well**, but **not perfectly**. Expect occasional artifacts.

## Examples

<img width="2560" height="2560" alt="Example_3_C" src="https://github.com/user-attachments/assets/c47dfb37-1e25-434a-bd0d-6e6057ee020c" />

<img width="2560" height="2560" alt="Example_5_C" src="https://github.com/user-attachments/assets/41634405-5019-4e26-bdc3-8c9e6add55e5" />

<img width="2560" height="2560" alt="Example_6_C" src="https://github.com/user-attachments/assets/518fe0b2-5a5b-48a8-8aff-28050c107028" />

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

### For Latin (English):
**Reference image must contain "Aa"**
> A technical font atlas grid of the Latin charset: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:-"&". The style is strictly derived from the reference image "Aa".

### For Cyrillic (Russian):
**Reference image must contain "Аа"**
> A technical font atlas grid of the Cyrillic charset: "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!?.,;:-"&". The style is strictly derived from the reference image "Аа".

### Input image rules
- **Strict black & white only** (no gray, no shadows, no volume)
- **1280×1280** (recommended) or 1024x1024
- Follow the examples in `Example/`

## Post-processing: Atlas → TTF
After you generate the atlas, use the pipeline script to convert the atlas into a TTF font.

### Improved glyph extraction and alignment

Grid conversion now defaults to `--metrics-mode normalized`. It estimates shared
capital and lowercase heights, aligns individual letters to the baseline, and
handles Latin/Cyrillic accents, descenders and punctuation separately. Scaling
preserves each drawing's proportions and is limited to 0.8–1.25 to avoid extreme
corrections. This is a heuristic for these alphabets: unusual letterforms may
still need manual editing. Use `--metrics-mode legacy` to keep atlas placement
and the previous `--baseline-mode` / `--descender-lift` controls.

Edge cleanup preserves antialiasing and rejects noise/neighbor-only components.
Tracing closes contours at crop boundaries. `--edge-blur 0.45` smooths in source
pixels; `--edge-blur 0` disables that smoothing. `--simplify` is now also measured
in **source pixels** (default `0.25`), independent of `--trace-scale`; old numerical
values therefore have a stronger effect. Horizontal metrics match actual outline
bounds, including in visual alignment mode. Line metrics are shared by the TTF
tables for more consistent text layout.

The V3 character presets include quotes and ampersand. Use `--language cyrillic`
for Russian or `--language latin` (default) for Latin. `--charset` overrides the
preset; pass the original charset explicitly for older atlases without `"&`.

### Example commands (Windows / PowerShell)

Run from the repository root after installing `requirements.txt`. These commands
use the tested V3 presets and normalized metrics. Component cleanup and automatic
background detection are enabled by default; no upscaler model is required.

**Cyrillic:**

```powershell
python flux_pipeline.py `
  --input "Example/V3/Example_3_Output_Cyrillic.png" `
  --output-dir "output/cyrillic" `
  --no-upscale `
  --use-grid `
  --language cyrillic `
  --metrics-mode normalized `
  --align-mode geometric `
  --edge-blur 0.45 `
  --simplify 0.25 `
  --trace-scale 8 `
  --debug-dir "output/debug"
```

**Latin:**

```powershell
python flux_pipeline.py `
  --input "Example/V3/Example_3_Output_L.png" `
  --output-dir "output/latin" `
  --no-upscale `
  --use-grid `
  --language latin `
  --metrics-mode normalized `
  --align-mode geometric `
  --edge-blur 0.45 `
  --simplify 0.25 `
  --trace-scale 8 `
  --debug-dir "output/debug"
```

Replace `--input` with your own atlas path. In PowerShell, the backtick must be
at the end of the line with no trailing spaces. For Command Prompt (`cmd.exe`),
put the command on one line or replace each continuation backtick with `^`.

The explicit quality settings above match the pipeline defaults. For a shorter
command, they can be omitted:

```powershell
python flux_pipeline.py --input "Example/V3/Example_3_Output_Cyrillic.png" --output-dir "output/cyrillic" --no-upscale --use-grid --language cyrillic
```

`--align-mode geometric` gives consistent geometric side bearings.
`--align-mode visual` is an optional alternative that shifts glyphs by foreground
centroid within the available side bearings. `--debug-dir` writes per-glyph bounds
and advance widths to `metrics.json` for grid conversion; omit it if not needed.
The non-grid converter retains its existing behavior.

To retain the atlas's original letter heights and positions, replace
`--metrics-mode normalized` with `--metrics-mode legacy --baseline-mode auto`.
This retains the previous alignment strategy while keeping the extraction fixes.

No automatic kerning or TrueType hinting is added; small-size rendering and
individual letter pairs may still benefit from a font editor.

For a rendered comparison of two existing fonts and regression checks:

```powershell
python scripts/preview_font.py --before "old.ttf" --after "new.ttf" --language cyrillic --output "output/comparison.png"
python -m unittest discover -s tests -v
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







