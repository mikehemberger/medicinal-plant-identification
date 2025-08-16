# Medicinal Plants Visual Search Demo

Quickstart:

1. Ensure Python 3.10+ is available. This environment uses Python 3.13.
2. Install dependencies:

```
python3 -m pip install --break-system-packages -q datasets pillow numpy imagehash gradio tqdm rapidocr-onnxruntime onnxruntime requests
```

3. Choose an indexing mode:
- Simple (default): pHash + HSV histogram, no API keys.
- CLIP (remote): Use online inference to compute CLIP embeddings.

To enable CLIP via Hugging Face Inference API:
```
export HF_API_TOKEN=hf_...your_token...
# optional: select model (default uses laion ViT-B/32)
# python3 -m app.visual_search --mode clip --clip-provider hf --hf-model-id laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

To enable CLIP via Roboflow Inference:
```
export ROBOFLOW_API_KEY=rf_...your_key...
# python3 -m app.visual_search --mode clip --clip-provider roboflow
```

4. Build the index (first run will download ~304 images from the Hugging Face dataset):

```
# Simple mode
python3 -m app.visual_search --out /workspace/app/index.npz --mode simple

# CLIP mode (requires one of the env vars above)
python3 -m app.visual_search --out /workspace/app/index.npz --mode clip --clip-provider hf
```

5. Launch the Gradio app:

```
python3 /workspace/app/app.py
```

6. Open the app in your browser at http://localhost:7860/ and upload a plant image.

Notes:
- If a CLIP API token is present at first launch, the app will auto-build a CLIP index. Otherwise it builds the simple index.
- The dataset (`mikehemberger/medicinal-plants`) has plant names printed at the bottom center. The app attempts to OCR this label crop using `rapidocr-onnxruntime` if available. It's optional.
- You can switch modes and rebuild the index from the UI under the Advanced section.