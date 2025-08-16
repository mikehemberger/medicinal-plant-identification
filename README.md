# Medicinal Plants Visual Search Demo

Quickstart:

1. Ensure Python 3.10+ is available. This environment uses Python 3.13.
2. Install dependencies:

```
python3 -m pip install --break-system-packages -q datasets pillow numpy imagehash gradio tqdm rapidocr-onnxruntime onnxruntime
```

3. Build the index (first run will download ~304 images from the Hugging Face dataset):

```
python3 -m app.visual_search --out /workspace/app/index.npz
```

4. Launch the Gradio app:

```
python3 /workspace/app/app.py
```

5. Open the app in your browser at http://localhost:7860/ and upload a plant image.

Notes:
- The dataset (`mikehemberger/medicinal-plants`) has plant names printed at the bottom center. The app attempts to OCR this label crop using `rapidocr-onnxruntime` if available. It's optional.
- The visual search uses a lightweight pHash + HSV histogram index to avoid heavy ML dependencies; swap in CLIP embeddings later if desired.