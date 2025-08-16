import os
from typing import List

import gradio as gr
from PIL import Image

from app.visual_search import PlantDatasetIndex

INDEX_PATH = '/workspace/app/index.npz'


def ensure_index() -> PlantDatasetIndex:
	if os.path.exists(INDEX_PATH):
		return PlantDatasetIndex.load(INDEX_PATH)
	mode = 'clip' if (os.getenv('HF_API_TOKEN') or os.getenv('ROBOFLOW_API_KEY')) else 'simple'
	provider = 'hf' if os.getenv('HF_API_TOKEN') else ('roboflow' if os.getenv('ROBOFLOW_API_KEY') else 'hf')
	idx = PlantDatasetIndex(mode=mode, clip_provider=provider)
	idx.build()
	os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
	idx.save(INDEX_PATH)
	return idx


INDEX = ensure_index()


def mode_label() -> str:
	if INDEX.mode == 'clip':
		return f"Mode: CLIP ({INDEX.clip_provider})"
	return 'Mode: simple (pHash + HSV)'


def rebuild_index(mode: str, provider: str):
	global INDEX
	idx = PlantDatasetIndex(mode=mode, clip_provider=provider)
	idx.build()
	idx.save(INDEX_PATH)
	INDEX = idx
	return mode_label()


def search_fn(image: Image.Image, top_k: int = 5):
	if image is None:
		return [], None
	results = INDEX.query(image, top_k=top_k)
	gallery_items: List[tuple] = []
	caption_lines: List[str] = []
	for r in results:
		label_text = r.predicted_label or '(OCR label unknown)'
		caption = f"{os.path.basename(r.file_id)}\nscore: {1.0 - r.distance:.3f}\nlabel: {label_text}"
		gallery_items.append((r.label_crop, f"Label: {label_text}"))
		caption_lines.append(caption)
	return gallery_items, '\n\n'.join(caption_lines)


demo = gr.Blocks()
with demo:
	gr.Markdown("""
	**Medicinal Plants Visual Search**
	
	Upload a plant photo. The app searches an index built from the
	`mikehemberger/medicinal-plants` dataset and returns the most similar entries.
	""")
	with gr.Row():
		with gr.Column(scale=1):
			mode_txt = gr.Markdown(mode_label())
			inp = gr.Image(type='pil', label='Upload plant image')
			topk = gr.Slider(1, 10, value=5, step=1, label='Top-K results')
			btn = gr.Button('Search')
		with gr.Column(scale=2):
			gallery = gr.Gallery(label='Matches (label crops)', columns=5, height=240)
			text = gr.Textbox(label='Matches details', lines=10)
	btn.click(fn=search_fn, inputs=[inp, topk], outputs=[gallery, text])

	with gr.Accordion('Advanced', open=False):
		with gr.Row():
			mode = gr.Radio(['simple', 'clip'], value=INDEX.mode, label='Index mode')
			provider = gr.Radio(['hf', 'roboflow'], value=INDEX.clip_provider, label='CLIP provider')
			rebuild = gr.Button('Rebuild Index')
		rebuild.click(fn=rebuild_index, inputs=[mode, provider], outputs=[mode_txt])


if __name__ == '__main__':
	demo.launch(server_name='0.0.0.0', server_port=7860)