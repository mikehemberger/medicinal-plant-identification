import io
import os
import json
import base64
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageOps
from datasets import load_dataset
from tqdm import tqdm
import imagehash


@dataclass
class SearchResult:
	row_index: int
	file_id: str
	distance: float
	predicted_label: Optional[str]
	label_crop: Optional[Image.Image]


class ClipRemoteEmbedder:
	"""
	Remote CLIP embedding client with pluggable providers.
	Supported providers:
	- provider == 'hf': Hugging Face Inference API for a CLIP image encoder
	- provider == 'roboflow': Roboflow Inference CLIP embed endpoint
	"""

	def __init__(self, provider: str = 'hf', hf_model_id: str = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
				 hf_api_token: Optional[str] = None, roboflow_api_key: Optional[str] = None,
				 request_timeout_sec: float = 30.0, max_retries: int = 3, retry_sleep_sec: float = 1.0):
		self.provider = provider
		self.hf_model_id = hf_model_id
		self.hf_api_token = hf_api_token or os.getenv('HF_API_TOKEN')
		self.roboflow_api_key = roboflow_api_key or os.getenv('ROBOFLOW_API_KEY')
		self.request_timeout_sec = request_timeout_sec
		self.max_retries = max_retries
		self.retry_sleep_sec = retry_sleep_sec

		if self.provider == 'hf' and not self.hf_api_token:
			raise ValueError('HF_API_TOKEN is required for provider=hf')
		if self.provider == 'roboflow' and not self.roboflow_api_key:
			raise ValueError('ROBOFLOW_API_KEY is required for provider=roboflow')

	def _pil_to_png_bytes(self, image: Image.Image) -> bytes:
		buf = io.BytesIO()
		image.save(buf, format='PNG')
		return buf.getvalue()

	def _request_with_retries(self, fn):
		last_exc = None
		for _ in range(max(1, self.max_retries)):
			try:
				return fn()
			except Exception as exc:
				last_exc = exc
				time.sleep(self.retry_sleep_sec)
		if last_exc is not None:
			raise last_exc
		return None

	def embed_image(self, image: Image.Image) -> np.ndarray:
		if self.provider == 'hf':
			return self._embed_image_hf(image)
		elif self.provider == 'roboflow':
			return self._embed_image_roboflow(image)
		else:
			raise ValueError(f'Unknown provider: {self.provider}')

	def _embed_image_hf(self, image: Image.Image) -> np.ndarray:
		import requests
		png_bytes = self._pil_to_png_bytes(image)
		url = f'https://api-inference.huggingface.co/models/{self.hf_model_id}'
		headers = {
			'Authorization': f'Bearer {self.hf_api_token}',
			'Accept': 'application/json',
			'Content-Type': 'image/png',
		}

		def do_request():
			resp = requests.post(url, headers=headers, data=png_bytes, timeout=self.request_timeout_sec)
			resp.raise_for_status()
			return resp.json()

		data = self._request_with_retries(do_request)
		# data may be nested lists; pool to a single vector
		arr = np.array(data, dtype=np.float32)
		# Shapes can vary: [1, T, D] or [T, D] or [D]
		if arr.ndim == 3:
			vec = arr.mean(axis=(0, 1))
		elif arr.ndim == 2:
			vec = arr.mean(axis=0)
		else:
			vec = arr
		# L2 normalize
		norm = np.linalg.norm(vec) + 1e-8
		return (vec / norm).astype(np.float32)

	def _embed_image_roboflow(self, image: Image.Image) -> np.ndarray:
		import requests
		# Prefer base64 to avoid hosting URLs
		png_bytes = self._pil_to_png_bytes(image)
		b64 = base64.b64encode(png_bytes).decode('utf-8')
		payload = {
			'image': {
				'type': 'base64',
				'value': f'data:image/png;base64,{b64}',
			}
		}
		url = f'https://infer.roboflow.com/clip/embed_image?api_key={self.roboflow_api_key}'

		def do_request():
			resp = requests.post(url, json=payload, timeout=self.request_timeout_sec)
			resp.raise_for_status()
			return resp.json()

		data = self._request_with_retries(do_request)
		emb = data.get('embeddings') or data.get('embedding')
		if emb is None:
			raise RuntimeError('Roboflow response missing embeddings')
		vec = np.array(emb, dtype=np.float32).reshape(-1)
		norm = np.linalg.norm(vec) + 1e-8
		return (vec / norm).astype(np.float32)


class PlantDatasetIndex:
	"""
	Lightweight visual search index for `mikehemberger/medicinal-plants`.

	Modes:
	- simple: pHash + HSV histogram (no heavy dependencies)
	- clip: remote CLIP embeddings via HF or Roboflow (requires API key)
	"""

	def __init__(self, alpha_phash_weight: float = 0.6, crop_bottom_ratio: float = 0.15,
				 hist_bins_per_channel: int = 8, resize_for_hist: Tuple[int, int] = (256, 256),
				 mode: str = 'simple', clip_provider: str = 'hf', hf_model_id: str = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
		self.alpha_phash_weight = alpha_phash_weight
		self.crop_bottom_ratio = crop_bottom_ratio
		self.hist_bins_per_channel = hist_bins_per_channel
		self.resize_for_hist = resize_for_hist

		self.mode = mode  # 'simple' or 'clip'
		self.clip_provider = clip_provider
		self.hf_model_id = hf_model_id
		self.clip_client: Optional[ClipRemoteEmbedder] = None

		self.dataset = None
		self.split_name = 'train'
		self.file_ids: List[str] = []
		self.row_indices: List[int] = []
		self.phash_bits: Optional[np.ndarray] = None  # shape [N, H] bool
		self.hsv_hists: Optional[np.ndarray] = None   # shape [N, B]
		self.clip_vecs: Optional[np.ndarray] = None   # shape [N, D]
		self.meta: Dict[str, str] = {}

	def _load_dataset(self):
		if self.dataset is None:
			self.dataset = load_dataset('mikehemberger/medicinal-plants')

	def _ensure_clip_client(self):
		if self.clip_client is not None:
			return
		if self.clip_provider == 'hf':
			self.clip_client = ClipRemoteEmbedder(provider='hf', hf_model_id=self.hf_model_id)
		elif self.clip_provider == 'roboflow':
			self.clip_client = ClipRemoteEmbedder(provider='roboflow')
		else:
			raise ValueError(f'Unsupported clip_provider: {self.clip_provider}')

	@staticmethod
	def _to_rgb(image: Image.Image) -> Image.Image:
		if image.mode != 'RGB':
			return image.convert('RGB')
		return image

	def _crop_plant_region(self, image: Image.Image) -> Image.Image:
		"""Crop away the bottom band likely containing the textual label."""
		w, h = image.size
		crop_h = int(h * (1.0 - self.crop_bottom_ratio))
		crop_h = max(1, min(crop_h, h))
		return image.crop((0, 0, w, crop_h))

	@staticmethod
	def _extract_label_crop(image: Image.Image, width_ratio: float = 0.7, height_ratio: float = 0.18) -> Image.Image:
		w, h = image.size
		crop_h = int(h * height_ratio)
		crop_w = int(w * width_ratio)
		left = int((w - crop_w) / 2)
		right = left + crop_w
		top = h - crop_h
		bottom = h
		return image.crop((left, top, right, bottom))

	@staticmethod
	def _compute_phash_bits(image: Image.Image, hash_size: int = 16) -> np.ndarray:
		h = imagehash.phash(image, hash_size=hash_size)
		bits = np.array(h.hash, dtype=np.bool_).reshape(-1)
		return bits

	def _compute_hsv_hist(self, image: Image.Image) -> np.ndarray:
		img = image.resize(self.resize_for_hist, Image.BICUBIC)
		img = img.convert('HSV')
		arr = np.asarray(img)
		bins = self.hist_bins_per_channel
		hist, _ = np.histogramdd(
			arr.reshape(-1, 3),
			bins=(bins, bins, bins),
			range=((0, 255), (0, 255), (0, 255))
		)
		hist = hist.astype(np.float32).reshape(-1)
		if hist.sum() > 0:
			hist /= (hist.sum() + 1e-8)
		return hist

	@staticmethod
	def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
		num = float(np.dot(a, b))
		den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
		return 1.0 - (num / den)

	@staticmethod
	def _hamming_distance(a_bits: np.ndarray, b_bits: np.ndarray) -> float:
		assert a_bits.shape == b_bits.shape
		return float(np.count_nonzero(a_bits != b_bits)) / float(a_bits.size)

	def _compute_features_for_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
		img_rgb = self._to_rgb(image)
		plant_region = self._crop_plant_region(img_rgb)
		phash_bits = self._compute_phash_bits(plant_region)
		hsv_hist = self._compute_hsv_hist(plant_region)
		return phash_bits, hsv_hist

	def _compute_clip_for_image(self, image: Image.Image) -> np.ndarray:
		self._ensure_clip_client()
		img_rgb = self._to_rgb(image)
		plant_region = self._crop_plant_region(img_rgb)
		return self.clip_client.embed_image(plant_region)

	def build(self) -> None:
		self._load_dataset()
		ds = self.dataset[self.split_name]

		file_ids: List[str] = []
		row_indices: List[int] = []

		if self.mode == 'simple':
			phash_list: List[np.ndarray] = []
			hsv_list: List[np.ndarray] = []
			for i in tqdm(range(len(ds)), desc='Indexing (simple)'):
				row = ds[i]
				img: Image.Image = row['image']
				phash_bits, hsv_hist = self._compute_features_for_image(img)
				phash_list.append(phash_bits)
				hsv_list.append(hsv_hist)
				file_id = getattr(img, 'filename', f'row_{i}')
				file_ids.append(os.path.basename(str(file_id)))
				row_indices.append(i)

			self.phash_bits = np.stack(phash_list, axis=0)
			self.hsv_hists = np.stack(hsv_list, axis=0)

		elif self.mode == 'clip':
			clip_vecs: List[np.ndarray] = []
			for i in tqdm(range(len(ds)), desc=f'Indexing (CLIP:{self.clip_provider})'):
				row = ds[i]
				img: Image.Image = row['image']
				vec = self._compute_clip_for_image(img)
				clip_vecs.append(vec.astype(np.float32))
				file_id = getattr(img, 'filename', f'row_{i}')
				file_ids.append(os.path.basename(str(file_id)))
				row_indices.append(i)

			self.clip_vecs = np.stack(clip_vecs, axis=0)
		else:
			raise ValueError(f'Unknown mode: {self.mode}')

		self.file_ids = file_ids
		self.row_indices = row_indices

	def save(self, index_path: str) -> None:
		payload = {
			'mode': self.mode,
			'alpha_phash_weight': self.alpha_phash_weight,
			'crop_bottom_ratio': self.crop_bottom_ratio,
			'hist_bins_per_channel': self.hist_bins_per_channel,
			'resize_for_hist': self.resize_for_hist,
			'split_name': self.split_name,
			'clip_provider': self.clip_provider,
			'hf_model_id': self.hf_model_id,
		}
		kwargs = {
			'row_indices': np.array(self.row_indices, dtype=np.int32),
			'file_ids': np.array(self.file_ids, dtype=object),
			'meta_json': json.dumps(payload),
		}
		if self.mode == 'simple':
			assert self.phash_bits is not None and self.hsv_hists is not None
			kwargs.update({
				'phash_bits': self.phash_bits.astype(np.uint8),
				'hsv_hists': self.hsv_hists.astype(np.float32),
			})
		elif self.mode == 'clip':
			assert self.clip_vecs is not None
			kwargs.update({'clip_vecs': self.clip_vecs.astype(np.float32)})
		else:
			raise ValueError(f'Unknown mode: {self.mode}')
		np.savez_compressed(index_path, **kwargs)

	@classmethod
	def load(cls, index_path: str) -> "PlantDatasetIndex":
		archive = np.load(index_path, allow_pickle=True)
		meta = json.loads(str(archive['meta_json']))
		inst = cls(
			mode=meta.get('mode', 'simple'),
			alpha_phash_weight=meta.get('alpha_phash_weight', 0.6),
			crop_bottom_ratio=meta.get('crop_bottom_ratio', 0.15),
			hist_bins_per_channel=int(meta.get('hist_bins_per_channel', 8)),
			resize_for_hist=tuple(meta.get('resize_for_hist', (256, 256))),
			clip_provider=meta.get('clip_provider', 'hf'),
			hf_model_id=meta.get('hf_model_id', 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'),
		)
		inst.split_name = meta.get('split_name', 'train')
		inst.row_indices = archive['row_indices'].astype(np.int32).tolist()
		inst.file_ids = archive['file_ids'].astype(object).tolist()
		if inst.mode == 'simple':
			inst.phash_bits = archive['phash_bits'].astype(np.bool_)
			inst.hsv_hists = archive['hsv_hists'].astype(np.float32)
		elif inst.mode == 'clip':
			inst.clip_vecs = archive['clip_vecs'].astype(np.float32)
		else:
			raise ValueError(f'Unknown mode in index: {inst.mode}')
		return inst

	def _maybe_ocr_label(self, image: Image.Image) -> Optional[str]:
		try:
			from rapidocr_onnxruntime import RapidOCR
			r = RapidOCR()
			np_img = np.array(image.convert('RGB'))[:, :, ::-1]
			res, _ = r(np_img)
			if res:
				texts = [x[1] for x in sorted(res, key=lambda t: t[2], reverse=True)]
				return ' '.join(texts).strip()
			return None
		except Exception:
			return None

	def query(self, query_image: Image.Image, top_k: int = 5) -> List[SearchResult]:
		self._load_dataset()
		ds = self.dataset[self.split_name]
		img_rgb = self._to_rgb(query_image)

		if self.mode == 'simple':
			assert self.phash_bits is not None and self.hsv_hists is not None
			phash_q = self._compute_phash_bits(img_rgb)
			hist_q = self._compute_hsv_hist(img_rgb)
			phash_dists = np.asarray([
				self._hamming_distance(phash_q, x) for x in self.phash_bits
			], dtype=np.float32)
			hist_dists = np.asarray([
				self._cosine_distance(hist_q, x) for x in self.hsv_hists
			], dtype=np.float32)
			alpha = float(self.alpha_phash_weight)
			dists = alpha * phash_dists + (1.0 - alpha) * hist_dists
		elif self.mode == 'clip':
			assert self.clip_vecs is not None
			self._ensure_clip_client()
			vec_q = self.clip_client.embed_image(img_rgb)
			# clip_vecs are expected L2-normalized
			dists = np.asarray([
				self._cosine_distance(vec_q, x) for x in self.clip_vecs
			], dtype=np.float32)
		else:
			raise ValueError(f'Unknown mode: {self.mode}')

		idxs = np.argsort(dists)[:max(1, top_k)]
		results: List[SearchResult] = []
		for rank_idx in idxs:
			row_idx = self.row_indices[rank_idx]
			img: Image.Image = ds[row_idx]['image']
			label_crop = self._extract_label_crop(img)
			pred_label = self._maybe_ocr_label(label_crop)
			results.append(
				SearchResult(
					row_index=row_idx,
					file_id=self.file_ids[rank_idx],
					distance=float(dists[rank_idx]),
					predicted_label=pred_label,
					label_crop=label_crop,
				)
			)
		return results


def build_index_cli(index_path: str, mode: str = 'simple', clip_provider: str = 'hf', hf_model_id: str = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K') -> None:
	idx = PlantDatasetIndex(mode=mode, clip_provider=clip_provider, hf_model_id=hf_model_id)
	idx.build()
	os.makedirs(os.path.dirname(index_path), exist_ok=True)
	idx.save(index_path)
	print(f"Saved index to: {index_path}")


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Build visual search index for medicinal plants dataset')
	parser.add_argument('--out', type=str, default='app/index.npz', help='Path to save index file')
	parser.add_argument('--mode', type=str, default='simple', choices=['simple', 'clip'], help='Index mode')
	parser.add_argument('--clip-provider', type=str, default='hf', choices=['hf', 'roboflow'], help='Remote CLIP provider')
	parser.add_argument('--hf-model-id', type=str, default='laion/CLIP-ViT-B-32-laion2B-s34B-b79K', help='HF model id (when provider=hf)')
	args = parser.parse_args()
	build_index_cli(args.out, mode=args.mode, clip_provider=args.clip_provider, hf_model_id=args.hf_model_id)