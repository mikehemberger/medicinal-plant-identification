import io
import os
import json
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


class PlantDatasetIndex:
	"""
	Lightweight visual search index for `mikehemberger/medicinal-plants` using:
	- Perceptual hash (pHash) for global structure
	- HSV color histogram for color distribution

	This avoids heavy ML dependencies and works on CPU quickly.
	"""

	def __init__(self, alpha_phash_weight: float = 0.6, crop_bottom_ratio: float = 0.15,
				 hist_bins_per_channel: int = 8, resize_for_hist: Tuple[int, int] = (256, 256)):
		self.alpha_phash_weight = alpha_phash_weight
		self.crop_bottom_ratio = crop_bottom_ratio
		self.hist_bins_per_channel = hist_bins_per_channel
		self.resize_for_hist = resize_for_hist

		self.dataset = None
		self.split_name = 'train'
		self.file_ids: List[str] = []
		self.row_indices: List[int] = []
		self.phash_bits: Optional[np.ndarray] = None  # shape [N, 256] bool
		self.hsv_hists: Optional[np.ndarray] = None   # shape [N, B]
		self.meta: Dict[str, str] = {}

	def _load_dataset(self):
		if self.dataset is None:
			self.dataset = load_dataset('mikehemberger/medicinal-plants')

	@staticmethod
	def _to_rgb(image: Image.Image) -> Image.Image:
		if image.mode != 'RGB':
			return image.convert('RGB')
		return image

	def _crop_plant_region(self, image: Image.Image) -> Image.Image:
		"""Crop away the bottom band likely containing the textual label."""
		w, h = image.size
		crop_h = int(h * (1.0 - self.crop_bottom_ratio))
		# Guard against tiny images
		crop_h = max(1, min(crop_h, h))
		return image.crop((0, 0, w, crop_h))

	@staticmethod
	def _extract_label_crop(image: Image.Image, width_ratio: float = 0.7, height_ratio: float = 0.18) -> Image.Image:
		"""Extract the lower-center region likely containing the plant name text.
		width_ratio, height_ratio are relative to the original image dimensions.
		"""
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
		# imagehash returns an ImageHash with a numpy array of booleans internally
		h = imagehash.phash(image, hash_size=hash_size)
		# Convert to 0/1 np array of shape (hash_size*hash_size,)
		bits = np.array(h.hash, dtype=np.bool_).reshape(-1)
		return bits

	def _compute_hsv_hist(self, image: Image.Image) -> np.ndarray:
		# Resize for stability and performance
		img = image.resize(self.resize_for_hist, Image.BICUBIC)
		img = img.convert('HSV')
		arr = np.asarray(img)
		# Compute 3D histogram across H,S,V
		bins = self.hist_bins_per_channel
		hist, edges = np.histogramdd(
			arr.reshape(-1, 3),
			bins=(bins, bins, bins),
			range=((0, 255), (0, 255), (0, 255))
		)
		hist = hist.astype(np.float32).reshape(-1)
		# Normalize to probability distribution
		if hist.sum() > 0:
			hist /= (hist.sum() + 1e-8)
		return hist

	@staticmethod
	def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
		# 1 - cosine similarity
		num = float(np.dot(a, b))
		den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
		return 1.0 - (num / den)

	@staticmethod
	def _hamming_distance(a_bits: np.ndarray, b_bits: np.ndarray) -> float:
		# normalized hamming [0,1]
		assert a_bits.shape == b_bits.shape
		return float(np.count_nonzero(a_bits != b_bits)) / float(a_bits.size)

	def _compute_features_for_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
		img_rgb = self._to_rgb(image)
		plant_region = self._crop_plant_region(img_rgb)
		phash_bits = self._compute_phash_bits(plant_region)
		hsv_hist = self._compute_hsv_hist(plant_region)
		return phash_bits, hsv_hist

	def build(self) -> None:
		self._load_dataset()
		ds = self.dataset[self.split_name]

		phash_list: List[np.ndarray] = []
		hsv_list: List[np.ndarray] = []
		file_ids: List[str] = []
		row_indices: List[int] = []

		for i in tqdm(range(len(ds)), desc='Indexing medicinal plants'):
			row = ds[i]
			img: Image.Image = row['image']
			phash_bits, hsv_hist = self._compute_features_for_image(img)
			phash_list.append(phash_bits)
			hsv_list.append(hsv_hist)
			# Attempt to derive a file_id from dataset's cached filename if present
			file_id = getattr(img, 'filename', f'row_{i}')
			file_ids.append(os.path.basename(str(file_id)))
			row_indices.append(i)

		self.file_ids = file_ids
		self.row_indices = row_indices
		self.phash_bits = np.stack(phash_list, axis=0)
		self.hsv_hists = np.stack(hsv_list, axis=0)

	def save(self, index_path: str) -> None:
		assert self.phash_bits is not None and self.hsv_hists is not None
		payload = {
			'alpha_phash_weight': self.alpha_phash_weight,
			'crop_bottom_ratio': self.crop_bottom_ratio,
			'hist_bins_per_channel': self.hist_bins_per_channel,
			'resize_for_hist': self.resize_for_hist,
			'split_name': self.split_name,
		}
		np.savez_compressed(
			index_path,
			phash_bits=self.phash_bits.astype(np.uint8),  # store as 0/1 uint8
			hsv_hists=self.hsv_hists.astype(np.float32),
			row_indices=np.array(self.row_indices, dtype=np.int32),
			file_ids=np.array(self.file_ids, dtype=object),
			meta_json=json.dumps(payload),
		)

	@classmethod
	def load(cls, index_path: str) -> "PlantDatasetIndex":
		archive = np.load(index_path, allow_pickle=True)
		meta = json.loads(str(archive['meta_json']))
		inst = cls(
			alpha_phash_weight=meta['alpha_phash_weight'],
			crop_bottom_ratio=meta['crop_bottom_ratio'],
			hist_bins_per_channel=int(meta['hist_bins_per_channel']),
			resize_for_hist=tuple(meta['resize_for_hist']),
		)
		inst.split_name = meta.get('split_name', 'train')
		inst.phash_bits = archive['phash_bits'].astype(np.bool_)
		inst.hsv_hists = archive['hsv_hists'].astype(np.float32)
		inst.row_indices = archive['row_indices'].astype(np.int32).tolist()
		inst.file_ids = archive['file_ids'].astype(object).tolist()
		return inst

	def _maybe_ocr_label(self, image: Image.Image) -> Optional[str]:
		"""Try to OCR the label crop using rapidocr_onnxruntime if available."""
		try:
			from rapidocr_onnxruntime import RapidOCR
			r = RapidOCR()
			# RapidOCR expects numpy BGR by default; but it can take PIL via np array
			np_img = np.array(image.convert('RGB'))[:, :, ::-1]  # RGB->BGR
			res, _ = r(np_img)
			if res:
				# res is list of (box, text, score)
				# Join texts with highest scores first
				texts = [x[1] for x in sorted(res, key=lambda t: t[2], reverse=True)]
				return ' '.join(texts).strip()
			return None
		except Exception:
			return None

	def query(self, query_image: Image.Image, top_k: int = 5) -> List[SearchResult]:
		assert self.phash_bits is not None and self.hsv_hists is not None
		self._load_dataset()
		ds = self.dataset[self.split_name]

		# Compute query features (do not crop bottom; query likely has no label)
		img_rgb = self._to_rgb(query_image)
		phash_q = self._compute_phash_bits(img_rgb)
		hist_q = self._compute_hsv_hist(img_rgb)

		# Compute distances
		phash_dists = np.asarray([
			self._hamming_distance(phash_q, x) for x in self.phash_bits
		], dtype=np.float32)
		hist_dists = np.asarray([
			self._cosine_distance(hist_q, x) for x in self.hsv_hists
		], dtype=np.float32)

		alpha = float(self.alpha_phash_weight)
		dists = alpha * phash_dists + (1.0 - alpha) * hist_dists
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


def build_index_cli(index_path: str) -> None:
	idx = PlantDatasetIndex()
	idx.build()
	os.makedirs(os.path.dirname(index_path), exist_ok=True)
	idx.save(index_path)
	print(f"Saved index to: {index_path}")


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Build visual search index for medicinal plants dataset')
	parser.add_argument('--out', type=str, default='app/index.npz', help='Path to save index file')
	args = parser.parse_args()
	build_index_cli(args.out)