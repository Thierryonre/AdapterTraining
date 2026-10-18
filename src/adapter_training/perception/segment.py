"""Find objects in an image from text ("water bottle") and return their masks.

Sam3Segmenter   SAM 3 through Hugging Face transformers (GPU recommended). The checkpoint is gated:
                request access at https://huggingface.co/facebook/sam3 and `hf auth login` first.
MaskDirSegmenter  pre-made masks (<dir>/<object name>.png, or <name>_1.png, <name>_2.png, ... for several
                  identical objects) -- for tests, or to fix a bad mask by hand.

Both return {prompt: [(mask HxW bool, score), ...]}: the distinct instances found, best first.

Not tested on this machine (no GPU, gated checkpoint): check the perception overlay images.
"""

from pathlib import Path

import cv2
import numpy as np


def distinct(instances: list[tuple[np.ndarray, float]], max_iou: float = 0.5) -> list[tuple[np.ndarray, float]]:
    """Best-first instances, dropping any that mostly overlap a better one (duplicate detections)."""
    kept = []
    for mask, score in sorted(instances, key=lambda m: -m[1]):
        if all((mask & k).sum() / max(1, (mask | k).sum()) <= max_iou for k, _ in kept):
            kept.append((mask, score))
    return kept


class Sam3Segmenter:
    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None, threshold: float = 0.4):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.threshold = threshold

    def __call__(self, image_rgb: np.ndarray, prompts: list[str]) -> dict[str, list[tuple[np.ndarray, float]]]:
        """{prompt: [(mask HxW bool, score), ...]}: every distinct instance found, best first."""
        from PIL import Image

        image = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        found = {}
        for prompt in prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold, mask_threshold=0.5, target_sizes=[(h, w)]
            )[0]
            instances = [(m.cpu().numpy().astype(bool), float(sc)) for m, sc in zip(result["masks"], result["scores"])]
            if instances:
                found[prompt] = distinct(instances)
        return found


class MaskDirSegmenter:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def __call__(self, image_rgb: np.ndarray, prompts: list[str]) -> dict[str, list[tuple[np.ndarray, float]]]:
        found = {}
        for prompt in prompts:
            stem = prompt.replace(" ", "_")
            paths = [self.directory / f"{stem}.png"] + sorted(self.directory.glob(f"{stem}_[0-9]*.png"))
            masks = []
            for path in paths:
                if path.exists():
                    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
                    if mask.shape != image_rgb.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), image_rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                    masks.append((mask, 1.0))
            if masks:
                found[prompt] = masks
        return found
