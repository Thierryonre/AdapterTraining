"""Find objects in an image from text ("water bottle") and return their masks.

Sam3Segmenter   SAM 3 through Hugging Face transformers (GPU recommended). The checkpoint is gated:
                request access at https://huggingface.co/facebook/sam3 and `hf auth login` first.
MaskDirSegmenter  pre-made masks (<dir>/<object name>.png) -- for tests, or to fix a bad mask by hand.

Not tested on this machine (no GPU, gated checkpoint): check the perception overlay images.
"""

from pathlib import Path

import cv2
import numpy as np


class Sam3Segmenter:
    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None, threshold: float = 0.4):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.threshold = threshold

    def __call__(self, image_rgb: np.ndarray, prompts: list[str]) -> dict[str, tuple[np.ndarray, float]]:
        """{prompt: (mask HxW bool, score)} for the highest-scoring instance of each prompt found."""
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
            if len(result["scores"]) == 0:
                continue
            best = int(result["scores"].argmax())
            found[prompt] = (result["masks"][best].cpu().numpy().astype(bool), float(result["scores"][best]))
        return found


class MaskDirSegmenter:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def __call__(self, image_rgb: np.ndarray, prompts: list[str]) -> dict[str, tuple[np.ndarray, float]]:
        found = {}
        for prompt in prompts:
            path = self.directory / f"{prompt.replace(' ', '_')}.png"
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
                if mask.shape != image_rgb.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), image_rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                found[prompt] = (mask, 1.0)
        return found
