#!/usr/bin/env python3
"""
Local image captioning using a vision-language model (e.g., Qwen2-VL Instruct).
This module is optional and only imported when `backend="local"` is requested.

You must install the optional dependencies first, e.g.:
  pip install .[local]

And have the model available locally or via your configured HF cache.
Default model: Qwen/Qwen2-VL-2B-Instruct
"""
from __future__ import annotations

from typing import Optional, Union
import io

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None  # type: ignore

import requests


class LocalCaptioner:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype: Optional[str] = "auto",
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Local backend requires transformers. Install with `pip install .[local]`."
            ) from e

        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoProcessor = AutoProcessor
        self.model_id = model_id

        # Load processor and model (expects model to be available locally/in cache)
        self.processor = self._AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = self._AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,  # type: ignore[arg-type]
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    def _load_image(self, image_ref: Union[str, Image.Image]) -> Image.Image:
        if Image is None:  # pragma: no cover
            raise RuntimeError(
                "Local backend requires Pillow. Install with `pip install .[local]`."
            )
        if isinstance(image_ref, Image.Image):
            return image_ref
        if isinstance(image_ref, str) and image_ref.startswith(("http://", "https://")):
            resp = requests.get(image_ref, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        # assume local path
        return Image.open(image_ref).convert("RGB")

    def caption(
        self,
        image: Union[str, "Image.Image"],
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate a caption for the given image using the local model."""
        img = self._load_image(image)

        # Qwen2-VL chat template style
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        out = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        # Heuristic: processor already returns the assistant text. If not, try to slice after last 'assistant\n\n'
        return out.strip()

