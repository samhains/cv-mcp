#!/usr/bin/env python3
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
            from transformers import (
                AutoModelForCausalLM,  # type: ignore
                AutoProcessor,  # type: ignore
                AutoConfig,  # type: ignore
            )
            try:
                # Optional: vision2seq is the correct head for many VLMs (e.g., Qwen2.5-VL)
                from transformers import AutoModelForVision2Seq  # type: ignore
            except Exception:  # pragma: no cover
                AutoModelForVision2Seq = None  # type: ignore
            try:
                # Qwen2.5-VL specific model class
                from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
            except Exception:  # pragma: no cover
                Qwen2_5_VLForConditionalGeneration = None  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Local backend requires transformers. Install with `pip install .[local]`."
            ) from e

        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoModelForVision2Seq = AutoModelForVision2Seq if 'AutoModelForVision2Seq' in locals() else None  # type: ignore
        self._Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration if 'Qwen2_5_VLForConditionalGeneration' in locals() else None  # type: ignore
        self._AutoProcessor = AutoProcessor
        self.model_id = model_id

        self.processor = self._AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        # Prefer a suitable head for VLMs; fall back gracefully.
        loaded = False
        # Inspect config to hint correct head
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model_type = getattr(cfg, "model_type", None) or getattr(cfg, "architectures", None)
        except Exception:
            model_type = None

        # Try Qwen2.5-VL specific class first for Qwen models
        if not loaded and self._Qwen2_5_VLForConditionalGeneration and ("qwen" in model_id.lower() or (isinstance(model_type, str) and "qwen" in model_type.lower())):
            try:
                self.model = self._Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,  # type: ignore[arg-type]
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                )
                loaded = True
            except Exception as e:
                loaded = False

        # Try Vision2Seq for Qwen2.5-VL and similar
        if not loaded and self._AutoModelForVision2Seq and (isinstance(model_type, str) and ("qwen2_5_vl" in model_type or "vision2seq" in model_type)):
            try:
                self.model = self._AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,  # type: ignore[arg-type]
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                )
                loaded = True
            except Exception as e:
                loaded = False

        # Try standard CausalLM
        if not loaded:
            try:
                self.model = self._AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,  # type: ignore[arg-type]
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                )
                loaded = True
            except Exception as e:
                loaded = False

        # Final fallback - raise error if nothing worked
        if not loaded:
            raise RuntimeError(
                f"Failed to load local model '{model_id}'. Ensure your transformers version supports this model."
            )

    def _load_image(self, image_ref: Union[str, "Image.Image"]) -> "Image.Image":
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
        return Image.open(image_ref).convert("RGB")

    def caption(
        self,
        image: Union[str, "Image.Image"],
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        img = self._load_image(image)
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
        return out.strip()
