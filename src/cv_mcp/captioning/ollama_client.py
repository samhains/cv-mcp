#!/usr/bin/env python3
from __future__ import annotations

import base64
import requests
from pathlib import Path
from typing import Any, Dict, Optional, Union


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")
        self.chat_url = f"{self.host}/api/chat"

    def _image_to_base64(self, image: str) -> str:
        if image.startswith(("http://", "https://")):
            resp = requests.get(image, timeout=30)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
        data = Path(image).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def analyze_single_image(
        self,
        image: str,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        if not model:
            raise ValueError("Ollama model is required")
        img_b64 = self._image_to_base64(image)

        # Ollama chat format for vision models: message with content string and images list (base64)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": [img_b64]})

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        resp = requests.post(self.chat_url, json=payload, timeout=120)
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        data = resp.json()
        # Response example:
        # { "model": "...", "message": {"role":"assistant","content":"..."}, "done": true }
        content = ""
        if isinstance(data, dict):
            msg = data.get("message") or {}
            content = str(msg.get("content", ""))
        return {"success": True, "content": content, "model": model, "response_data": data}

