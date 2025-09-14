#!/usr/bin/env python3
import os
import json
import time
import logging
import base64
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        self.model = model or "google/gemini-2.5-flash"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def download_and_encode_image(self, image_url: str) -> str:
        try:
            headers = { 'User-Agent': 'Mozilla/5.0' }
            response = requests.get(image_url, timeout=30, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if 'image/' in content_type:
                mime_type = content_type
            else:
                ext = Path(image_url).suffix.lower()
                mime_type = {
                    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                    '.webp': 'image/webp', '.gif': 'image/gif'
                }.get(ext, 'image/jpeg')
            base64_image = base64.b64encode(response.content).decode('utf-8')
            return f"data:{mime_type};base64,{base64_image}"
        except Exception as e:
            logger.error(f"Failed to download and encode image from {image_url}: {e}")
            raise

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.webp': 'image/webp', '.gif': 'image/gif'
        }.get(ext, 'image/jpeg')
        return f"data:{mime_type};base64,{base64_image}"

    def analyze_images(self, 
                      images: List[Union[str, Dict]], 
                      prompt: str,
                      max_retries: int = 3,
                      retry_delay: float = 1.0,
                      model: Optional[str] = None,
                      system: Optional[str] = None) -> Dict[str, Any]:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    data_url = self.download_and_encode_image(image)
                    content.append({"type": "image_url","image_url": {"url": data_url}})
                else:
                    data_url = self.encode_image_to_base64(image)
                    content.append({"type": "image_url","image_url": {"url": data_url}})
            elif isinstance(image, dict) and "url" in image:
                image_url = image["url"]
                if image_url.startswith(('http://', 'https://')):
                    data_url = self.download_and_encode_image(image_url)
                    content.append({"type": "image_url","image_url": {"url": data_url}})
                else:
                    content.append({"type": "image_url","image_url": {"url": image_url}})

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model or self.model,
            "messages": messages,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "content": data["choices"][0]["message"]["content"],
                        "model": payload["model"],
                        "usage": data.get("usage", {}),
                        "response_data": data
                    }
                if response.status_code == 429 and attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}", "status_code": response.status_code}
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    return {"success": False, "error": f"Request failed: {str(e)}"}

    def analyze_single_image(self, image: Union[str, Dict], prompt: str, *, model: Optional[str] = None, system: Optional[str] = None) -> Dict[str, Any]:
        return self.analyze_images([image], prompt, model=model, system=system)

    def chat(self, *, messages: List[Dict[str, Any]], model: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
        payload = {"model": model or self.model, "messages": messages}
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    return {"success": True, "content": data["choices"][0]["message"]["content"], "model": payload["model"], "usage": data.get("usage", {}), "response_data": data}
                if response.status_code == 429 and attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}", "status_code": response.status_code}
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    return {"success": False, "error": f"Request failed: {str(e)}"}

