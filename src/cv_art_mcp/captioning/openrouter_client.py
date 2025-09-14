#!/usr/bin/env python3
"""
OpenRouter client for image analysis using Claude Sonnet 3.5
Supports both URL and base64 image inputs for multimodal AI processing
"""
import os
import json
import time
import logging
import base64
import requests
import tempfile
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-3.5-sonnet"):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key (defaults to env var OPENROUTER_API_KEY)
            model: Model to use (defaults to Claude Sonnet 3.5)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def download_and_encode_image(self, image_url: str) -> str:
        """
        Download image from URL and encode to base64 data URL
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Base64 data URL string
        """
        try:
            # Download the image with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(image_url, timeout=30, headers=headers)
            response.raise_for_status()
            
            # Determine MIME type from Content-Type header or URL extension
            content_type = response.headers.get('content-type', '')
            if 'image/' in content_type:
                mime_type = content_type
            else:
                # Fallback to extension-based detection
                ext = Path(image_url).suffix.lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg', 
                    '.png': 'image/png',
                    '.webp': 'image/webp',
                    '.gif': 'image/gif'
                }.get(ext, 'image/jpeg')
            
            # Encode to base64
            base64_image = base64.b64encode(response.content).decode('utf-8')
            return f"data:{mime_type};base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"Failed to download and encode image from {image_url}: {e}")
            raise

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode local image file to base64 data URL
        
        Args:
            image_path: Path to local image file
            
        Returns:
            Base64 data URL string
        """
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Determine MIME type from file extension
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }.get(ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{base64_image}"
    
    def analyze_images(self, 
                      images: List[Union[str, Dict]], 
                      prompt: str,
                      max_retries: int = 3,
                      retry_delay: float = 1.0) -> Dict[str, Any]:
        """
        Analyze images with OpenRouter API
        
        Args:
            images: List of image URLs or local file paths
            prompt: Text prompt for analysis
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            API response data
        """
        # Build content array with text prompt first, then images
        content = [{"type": "text", "text": prompt}]
        
        # Add images to content (download and encode all images to base64)
        for image in images:
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    # URL image - download and encode to base64
                    data_url = self.download_and_encode_image(image)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                else:
                    # Local file path
                    data_url = self.encode_image_to_base64(image)
                    content.append({
                        "type": "image_url", 
                        "image_url": {"url": data_url}
                    })
            elif isinstance(image, dict) and "url" in image:
                # Already formatted image dict - check if it's a URL or base64
                image_url = image["url"]
                if image_url.startswith(('http://', 'https://')):
                    # Download and encode to base64
                    data_url = self.download_and_encode_image(image_url)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                else:
                    # Assume it's already base64
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to OpenRouter (attempt {attempt + 1}/{max_retries})")
                logger.info(f"Model: {self.model}, Images: {len(images)}")
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info("OpenRouter request successful")
                    return {
                        "success": True,
                        "content": data["choices"][0]["message"]["content"],
                        "model": self.model,
                        "usage": data.get("usage", {}),
                        "response_data": data
                    }
                else:
                    logger.warning(f"OpenRouter API error: {response.status_code} - {response.text}")
                    if response.status_code == 429:  # Rate limit
                        wait_time = retry_delay * (2 ** attempt)
                        logger.info(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    elif attempt == max_retries - 1:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}: {response.text}",
                            "status_code": response.status_code
                        }
                        
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    return {
                        "success": False,
                        "error": f"Request failed: {str(e)}"
                    }
                    
        return {
            "success": False,
            "error": "Max retries exceeded"
        }
    
    def analyze_single_image(self, image: Union[str, Dict], prompt: str) -> Dict[str, Any]:
        """
        Analyze a single image (convenience method)
        
        Args:
            image: Image URL or local file path
            prompt: Text prompt for analysis
            
        Returns:
            API response data
        """
        return self.analyze_images([image], prompt)
