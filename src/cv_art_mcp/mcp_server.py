#!/usr/bin/env python3
"""
Minimal MCP server exposing a single image recognition (captioning) tool.

Tool: caption_image
- Inputs: `image_url` (http/https) or `file_path` (local), optional `prompt`.
- Optional: `backend` = "openrouter" | "local" (default: openrouter), `local_model_id` (default: Qwen/Qwen2-VL-2B-Instruct)
- Output: caption text

Requires env var: OPENROUTER_API_KEY
"""
from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from cv_art_mcp.captioning.openrouter_client import OpenRouterClient

# Default prompt mirrors the CLI
DEFAULT_PROMPT = (
    "Write a concise, vivid caption for this image. "
    "Describe key subjects, scene, and mood in 1-2 sentences."
)


# Prefer the higher-level FastMCP API if available
try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:
    raise RuntimeError(
        "The 'mcp' package is required. Install with `pip install mcp`."
    ) from e


mcp = FastMCP("cv-art-mcp")


@mcp.tool()
def caption_image(
    image_url: Optional[str] = None,
    file_path: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    backend: str = "openrouter",
    local_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
) -> str:
    """
    Generate a concise caption for an image.

    Provide either `image_url` (http/https) or `file_path` (local path).
    Choose backend with `backend` param: "openrouter" (default) or "local".
    Returns caption text.
    """
    if not image_url and not file_path:
        raise ValueError("Provide either image_url or file_path")
    if image_url and file_path:
        raise ValueError("Provide only one of image_url or file_path, not both")

    image_ref = image_url or file_path  # type: ignore

    if backend.lower() == "openrouter":
        client = OpenRouterClient()
        res = client.analyze_single_image(image_ref, prompt)
        if not res.get("success"):
            raise RuntimeError(str(res.get("error", "Captioning failed")))
        content = res.get("content", "")
        return str(content)
    elif backend.lower() == "local":
        # Lazy import to avoid hard deps when not using local
        try:
            from cv_art_mcp.captioning.local_captioner import LocalCaptioner
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Local backend not available. Install optional deps with `pip install .[local]`."
            ) from e
        local = LocalCaptioner(model_id=local_model_id)
        return local.caption(image_ref, prompt)
    else:
        raise ValueError("Invalid backend. Use 'openrouter' or 'local'.")


def main() -> None:
    # Runs a stdio MCP server. Compatible with most MCP clients (e.g., Claude Desktop).
    mcp.run()


if __name__ == "__main__":
    main()
