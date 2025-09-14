#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from cv_mcp.captioning.openrouter_client import OpenRouterClient
from cv_mcp.metadata.runner import (
    run_alt_text,
    run_dense_caption,
    run_structured_json,
    run_pipeline_double,
    run_pipeline_triple,
)

DEFAULT_PROMPT = (
    "Write a concise, vivid caption for this image. "
    "Describe key subjects, scene, and mood in 1-2 sentences."
)

try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:
    raise RuntimeError(
        "The 'mcp' package is required. Install with `pip install mcp`."
    ) from e

mcp = FastMCP("cv-mcp")


@mcp.tool()
def caption_image(
    image_url: Optional[str] = None,
    file_path: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    backend: str = "openrouter",
    local_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
) -> str:
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
        try:
            from cv_mcp.captioning.local_captioner import LocalCaptioner
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Local backend not available. Install optional deps with `pip install .[local]`."
            ) from e
        local = LocalCaptioner(model_id=local_model_id)
        return local.caption(image_ref, prompt)
    else:
        raise ValueError("Invalid backend. Use 'openrouter' or 'local'.")


@mcp.tool()
def alt_text(
    image_url: Optional[str] = None,
    file_path: Optional[str] = None,
    max_words: int = 20,
) -> str:
    if not image_url and not file_path:
        raise ValueError("Provide either image_url or file_path")
    if image_url and file_path:
        raise ValueError("Provide only one of image_url or file_path, not both")
    image_ref = image_url or file_path  # type: ignore
    return run_alt_text(image_ref, max_words=max_words)


@mcp.tool()
def dense_caption(
    image_url: Optional[str] = None,
    file_path: Optional[str] = None,
) -> str:
    if not image_url and not file_path:
        raise ValueError("Provide either image_url or file_path")
    if image_url and file_path:
        raise ValueError("Provide only one of image_url or file_path, not both")
    image_ref = image_url or file_path  # type: ignore
    return run_dense_caption(image_ref)


@mcp.tool()
def image_metadata(
    image_url: Optional[str] = None,
    file_path: Optional[str] = None,
    caption_override: Optional[str] = None,
    config_path: Optional[str] = None,
    mode: str = "double",
) -> dict:
    if not image_url and not file_path:
        raise ValueError("Provide either image_url or file_path")
    if image_url and file_path:
        raise ValueError("Provide only one of image_url or file_path, not both")
    image_ref = image_url or file_path  # type: ignore

    if caption_override:
        schema_path = os.path.join(os.path.dirname(__file__), "metadata", "schema.json")
        if mode == "double":
            # Text-only metadata from provided caption
            from cv_mcp.metadata.runner import run_metadata_from_caption
            meta = run_metadata_from_caption(caption_override, schema_path=schema_path)
            alt = run_alt_text(image_ref)
            return {"alt_text": alt, "caption": caption_override, "metadata": meta}
        elif mode == "triple":
            # Vision+caption metadata
            meta = run_structured_json(image_ref, caption_override, schema_path=schema_path)
            alt = run_alt_text(image_ref)
            return {"alt_text": alt, "caption": caption_override, "metadata": meta}
        else:
            raise ValueError("mode must be 'double' or 'triple'")

    if mode == "double":
        return run_pipeline_double(
            image_ref,
            config_path=config_path,
            schema_path=os.path.join(os.path.dirname(__file__), "metadata", "schema.json"),
        )
    elif mode == "triple":
        return run_pipeline_triple(
            image_ref,
            config_path=config_path,
            schema_path=os.path.join(os.path.dirname(__file__), "metadata", "schema.json"),
        )
    else:
        raise ValueError("mode must be 'double' or 'triple'")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
