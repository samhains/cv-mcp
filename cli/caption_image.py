#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Ensure local src/ is on sys.path when running from repo without installing
try:
    import cv_mcp  # type: ignore
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

from cv_mcp.captioning.openrouter_client import OpenRouterClient

DEFAULT_PROMPT = (
    "Write a concise, vivid caption for this image. "
    "Describe key subjects, scene, and mood in 1-2 sentences."
)


def main():
    # Load .env if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Caption a single image (OpenRouter or local backend)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-url", help="HTTP/HTTPS URL of the image")
    g.add_argument("--file-path", help="Local file path to the image")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--backend", choices=["openrouter", "local"], default=None)
    p.add_argument("--local-model-id", default=None)
    args = p.parse_args()

    image_ref = args.image_url or args.file_path  # type: ignore

    # Resolve defaults from global config
    try:
        from cv_mcp.metadata.runner import _CFG as _GLOBAL_CFG  # type: ignore
    except Exception:
        _GLOBAL_CFG = {}
    backend = (args.backend or str(_GLOBAL_CFG.get("caption_backend", "openrouter"))).lower()
    local_model_id = args.local_model_id or str(_GLOBAL_CFG.get("local_vlm_id", "Qwen/Qwen2-VL-2B-Instruct"))

    if backend == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            print("Error: OPENROUTER_API_KEY is not set. Add it to your environment or a .env file.", file=sys.stderr)
            sys.exit(1)
        client = OpenRouterClient()
        res = client.analyze_single_image(image_ref, args.prompt)
        if not res.get("success"):
            print(f"Error: {res.get('error')}", file=sys.stderr)
            sys.exit(1)
        print(res["content"])  # caption text
        return

    # Local backend
    try:
        from cv_mcp.captioning.local_captioner import LocalCaptioner
    except Exception:
        print(
            "Local backend not available. Install optional deps with `pip install .[local]`.",
            file=sys.stderr,
        )
        sys.exit(1)

    local = LocalCaptioner(model_id=local_model_id)
    out = local.caption(image_ref, args.prompt)
    print(out)


if __name__ == "__main__":
    main()
