#!/usr/bin/env python3
import argparse
import os
import json
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

from cv_mcp.metadata.runner import (
    run_alt_text,
    run_structured_json,
)

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _default_schema_path() -> Path:
    # Use the schema file shipped with the package
    from cv_mcp.metadata import runner as md_runner  # type: ignore
    return Path(md_runner.__file__).with_name("schema.json")


def main():
    p = argparse.ArgumentParser(description="Run the 3-step image metadata pipeline and print JSON")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-url", help="HTTP/HTTPS URL of the image")
    g.add_argument("--file-path", help="Local file path to the image")

    p.add_argument("--caption-override", help="Provide an existing dense caption to skip the vision caption step")
    p.add_argument("--config-path", default=None, help="Path to model config JSON (defaults to packaged config)")
    p.add_argument("--schema-path", default=None, help="Path to schema.json (defaults to packaged schema)")
    p.add_argument("--mode", choices=["double", "triple"], default="double", help="Pipeline mode: double (vision alt+caption + text metadata) or triple (vision alt+caption + vision metadata)")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")

    args = p.parse_args()
    image_ref = args.image_url or args.file_path  # type: ignore

    try:
        # Early env check for clearer error messages
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your environment or a .env file.")

        if args.caption_override:
            # If a config is provided, use it to select models for steps as applicable
            schema_path = Path(args.schema_path) if args.schema_path else _default_schema_path()
            cfg_path = Path(args.config_path) if args.config_path else None
            if args.mode == "double":
                # Text-only metadata from provided caption
                from cv_mcp.metadata.runner import run_metadata_from_caption
                if cfg_path and cfg_path.exists():
                    import json as _json
                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    meta = run_metadata_from_caption(args.caption_override, schema_path=schema_path, model=cfg.get("meta_text_model"))
                else:
                    meta = run_metadata_from_caption(args.caption_override, schema_path=schema_path)
                alt = run_alt_text(image_ref)
            else:
                # Triple: Vision+caption metadata
                if cfg_path and cfg_path.exists():
                    import json as _json
                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    meta = run_structured_json(image_ref, args.caption_override, schema_path=schema_path, model=cfg.get("meta_vision_model"))
                else:
                    meta = run_structured_json(image_ref, args.caption_override, schema_path=schema_path)
                alt = run_alt_text(image_ref)
            out = {"alt_text": alt, "caption": args.caption_override, "metadata": meta}
        else:
            schema_path = Path(args.schema_path) if args.schema_path else _default_schema_path()
            cfg_path = Path(args.config_path) if args.config_path else None
            if args.mode == "double":
                from cv_mcp.metadata.runner import run_pipeline_double
                out = run_pipeline_double(image_ref, config_path=cfg_path, schema_path=schema_path)
            else:
                from cv_mcp.metadata.runner import run_pipeline_triple
                out = run_pipeline_triple(image_ref, config_path=cfg_path, schema_path=schema_path)

        print(json.dumps(out, indent=args.indent))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
