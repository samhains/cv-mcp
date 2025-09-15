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
    p = argparse.ArgumentParser(description="Run the image metadata pipeline (double or triple) and print JSON")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-url", help="HTTP/HTTPS URL of the image")
    g.add_argument("--file-path", help="Local file path to the image")

    p.add_argument("--caption-override", help="Provide an existing dense caption to skip the vision caption step")
    p.add_argument("--config-path", default=None, help="Path to model config JSON (defaults to packaged config)")
    p.add_argument("--schema-path", default=None, help="Path to schema.json (defaults to packaged schema)")
    p.add_argument("--mode", choices=["double", "triple"], default="double", help="Pipeline mode: double (vision alt+caption + text metadata) or triple (vision alt+caption + vision metadata)")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    # Backend overrides (useful for local testing without editing global config)
    # New flag names
    p.add_argument("--caption-backend", choices=["openrouter", "local", "ollama"], default=None, help="Backend for alt+caption step (default from global config)")
    p.add_argument("--metadata-vision-backend", choices=["openrouter", "local", "ollama"], default=None, help="Backend for metadata (vision) in triple mode (default from global config)")
    p.add_argument("--local-vlm-id", default=None, help="Local VLM model id (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    p.add_argument("--ollama-host", default=None, help="Ollama base URL (default http://localhost:11434)")
    # Legacy flags (kept for compatibility)
    p.add_argument("--ac-backend", dest="_legacy_ac_backend", choices=["openrouter", "local"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--meta-vision-backend", dest="_legacy_meta_vision_backend", choices=["openrouter", "local"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--local-model-id", dest="_legacy_local_model_id", default=None, help=argparse.SUPPRESS)

    args = p.parse_args()
    image_ref = args.image_url or args.file_path  # type: ignore

    try:
        # Apply backend overrides to the in-memory global config
        from cv_mcp.metadata import runner as md_runner  # type: ignore
        effective_cfg = dict(md_runner._CFG)
        # Resolve flags (new preferred, fall back to legacy ones)
        ac_backend = args.caption_backend or args._legacy_ac_backend
        mv_backend = args.metadata_vision_backend or args._legacy_meta_vision_backend
        local_vlm_id = args.local_vlm_id or args._legacy_local_model_id
        if ac_backend:
            effective_cfg["caption_backend"] = ac_backend
        if mv_backend:
            effective_cfg["metadata_vision_backend"] = mv_backend
        if local_vlm_id:
            effective_cfg["local_vlm_id"] = local_vlm_id
        if args.ollama_host:
            effective_cfg["ollama_host"] = args.ollama_host
        # Persist overrides for this process
        md_runner._CFG.update(effective_cfg)

        # Early env check only if any step uses OpenRouter
        def _is_local(key: str) -> bool:
            return str(effective_cfg.get(f"{key}_backend", "openrouter")).lower() == "local"

        needs_or_key = False
        if args.mode == "double":
            # Double always uses text LLM for metadata via OpenRouter
            needs_or_key = True
        else:
            # Triple only needs OR if either step is remote
            needs_or_key = not (_is_local("caption") and _is_local("metadata_vision"))

        if needs_or_key and not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is required for the selected mode/backends. Add it to your environment or a .env file.")

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
                    meta = run_metadata_from_caption(args.caption_override, schema_path=schema_path, model=cfg.get("metadata_text_model"))
                else:
                    meta = run_metadata_from_caption(args.caption_override, schema_path=schema_path)
                alt = run_alt_text(image_ref)
            else:
                # Triple: Vision+caption metadata
                if cfg_path and cfg_path.exists():
                    import json as _json
                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    meta = run_structured_json(image_ref, args.caption_override, schema_path=schema_path, model=cfg.get("metadata_vision_model"))
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
