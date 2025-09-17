# cv-mcp

Minimal MCP server focused on computer vision: image recognition and metadata generation via OpenRouter (Gemini 2.5 family).

Goals
- Keep it tiny and composable
- Single tool: caption an image via URL or local file
- No DB or app logic

Structure
- `src/cv_mcp/captioning/openrouter_client.py` – image analysis client
- `src/cv_mcp/metadata/` – prompts, JSON schema, and pipeline runner
- `src/cv_mcp/mcp_server.py` – MCP server exposing tools
- `cli/caption_image.py` – optional CLI to test captioning locally

Env vars
- `OPENROUTER_API_KEY`

Dotenv
- Put `OPENROUTER_API_KEY` in a local `.env` file (see `.env.example`).
- CLI scripts and the MCP server auto-load `.env` if present.

Install
- `pip install -e .` (or `pip install .`)

⚠️ **Development Note**: If you have the package installed via `pip install`, uninstall it before working with the local development version to avoid import conflicts. Use `pip uninstall cv-mcp` first, then run commands directly from the repo directory.

Run MCP server (stdio)
- Console script: `cv-mcp-server` (provides an MCP stdio server)
- Configure your MCP client to launch `cv-mcp-server`.

MCP integration (Claude Desktop)
- Add to Claude Desktop config (see their docs for the config location):
  {
    "mcpServers": {
      "cv-mcp": {
        "command": "cv-mcp-server",
        "env": {
          "OPENROUTER_API_KEY": "sk-or-..."
        }
      }
    }
  }
- After saving, restart Claude Desktop and enable the tool.

Tools
- `caption_image`: one-off caption (kept for compatibility); accepts `model` for per-call OpenRouter overrides and optional `context` appended to the prompt.
- `alt_text`: short alt text (<= 20 words); accepts `model` and optional `context` for per-call overrides.
- `dense_caption`: detailed 2–6 sentence caption; accepts `model` and optional `context` for per-call overrides.
- `image_metadata`: structured JSON metadata with alt + caption. Params:
  - `mode`: `double` (default) uses 2 calls: vision (alt+caption) + text-only (metadata). `triple` uses vision for both steps.
  - `caption_override`: supply your own dense caption; skips the vision caption step.
  - Optional per-call model overrides: `caption_model`, `metadata_text_model`, `metadata_vision_model`.
  - Optional `context`: free-form hints (e.g., filename, operator notes) appended to every vision call during the request.

MCP tool reference
- Server: `cv-mcp` (stdio)
- `caption_image(image_url|file_path, prompt?, backend?, local_model_id?, model?, context?) -> string`
- `alt_text(image_url|file_path, max_words?, model?, context?) -> string`
- `dense_caption(image_url|file_path, model?, context?) -> string`
- `image_metadata(image_url|file_path, caption_override?, config_path?, mode?, caption_model?, metadata_text_model?, metadata_vision_model?, context?) -> { alt_text, caption, metadata }`

Examples
- MCP call (OpenRouter):
  {"image_url": "https://example.com/image.jpg"}
- MCP call (local):
  {"backend": "local", "file_path": "./image.jpg"}

Quick test (CLI)
- URL: `python cli/caption_image.py --image-url https://example.com/img.jpg`
- File: `python cli/caption_image.py --file-path ./image.png`

Metadata pipeline (CLI)
- Double (default):
  - `python cli/image_metadata.py --image-url https://example.com/img.jpg --mode double`
  - Local alt+caption (still requires OpenRouter for metadata):
    - `python cli/image_metadata.py --image-url https://example.com/img.jpg --mode double --ac-backend local`
- Triple (vision metadata):
  - `python cli/image_metadata.py --image-url https://example.com/img.jpg --mode triple`
  - Fully local (no OpenRouter required):
    - `python cli/image_metadata.py --image-url https://example.com/img.jpg --mode triple --ac-backend local --meta-vision-backend local`
- With existing caption (skips the caption step):
  - `python cli/image_metadata.py --image-url https://example.com/img.jpg --caption-override "<dense caption>" --mode double`
- Custom model config (JSON with `caption_model`, `metadata_text_model`, `metadata_vision_model`):
  - `python cli/image_metadata.py --image-url https://example.com/img.jpg --config-path ./my_models.json --mode double`

Schema & vocab
- JSON schema (lean): `src/cv_mcp/metadata/schema.json`
- Controlled vocab (non-binding reference): `src/cv_mcp/metadata/vocab.json`

Global config
- Root file: `cv_mcp.config.json` (auto-detected from project root / CWD)
- Env override: set `CV_MCP_CONFIG=/path/to/config.json`
- Keys (renamed for clarity):
  - `caption_model`: vision model for alt+caption (OpenRouter)
  - `metadata_text_model`: text model for metadata (double mode)
  - `metadata_vision_model`: vision model for metadata (triple mode)
  - `caption_backend`: `openrouter` (default) or `local` for alt/dense/AC steps
  - `metadata_vision_backend`: `openrouter` (default) or `local` for triple mode
  - `local_vlm_id`: default local VLM (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`)
  - Backwards-compat: legacy keys (`ac_model`, `meta_text_model`, `meta_vision_model`, `ac_backend`, `meta_vision_backend`, `local_model_id`) are still accepted.
- Packaged defaults still live at `src/cv_mcp/metadata/config.json` and are used if no root config is found.
- You can still provide a custom config file per-call via `--config-path` or the `config_path` tool param.

Local backends (optional)
- Install optional deps: `pip install .[local]`
- Global default: set `"caption_backend": "local"` (and optionally `"metadata_vision_backend": "local"`) in `cv_mcp.config.json`
- Use with MCP: pass `backend: "local"` in the tool params (overrides global)
- Use with CLI: add `--backend local` and optionally `--local-model-id Qwen/Qwen2-VL-2B-Instruct` (overrides global)
- Requires a locally available model (default: `Qwen/Qwen2-VL-2B-Instruct` via HF cache)

- Or run without transformers using Ollama (no Python ML deps):
  - Install and run Ollama; pull a vision model (e.g., `ollama pull qwen2.5-vl`)
  - Use backend `ollama` and set models in the config (e.g., `caption_model: "qwen2.5-vl"`)
  - CLI example (triple, fully local):
    - `python cli/image_metadata.py --image-url https://... --mode triple --caption-backend ollama --metadata-vision-backend ollama --config-path ./configs/triple_ollama_qwen.json`
  - Configure host with `--ollama-host http://localhost:11434` if not default

Per-call overrides (CLI)
- Metadata CLI now supports per-call backend overrides without editing global config:
  - `--caption-backend local|openrouter|ollama` (legacy: `--ac-backend`)
  - `--metadata-vision-backend local|openrouter|ollama` (legacy: `--meta-vision-backend`)
  - `--local-vlm-id Qwen/Qwen2.5-VL-7B-Instruct` (legacy: `--local-model-id`)
  - `--ollama-host http://localhost:11434`

Justfile tasks
- A `Justfile` provides quick test scenarios. Use URL-only inputs, e.g. `just double_flash https://example.com/img.jpg`.
- Scenarios included:
  - `double_flash`: Gemini 2.5 Flash for both steps
  - `double_pro`: Gemini 2.5 Pro for both steps
  - `double_mixed_pro_text`: Flash for vision alt+caption, Pro for text metadata (recommended mix for JSON reliability)
  - `triple_flash` / `triple_pro`: Flash/Pro for both vision steps
  - `double_qwen_local <url> <qwen_id>`: Local Qwen 2.5 VL for vision step, Pro for text metadata
  - `triple_qwen_local <url> <qwen_id>`: Fully local Qwen 2.5 VL for both vision steps
  - Convenience (no extra args):
    - `double_qwen2b_local <url>` / `triple_qwen2b_local <url>`
    - `double_qwen7b_local <url>` / `triple_qwen7b_local <url>`

Recommendation for mixed double
- Put Gemini 2.5 Pro on the text metadata step and Flash on the vision alt+caption step. The metadata step benefits from better structured-JSON compliance and reasoning, while Flash keeps latency/cost down for the vision caption.
- OpenRouter key requirements:
  - Double mode always requires `OPENROUTER_API_KEY` (text LLM for metadata).
  - Triple mode requires `OPENROUTER_API_KEY` unless both `--ac-backend local` and `--meta-vision-backend local` are set.
- Per-call overrides (MCP): pass `model`, `caption_model`, `metadata_text_model`, `metadata_vision_model`, or `context` in the tool payload to experiment without editing config files.

Examples
- MCP tool (local): `{"backend": "local", "file_path": "./image.jpg"}`
- CLI (local): `python cli/caption_image.py --file-path ./image.jpg --backend local`

Troubleshooting
- 401/403 from OpenRouter: ensure `OPENROUTER_API_KEY` is set and valid.
- Model selection: prefer `cv_mcp.config.json` at project root; or pass `--config-path`.
- Large images: remote images are downloaded and sent as base64; ensure the URL is accessible.
- Local backend: install optional deps `pip install .[local]` and ensure model is present/cached.

Changelog
- See `docs/CHANGELOG.md` for notable changes and release notes.
