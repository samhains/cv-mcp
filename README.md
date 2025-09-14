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
- `caption_image`: one-off caption (kept for compatibility)
- `alt_text`: short alt text (<= 20 words)
- `dense_caption`: detailed 2–6 sentence caption
- `image_metadata`: structured JSON metadata with alt + caption. Params:
  - `mode`: `double` (default) uses 2 calls: vision (alt+caption) + text-only (metadata). `triple` uses vision for both steps.
  - `caption_override`: supply your own dense caption; skips the vision caption step.

MCP tool reference
- Server: `cv-mcp` (stdio)
- `caption_image(image_url|file_path, prompt?, backend?, local_model_id?) -> string`
- `alt_text(image_url|file_path, max_words?) -> string`
- `dense_caption(image_url|file_path) -> string`
- `image_metadata(image_url|file_path, caption_override?, config_path?) -> { alt_text, caption, metadata }`

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
- Triple (vision metadata):
  - `python cli/image_metadata.py --image-url https://example.com/img.jpg --mode triple`
- With existing caption (skips the caption step):
  - `python cli/image_metadata.py --image-url https://example.com/img.jpg --caption-override "<dense caption>" --mode double`
- Custom model config (JSON with `ac_model`, `meta_text_model`, `meta_vision_model`):
  - `python cli/image_metadata.py --image-url https://example.com/img.jpg --config-path ./my_models.json --mode double`

Schema & vocab
- JSON schema (lean): `src/cv_mcp/metadata/schema.json`
- Controlled vocab (non-binding reference): `src/cv_mcp/metadata/vocab.json`

Global config
- Root file: `cv_mcp.config.json` (auto-detected from project root / CWD)
- Env override: set `CV_MCP_CONFIG=/path/to/config.json`
- Keys:
  - `ac_model`: vision model for alt+caption (OpenRouter)
  - `meta_text_model`: text model for metadata (double mode)
  - `meta_vision_model`: vision model for metadata (triple mode)
  - `ac_backend`: `openrouter` (default) or `local` for alt/dense/AC steps
  - `meta_vision_backend`: `openrouter` (default) or `local` for triple mode
  - `local_model_id`: default local VLM (e.g. `Qwen/Qwen2-VL-2B-Instruct`)
- Packaged defaults still live at `src/cv_mcp/metadata/config.json` and are used if no root config is found.
- You can still provide a custom config file per-call via `--config-path` or the `config_path` tool param.

Local backend (optional)
- Install optional deps: `pip install .[local]`
- Global default: set `"ac_backend": "local"` (and optionally `"meta_vision_backend": "local"`) in `cv_mcp.config.json`
- Use with MCP: pass `backend: "local"` in the tool params (overrides global)
- Use with CLI: add `--backend local` and optionally `--local-model-id Qwen/Qwen2-VL-2B-Instruct` (overrides global)
- Requires a locally available model (default: `Qwen/Qwen2-VL-2B-Instruct` via HF cache)

Examples
- MCP tool (local): `{"backend": "local", "file_path": "./image.jpg"}`
- CLI (local): `python cli/caption_image.py --file-path ./image.jpg --backend local`

Troubleshooting
- 401/403 from OpenRouter: ensure `OPENROUTER_API_KEY` is set and valid.
- Model selection: prefer `cv_mcp.config.json` at project root; or pass `--config-path`.
- Large images: remote images are downloaded and sent as base64; ensure the URL is accessible.
- Local backend: install optional deps `pip install .[local]` and ensure model is present/cached.
