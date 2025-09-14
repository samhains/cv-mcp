# cv-art-mcp

Minimal MCP server focused on computer vision: image recognition (captioning) via OpenRouter (Gemini 2.5 Flash).

Goals
- Keep it tiny and composable
- Single tool: caption an image via URL or local file
- No DB or app logic

Structure
- `src/cv_art_mcp/captioning/openrouter_client.py` – image analysis client
- `src/cv_art_mcp/mcp_server.py` – MCP server exposing `caption_image`
- `cli/caption_image.py` – optional CLI to test captioning locally

Env vars
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (optional, default: `google/gemini-2.5-flash`)

Install
- `pip install -e .` (or `pip install .`)

Run MCP server (stdio)
- Console script: `cv-art-mcp-server` (provides an MCP stdio server)
- Configure your MCP client to launch `cv-art-mcp-server`.

MCP integration (Claude Desktop)
- Add to Claude Desktop config (see their docs for the config location):
  {
    "mcpServers": {
      "cv-art-mcp": {
        "command": "cv-art-mcp-server",
        "env": {
          "OPENROUTER_API_KEY": "sk-or-...",
          "OPENROUTER_MODEL": "google/gemini-2.5-flash"
        }
      }
    }
  }
- After saving, restart Claude Desktop and enable the tool.

Tool: caption_image
- Inputs: `image_url` (http/https) or `file_path` (local), optional `prompt`
- Output: concise caption text

MCP tool reference
- Server: `cv-art-mcp` (stdio)
- Tool name: `caption_image`
- Description: Generate a concise caption for an image using OpenRouter (default) or a local VLM
- Parameters:
  - `image_url` (string): HTTP/HTTPS image URL. Mutually exclusive with `file_path`.
  - `file_path` (string): Local file path to the image. Mutually exclusive with `image_url`.
  - `prompt` (string, optional): Defaults to “Write a concise, vivid caption for this image. Describe key subjects, scene, and mood in 1-2 sentences.”
  - `backend` (string, optional): `openrouter` | `local` (default: `openrouter`).
  - `local_model_id` (string, optional): For `backend=local`, defaults to `Qwen/Qwen2-VL-2B-Instruct`.
- Returns: caption text (string)

Examples
- MCP call (OpenRouter):
  {"image_url": "https://example.com/image.jpg"}
- MCP call (local):
  {"backend": "local", "file_path": "./image.jpg"}

Quick test (CLI)
- URL: `python cli/caption_image.py --image-url https://example.com/img.jpg`
- File: `python cli/caption_image.py --file-path ./image.png`

Local backend (optional)
- Install optional deps: `pip install .[local]`
- Use with MCP: pass `backend: "local"` in the tool params
- Use with CLI: add `--backend local` and optionally `--local-model-id Qwen/Qwen2-VL-2B-Instruct`
- Requires a locally available model (default: `Qwen/Qwen2-VL-2B-Instruct` via HF cache)

Examples
- MCP tool (local): `{"backend": "local", "file_path": "./image.jpg"}`
- CLI (local): `python cli/caption_image.py --file-path ./image.jpg --backend local`

Troubleshooting
- 401/403 from OpenRouter: ensure `OPENROUTER_API_KEY` is set and valid.
- Model selection: override default with `OPENROUTER_MODEL`.
- Large images: remote images are downloaded and sent as base64; ensure the URL is accessible.
- Local backend: install optional deps `pip install .[local]` and ensure model is present/cached.
