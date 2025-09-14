# cv-art-mcp

Minimal MCP server focused on computer vision: image recognition (captioning) via OpenRouter (Claude 3.5 Sonnet).

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

Install
- `pip install -e .` (or `pip install .`)

Run MCP server (stdio)
- Console script: `cv-art-mcp-server` (provides an MCP stdio server)
- Configure your MCP client (e.g., Claude Desktop) to launch `cv-art-mcp-server`.

Tool: caption_image
- Inputs: `image_url` (http/https) or `file_path` (local), optional `prompt`
- Output: concise caption text

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
