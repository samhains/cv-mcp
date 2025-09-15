from __future__ import annotations

import json
import os
from typing import Optional, Union, Dict, Any
from pathlib import Path

from cv_mcp.captioning.openrouter_client import OpenRouterClient
from cv_mcp.metadata import prompts

_PKG_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy config keys to new, clearer names.

    Canonical keys:
      - caption_model (was ac_model)
      - metadata_text_model (was meta_text_model)
      - metadata_vision_model (was meta_vision_model)
      - caption_backend (was ac_backend)
      - metadata_vision_backend (was meta_vision_backend)
      - local_vlm_id (was local_model_id)
    """
    mapping = {
        "ac_model": "caption_model",
        "meta_text_model": "metadata_text_model",
        "meta_vision_model": "metadata_vision_model",
        "ac_backend": "caption_backend",
        "meta_vision_backend": "metadata_vision_backend",
        "local_model_id": "local_vlm_id",
    }
    out: Dict[str, Any] = {}
    # First copy already-new keys as-is
    for k in (
        "caption_model",
        "metadata_text_model",
        "metadata_vision_model",
        "caption_backend",
        "metadata_vision_backend",
        "local_vlm_id",
        "ollama_host",
    ):
        if k in raw:
            out[k] = raw[k]
    # Then map any legacy keys not already set
    for old, new in mapping.items():
        if new not in out and old in raw:
            out[new] = raw[old]
    return out


def _load_global_config() -> Dict[str, Any]:
    # 1) Explicit env override
    env_path = os.getenv("CV_MCP_CONFIG")
    if env_path and Path(env_path).exists():
        try:
            return _normalize_config(_read_json(env_path))
        except Exception:
            pass
    # 2) Project-root default (cwd)
    cwd_cfg = Path.cwd() / "cv_mcp.config.json"
    if cwd_cfg.exists():
        try:
            return _normalize_config(_read_json(cwd_cfg))
        except Exception:
            pass
    # 3) Packaged defaults
    try:
        return _normalize_config(_read_json(_PKG_DEFAULT_CONFIG_PATH))
    except Exception:
        return _normalize_config({
            "caption_model": "google/gemini-2.5-pro",
            "metadata_text_model": "google/gemini-2.5-pro",
            "metadata_vision_model": "google/gemini-2.5-pro",
        })


# Loaded on import; used as base defaults and merged per call if file is provided
_CFG: Dict[str, Any] = _load_global_config()


def _load_text(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8")


def _cfg_value(key: str, default: Any = None) -> Any:
    return _CFG.get(key, default)


def _backend_for(key: str) -> str:
    # Returns: openrouter (default), local, or ollama
    return str(_cfg_value(f"{key}_backend", "openrouter")).lower()

def _use_local_for(key: str) -> bool:
    return _backend_for(key) == "local"

def _use_ollama_for(key: str) -> bool:
    return _backend_for(key) == "ollama"


def _local_gen(image_ref: str, prompt: str, *, max_new_tokens: int = 256) -> str:
    try:
        from cv_mcp.captioning.local_captioner import LocalCaptioner
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Local backend not available. Install optional deps with `pip install .[local]`."
        ) from e
    model_id = str(_cfg_value("local_vlm_id", "Qwen/Qwen2-VL-2B-Instruct"))
    local = LocalCaptioner(model_id=model_id)
    return local.caption(image_ref, prompt, max_new_tokens=max_new_tokens)


def run_alt_text(image_ref: str, *, model: Optional[str] = None, max_words: int = 20) -> str:
    if _use_local_for("caption"):
        prompt = f"{prompts.ALT_SYSTEM}\n\n{prompts.alt_user_prompt(max_words)}"
        return _local_gen(image_ref, prompt)
    if _use_ollama_for("caption"):
        from cv_mcp.captioning.ollama_client import OllamaClient
        client = OllamaClient(host=str(_cfg_value("ollama_host", "http://localhost:11434")))
        res = client.analyze_single_image(
            image_ref,
            prompts.alt_user_prompt(max_words),
            model=_cfg_value("caption_model"),
            system=prompts.ALT_SYSTEM,
        )
        if not res.get("success"):
            raise RuntimeError(str(res.get("error", "Alt text generation failed (ollama)")))
        return str(res.get("content", "")).strip()
    client = OpenRouterClient()
    res = client.analyze_single_image(
        image_ref,
        prompts.alt_user_prompt(max_words),
        model=model or _cfg_value("caption_model"),
        system=prompts.ALT_SYSTEM,
    )
    if not res.get("success"):
        raise RuntimeError(str(res.get("error", "Alt text generation failed")))
    return str(res.get("content", "")).strip()


def run_dense_caption(image_ref: str, *, model: Optional[str] = None) -> str:
    if _use_local_for("caption"):
        prompt = f"{prompts.CAPTION_SYSTEM}\n\n{prompts.CAPTION_USER}"
        return _local_gen(image_ref, prompt)
    if _use_ollama_for("caption"):
        from cv_mcp.captioning.ollama_client import OllamaClient
        client = OllamaClient(host=str(_cfg_value("ollama_host", "http://localhost:11434")))
        res = client.analyze_single_image(
            image_ref,
            prompts.CAPTION_USER,
            model=_cfg_value("caption_model"),
            system=prompts.CAPTION_SYSTEM,
        )
        if not res.get("success"):
            raise RuntimeError(str(res.get("error", "Dense caption generation failed (ollama)")))
        return str(res.get("content", "")).strip()
    client = OpenRouterClient()
    res = client.analyze_single_image(
        image_ref,
        prompts.CAPTION_USER,
        model=model or _cfg_value("caption_model"),
        system=prompts.CAPTION_SYSTEM,
    )
    if not res.get("success"):
        raise RuntimeError(str(res.get("error", "Dense caption generation failed")))
    return str(res.get("content", "")).strip()


def run_structured_json(
    image_ref: str,
    dense_caption: str,
    *,
    schema_path: Union[str, Path],
    model: Optional[str] = None,
        ) -> Dict[str, Any]:
    user_prompt = prompts.structured_user(dense_caption)
    if _use_local_for("metadata_vision"):
        prompt = f"{prompts.structured_system()}\n\n{user_prompt}"
        text = _local_gen(image_ref, prompt, max_new_tokens=512)
    elif _use_ollama_for("metadata_vision"):
        from cv_mcp.captioning.ollama_client import OllamaClient
        client = OllamaClient(host=str(_cfg_value("ollama_host", "http://localhost:11434")))
        res = client.analyze_single_image(
            image_ref,
            user_prompt,
            model=_cfg_value("metadata_vision_model"),
            system=prompts.structured_system(),
        )
        if not res.get("success"):
            raise RuntimeError(str(res.get("error", "JSON metadata generation failed (ollama)")))
        text = str(res.get("content", "")).strip()
    else:
        client = OpenRouterClient()
        res = client.analyze_single_image(
            image_ref,
            user_prompt,
            model=model or _cfg_value("metadata_vision_model"),
            system=prompts.structured_system(),
        )
        if not res.get("success"):
            raise RuntimeError(str(res.get("error", "JSON metadata generation failed")))
        text = str(res.get("content", "")).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            raise RuntimeError("Model did not return valid JSON")

    _post_validate(data)
    return data


def _clamp(v: Any, lo: float = 0.0, hi: float = 1.0) -> Any:
    try:
        f = float(v)
        if f < lo:
            return lo
        if f > hi:
            return hi
        return f
    except Exception:
        return v


def _post_validate(data: Dict[str, Any]) -> None:
    # Enforce array caps and build tags if missing
    def _cap(key: str, n: int):
        if isinstance(data.get(key), list) and len(data[key]) > n:
            data[key] = data[key][:n]

    for k, n in ("objects", 6), ("scene", 3), ("lighting", 3), ("style", 5), ("palette", 6), ("tags", 20):
        _cap(k, n)

    # Ensure people fields exist with defaults
    if not isinstance(data.get("people"), dict):
        data["people"] = {"count": 0, "faces_visible": False}
    else:
        data["people"].setdefault("count", 0)
        data["people"].setdefault("faces_visible", False)

    # Compute tags union if missing or empty
    if not isinstance(data.get("tags"), list) or not data.get("tags"):
        def norm_list(v):
            return v if isinstance(v, list) else []
        tags = []
        if isinstance(data.get("media_type"), str):
            tags.append(data["media_type"])
        for k in ("scene", "lighting", "style", "palette", "objects"):
            tags.extend(norm_list(data.get(k)))
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for t in tags:
            if isinstance(t, str) and t not in seen:
                seen.add(t)
                uniq.append(t)
        data["tags"] = uniq[:20]

    # Always include essential keys; drop empty/null fields for others
    essentials = {"media_type", "objects", "people", "tags"}
    to_delete = []
    for k, v in list(data.items()):
        if k in essentials:
            continue
        if v is None:
            to_delete.append(k)
        elif isinstance(v, list) and len(v) == 0:
            to_delete.append(k)
        elif isinstance(v, dict) and len(v.keys()) == 0:
            to_delete.append(k)
    for k in to_delete:
        try:
            del data[k]
        except Exception:
            pass


def run_alt_and_caption(image_ref: str, *, model: Optional[str] = None) -> Dict[str, str]:
    if _use_local_for("caption"):
        prompt = f"{prompts.AC_SYSTEM}\n\n{prompts.ac_user()}"
        text = _local_gen(image_ref, prompt, max_new_tokens=512)
    elif _use_ollama_for("caption"):
        from cv_mcp.captioning.ollama_client import OllamaClient
        client = OllamaClient(host=str(_cfg_value("ollama_host", "http://localhost:11434")))
        res = client.analyze_single_image(
            image_ref,
            prompts.ac_user(),
            model=_cfg_value("caption_model"),
            system=prompts.AC_SYSTEM,
        )
        if not res.get("success"):
            raise RuntimeError(str(res.get("error", "Alt+caption generation failed (ollama)")))
        text = str(res.get("content", "")).strip()
    else:
        client = OpenRouterClient()
        res = client.analyze_single_image(
            image_ref,
            prompts.ac_user(),
            model=model or _cfg_value("caption_model"),
            system=prompts.AC_SYSTEM,
        )
        if not res.get("success"):
            raise RuntimeError(str(res.get("error", "Alt+caption generation failed")))
        text = str(res.get("content", "")).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            raise RuntimeError("Model did not return valid JSON for alt+caption")
    return {"alt_text": str(data.get("alt_text", "")).strip(), "caption": str(data.get("caption", "")).strip()}


def run_metadata_from_caption(caption: str, *, schema_path: Union[str, Path], model: Optional[str] = None) -> Dict[str, Any]:
    # Text metadata requires a chat LLM; only OpenRouter supported here.
    client = OpenRouterClient()
    res = client.chat(
        messages=[
            {"role": "system", "content": prompts.structured_text_system()},
            {"role": "user", "content": prompts.structured_text_user(caption)},
        ],
        model=model or _cfg_value("metadata_text_model"),
    )
    if not res.get("success"):
        raise RuntimeError(str(res.get("error", "Metadata (text) generation failed")))
    text = str(res.get("content", "")).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            raise RuntimeError("Model did not return valid JSON for metadata (text)")
    _post_validate(data)
    return data


def run_pipeline_double(
    image_ref: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
    schema_path: Union[str, Path] = Path(__file__).with_name("schema.json"),
) -> Dict[str, Any]:
    cfg = dict(_CFG)
    if config_path:
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read config from {config_path}: {e}")
    ac = run_alt_and_caption(image_ref, model=cfg.get("caption_model"))
    meta = run_metadata_from_caption(ac["caption"], schema_path=schema_path, model=cfg.get("metadata_text_model"))
    return {"alt_text": ac["alt_text"], "caption": ac["caption"], "metadata": meta}


def run_pipeline_triple(
    image_ref: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
    schema_path: Union[str, Path] = Path(__file__).with_name("schema.json"),
) -> Dict[str, Any]:
    cfg = dict(_CFG)
    if config_path:
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read config from {config_path}: {e}")
    ac = run_alt_and_caption(image_ref, model=cfg.get("caption_model"))
    meta = run_structured_json(image_ref, ac["caption"], schema_path=schema_path, model=cfg.get("metadata_vision_model"))
    return {"alt_text": ac["alt_text"], "caption": ac["caption"], "metadata": meta}
