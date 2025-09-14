from __future__ import annotations

import json
from typing import Optional, Union, Dict, Any
from pathlib import Path

from cv_mcp.captioning.openrouter_client import OpenRouterClient
from cv_mcp.metadata import prompts

_CONFIG_PATH = Path(__file__).with_name("config.json")
try:
    _MODELS = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
except Exception:
    _MODELS = {
        "ac_model": "google/gemini-2.5-pro",
        "meta_text_model": "google/gemini-2.5-pro",
        "meta_vision_model": "google/gemini-2.5-pro",
    }


def _load_text(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8")


def run_alt_text(image_ref: str, *, model: Optional[str] = None, max_words: int = 20) -> str:
    client = OpenRouterClient()
    res = client.analyze_single_image(
        image_ref,
        prompts.alt_user_prompt(max_words),
        model=model or _MODELS.get("ac_model"),
        system=prompts.ALT_SYSTEM,
    )
    if not res.get("success"):
        raise RuntimeError(str(res.get("error", "Alt text generation failed")))
    return str(res.get("content", "")).strip()


def run_dense_caption(image_ref: str, *, model: Optional[str] = None) -> str:
    client = OpenRouterClient()
    res = client.analyze_single_image(
        image_ref,
        prompts.CAPTION_USER,
        model=model or _MODELS.get("ac_model"),
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
    client = OpenRouterClient()
    user_prompt = prompts.structured_user(dense_caption)
    res = client.analyze_single_image(
        image_ref,
        user_prompt,
        model=model or _MODELS.get("meta_vision_model"),
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
    client = OpenRouterClient()
    res = client.analyze_single_image(
        image_ref,
        prompts.ac_user(),
        model=model or _MODELS.get("ac_model"),
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
    client = OpenRouterClient()
    res = client.chat(
        messages=[
            {"role": "system", "content": prompts.structured_text_system()},
            {"role": "user", "content": prompts.structured_text_user(caption)},
        ],
        model=model or _MODELS.get("meta_text_model"),
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
    cfg = _MODELS
    if config_path:
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read config from {config_path}: {e}")
    ac = run_alt_and_caption(image_ref, model=cfg.get("ac_model"))
    meta = run_metadata_from_caption(ac["caption"], schema_path=schema_path, model=cfg.get("meta_text_model"))
    return {"alt_text": ac["alt_text"], "caption": ac["caption"], "metadata": meta}


def run_pipeline_triple(
    image_ref: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
    schema_path: Union[str, Path] = Path(__file__).with_name("schema.json"),
) -> Dict[str, Any]:
    cfg = _MODELS
    if config_path:
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read config from {config_path}: {e}")
    ac = run_alt_and_caption(image_ref, model=cfg.get("ac_model"))
    meta = run_structured_json(image_ref, ac["caption"], schema_path=schema_path, model=cfg.get("meta_vision_model"))
    return {"alt_text": ac["alt_text"], "caption": ac["caption"], "metadata": meta}
