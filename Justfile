# Usage examples (URL only):
#   just double_flash https://example.com/img.jpg
#   just double_pro https://example.com/img.jpg
#   just double_qwen_local https://example.com/img.jpg Qwen/Qwen2.5-VL-7B-Instruct

set shell := ["bash", "-cu"]

# Double (vision alt+caption + text metadata)
double_flash url:
    python cli/image_metadata.py --image-url "{{url}}" --mode double --config-path configs/double_flash.json

double_pro url:
    python cli/image_metadata.py --image-url "{{url}}" --mode double --config-path configs/double_pro.json

# Recommended mix: Flash for vision, Pro for text metadata
double_mixed_pro_text url:
    python cli/image_metadata.py --image-url "{{url}}" --mode double --config-path configs/double_mixed_pro_text.json

# Triple (vision alt+caption + vision metadata)
triple_flash url:
    python cli/image_metadata.py --image-url "{{url}}" --mode triple --config-path configs/triple_flash.json

triple_pro url:
    python cli/image_metadata.py --image-url "{{url}}" --mode triple --config-path configs/triple_pro.json

# Local Qwen 2.5 VL scenarios
# Double: local vision, remote text metadata (Pro)
double_qwen_local url qwen_id:
    python cli/image_metadata.py --image-url "{{url}}" --mode double --caption-backend local --local-vlm-id "{{qwen_id}}" --config-path configs/double_pro.json

# Triple: fully local (no OpenRouter)
triple_qwen_local url qwen_id:
    python cli/image_metadata.py --image-url "{{url}}" --mode triple --caption-backend local --metadata-vision-backend local --local-vlm-id "{{qwen_id}}"

# Convenience: fixed Qwen variants
double_qwen2b_local url:
    python cli/image_metadata.py --image-url "{{url}}" --mode double --caption-backend local --local-vlm-id "Qwen/Qwen2-VL-2B-Instruct" --config-path configs/double_pro.json

triple_qwen2b_local url:
    python cli/image_metadata.py --image-url "{{url}}" --mode triple --caption-backend local --metadata-vision-backend local --local-vlm-id "Qwen/Qwen2-VL-2B-Instruct"

double_qwen7b_local url:
    python cli/image_metadata.py --image-url "{{url}}" --mode double --caption-backend local --local-vlm-id "Qwen/Qwen2.5-VL-7B-Instruct" --config-path configs/double_pro.json

triple_qwen7b_local url:
    python cli/image_metadata.py --image-url "{{url}}" --mode triple --caption-backend local --metadata-vision-backend local --local-vlm-id "Qwen/Qwen2.5-VL-7B-Instruct"
# Install optional local dependencies (transformers, torch, etc.)
setup_local:
    python -m pip install -U pip wheel
    python -m pip install '.[local]'
# Ollama (no transformers) — fully local vision (triple)
triple_qwen_ollama url:
    python cli/image_metadata.py --image-url "{{url}}" --mode triple --caption-backend ollama --metadata-vision-backend ollama --config-path configs/triple_ollama_qwen.json
