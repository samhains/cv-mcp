# Changelog

All notable changes to this project are documented here.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning where practical.

## [Unreleased]
- Planned: additional CLI ergonomics and presets for local/remote testing.
- Planned: Makefile targets for common workflows.

### Fixed
- Local backend transformers compatibility: Replace non-existent `AutoModelForConditionalGeneration` with proper model classes (`Qwen2_5_VLForConditionalGeneration` for Qwen models)
- JSON parsing for model outputs wrapped in markdown code blocks (```json...```)
- Model loading fallback logic improved with better error handling

## [0.0.1] - 2025-09-14
### Added
- Metadata CLI backend overrides: `--ac-backend`, `--meta-vision-backend`, and `--local-model-id`.
- Conditional OpenRouter key requirement in CLI (only when a selected step uses OpenRouter).
- Documentation for testing local "double" with URL and fully local "triple" flows.

### Changed
- `cli/image_metadata.py` help text clarified to cover both double and triple modes.

### Notes
- Double mode still requires a text LLM via OpenRouter for metadata. Use triple mode with both backends set to `local` for fully local operation.

