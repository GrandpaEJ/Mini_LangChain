# Changelog

All notable changes to Mini LangChain will be documented in this file.

## [0.1.0] - 2026-02-01

### Added
- **Multi-Provider LLM Support**: Added native Rust implementations and bindings for:
  - OpenAI (Standard and OpenAI-compatible like OpenRouter)
  - Anthropic Claude
  - Google Gemini
  - Ollama (Local)
  - SambaNova
- **Memory & Persistence**: 
  - `ConversationBufferMemory` for chat history management.
  - `InMemoryCache` for prompt/response caching.
- **RAG Implementation**:
  - `Document` and `TextLoader` for data ingestion.
  - `InMemoryVectorStore` with Cosine Similarity.
  - `Embeddings` trait and `MockEmbeddings` (Transitioning to real providers).
- **Agent Framework**: 
  - `AgentExecutor` for tool-calling logic.
  - `Tool` trait for custom tool implementation.
- **Cross-Language Bindings**:
  - **Python**: Universal ABI3 wheels support (Python 3.9+), `maturin` integration.
  - **Node.js**: Type-safe `napi-rs` bindings with support for multiple LLM inputs via `Either`.
- **CI/CD Workflows**:
  - Automated publishing to **crates.io**, **NPM**, and **PyPI**.
  - Manual dispatch support for release workflows.

### Changed
- **Dependency Upgrades**:
  - Upgraded **PyO3 to v0.27.2** (Modern Bound API refactor).
  - Upgraded **Napi to v3.8.2**.
- **Python Packaging**: Moved from manual `.so` linking to standard `pyproject.toml` and `maturin develop` workflow.
- **Node.js Type Safety**: Refactored `Chain` and `Agent` constructors to use `Either` for robust LLM type handling.

### Fixed
- Resolved PyO3 compilation errors for Python 3.14 by enabling ABI3 support.
- Fixed casing inconsistencies in Node.js class names (`SambaNovaLLM` instead of `SambaNovaLlm`).
- Removed unused imports and addressed Rust compiler warnings across `core`, `python`, and `node` modules.
- Fixed GitHub Actions release job skipping by allowing manual workflow dispatch.

### Testing
- Comprehensive Python test suite (`tests/python/test_providers.py`).
- Comprehensive Node.js test suite (`tests/node/test_providers.js`).
- Rust unit tests for prompt formatting and provider serialization.
