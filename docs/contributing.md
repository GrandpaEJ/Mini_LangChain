# Contributing

We welcome contributions to Mini_LangChain! Whether it's adding a new LLM provider, fixing a bug, or improving documentation.

## Development Setup

### Prerequisite
- Rust (latest stable)
- Node.js (v18+)
- Python (3.9+)

### Project Structure
- `/core`: The Rust source code.
- `/python`: PyO3 bindings.
- `/node`: Napi-rs bindings.
- `/docs`: Documentation source.

### Testing
Each layer has its own test suite.

#### Rust Core
```bash
cd core
cargo test
```

#### Python SDK
```bash
# Requires maturin
maturin develop
pytest tests/python
```

#### Node.js SDK
```bash
cd node
npm install
npm run build
npm test
```

## Creating a Pull Request
1. Fork the repo.
2. Create a feature branch.
3. Commit your changes.
4. Open a PR with a clear description of the "why" and "how".
