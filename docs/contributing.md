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

## CI/CD & Releases

### NPM (Node.js)
To publish to NPM from GitHub Actions (non-interactively), you must use an **Automation Token**:
1. Log in to [npmjs.com](https://www.npmjs.com/).
2. Go to **Access Tokens** -> **Generate New Token**.
3. Select **"Automation"** as the type. This is crucial as it bypasses 2FA challenges during `npm publish`.
4. Copy the token and add it to your GitHub Repository Secrets as `NPM_SECRET_TOKEN`.

### PyPI (Python)
Automated releases to PyPI use a standard API Token:
1. Generate an API token on [pypi.org](https://pypi.org/manage/account/).
2. Add it to GitHub Secrets as `PYPI_API_TOKEN`.

## Creating a Pull Request
1. Fork the repo.
2. Create a feature branch.
3. Commit your changes.
4. Open a PR with a clear description of the "why" and "how".
