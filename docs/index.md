# Mini LangChain

**A high-performance, low-cost LLM framework.**

Mini LangChain is built with a **Rust Core** for speed and cost-efficiency (token minification, strict caching) and exposed via **Python Bindings** for ease of use.

## Key Features

- 🚀 **Rust Core**: Blazing fast execution and memory safety.
- 🐍 **Python Bindings**: Familiar API for Data Scientists and ML Engineers.
- 💰 **Cost Optimized**: 
    - Automatic Prompt Minification (whitespace trimming).
    - Built-in Semantic/Hash Caching.
    - Budget controls (Coming Soon).
- 🧩 **Cross-Provider**: Support for SambaNova, OpenAI (via compatibility), and more.

## Quick Start

```bash
pip install mini_langchain  # (Once published)
```

```python
from mini_langchain import SambaNovaLLM, Chain, PromptTemplate

tmpl = PromptTemplate("Explain {topic}", ["topic"])
llm = SambaNovaLLM(model="Meta-Llama-3.1-8B-Instruct")
chain = Chain(tmpl, llm)

print(chain.invoke({"topic": "Rust"}))
```
