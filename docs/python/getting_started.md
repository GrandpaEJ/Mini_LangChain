# Getting Started with Python

## Installation

Ensure you have the shared library (`mini_langchain.so`) available in your python path.

## Basic Usage

### 1. Templates
```python
from mini_langchain import PromptTemplate

# Whitespace is automatically minified to save tokens!
tmpl = PromptTemplate("  Hello {name}   ", ["name"])
```

### 2. Connect to LLM
We support customizable providers.

**SambaNova**:
```python
from mini_langchain import SambaNovaLLM

llm = SambaNovaLLM(
    model="Meta-Llama-3.1-8B-Instruct",
    temperature=0.7,
    max_tokens=200,
    top_k=50
)
```

### 3. Chains & Caching
Combine them into a Chain and enable caching to save money.

```python
from mini_langchain import Chain, InMemoryCache

chain = Chain(tmpl, llm)
chain.set_cache(InMemoryCache())

# First call: Costs Money (API Call)
res = chain.invoke({"name": "Alice"})

# Second call: FREE (Cached)
res = chain.invoke({"name": "Alice"})
```
