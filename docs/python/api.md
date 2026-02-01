# Python API Reference

Mini_LangChain's Python SDK is built on a high-performance Rust core, exposed via PyO3.

## Core Components

### `Chain`
The central orchestration unit.
```python
from mini_langchain import Chain, PromptTemplate, OpenAILLM

prompt = PromptTemplate("What is {topic}?")
llm = OpenAILLM(api_key="...", model="gpt-4o")
chain = Chain(prompt, llm)

result = chain.invoke({"topic": "Rust"})
```

### `PromptTemplate`
Handles input orchestration and variable injection.
```python
template = PromptTemplate("Translate {text} to {language}")
```

### `Document` & `VectorStore`
For RAG workflows.
```python
from mini_langchain import Document, InMemoryVectorStore, MockEmbeddings

doc = Document(page_content="Rust is fast.")
vectorstore = InMemoryVectorStore(MockEmbeddings())
vectorstore.add_documents([doc])
```

## LLM Providers
- `OpenAILLM`
- `AnthropicLLM`
- `SambaNovaLLM` (Optimized for tokens/sec)
- `GoogleGenAILLM`
- `OllamaLLM` (Local inference)
