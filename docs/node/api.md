# Node.js API Reference

The Node.js SDK provides high-performance bindings to the Rust core using Napi-rs.

## Core Classes

### `Chain`
Orchestrates prompts and LLMs.
```javascript
const { Chain, PromptTemplate, OpenAILLM } = require('mini-langchain');

const prompt = new PromptTemplate("What is {topic}?");
const llm = new OpenAILLM("api-key", "gpt-4");
const chain = new Chain(prompt, llm);

const result = await chain.invoke({ topic: "Rust" });
```

### `PromptTemplate`
```javascript
const template = new PromptTemplate("Hello {name}!");
```

### `InMemoryVectorStore`
High-speed vector storage in the Rust layer.
```javascript
const { Document, InMemoryVectorStore, MockEmbeddings } = require('mini-langchain');

const store = new InMemoryVectorStore(new MockEmbeddings());
await store.addDocuments([new Document("Data point")]);
```

## LLM Providers
- `OpenAILLM`
- `AnthropicLLM`
- `SambaNovaLLM`
- `GoogleGenAILLM`
- `OllamaLLM`
