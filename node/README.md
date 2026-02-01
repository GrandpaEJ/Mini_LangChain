# Mini LangChain (Node.js) ⚡

**High-performance LLM orchestration for Node.js.**  
Powered by a Rust core for maximum throughput and memory safety.

---

## 🚀 Why Mini LangChain for Node?

Traditional LLM frameworks in Node.js often act as thin wrappers around fetch calls. **Mini LangChain** moves the logic—prompt formatting, memory management, and vector similarity search—into **native Rust** via N-API.

- **⚡ Blazing Fast**: Sub-millisecond orchestration overhead.
- **🔗 Native Performance**: Similarity search and token minification happen at native speed.
- **🦾 Robust**: Type-safe bindings for a seamless JavaScript/TypeScript experience.

## 🛠️ Installation

```bash
npm install mini-langchain
```

## 💻 Usage

### Basic Chain
```javascript
const { Chain, PromptTemplate, OpenAILLM } = require('mini-langchain');

async function main() {
  const llm = new OpenAILLM("your-api-key", "gpt-4o");
  const prompt = new PromptTemplate("What is {topic}?");
  const chain = new Chain(prompt, llm);

  const res = await chain.invoke({ topic: "Rust" });
  console.log(res);
}

main();
```

### Conversational Memory
```javascript
const { Chain, PromptTemplate, SambaNovaLLM, ConversationBufferMemory } = require('mini-langchain');

const memory = new ConversationBufferMemory();
const llm = new SambaNovaLLM("Meta-Llama-3.1-8B-Instruct");
const prompt = new PromptTemplate("History: {history}\nUser: {input}");

const chain = new Chain(prompt, llm, memory);

await chain.invoke({ input: "My name is Alice." });
const res = await chain.invoke({ input: "What is my name?" }); // Remembers!
```

## 🔋 Supported Providers
- **OpenAI** (Standard & OpenRouter)
- **Anthropic** (Claude)
- **SambaNova** (Optimized for performance)
- **Google GenAI** (Gemini)
- **Ollama** (Local Inference)

---

## 📚 Documentation
For full API reference and advanced use cases (RAG, Agents), visit our documentation:  
👉 **[Documentation Site](https://grandpaej.github.io/Mini_LangChain/)**

---
*Built with ❤️ by GrandpaEJ*
