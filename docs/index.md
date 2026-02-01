# Mini LangChain

<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/master/icons/rust.svg" width="100" />
</p>

> **The high-performance LLM orchestration framework.**  
> Built in Rust. Optimized for Python and Node.js.

---

Mini LangChain is engineered for developers who need **throughput, low latency, and cost-efficiency**. By moving orchestration and RAG logic to a memory-safe Rust core, we eliminate the overhead of traditional interpreted frameworks.

## 🚀 Key Features

- **⚡ Blazing Fast**: Core logic in Rust for sub-millisecond orchestration overhead.
- **💰 Cost-Optimizer**: Automatic whitespace minification and aggressive prompt caching.
- **🔗 Unified SDKs**: Identical API patterns across Python and Node.js.
- **🛠️ Production Ready**: Support for OpenAI, Anthropic, Gemini, SambaNova, and Ollama.

## 🛠️ Quick Start

=== "Python"
    ```bash
    pip install mini-langchain
    ```
    ```python
    from mini_langchain import SambaNovaLLM, Chain, PromptTemplate

    tmpl = PromptTemplate("Explain {topic} in 5 words.")
    llm = SambaNovaLLM(model="Meta-Llama-3.1-8B-Instruct")
    chain = Chain(tmpl, llm)

    print(chain.invoke({"topic": "Rust"}))
    ```

=== "Node.js"
    ```bash
    npm install mini-langchain
    ```
    ```javascript
    const { Chain, PromptTemplate, OpenAILLM } = require('mini-langchain');

    const chain = new Chain(
      new PromptTemplate("Hello {name}!"),
      new OpenAILLM("key", "gpt-4")
    );

    console.log(await chain.invoke({ name: "Alice" }));
    ```

## 📊 Performance

Mini LangChain is designed to handle thousands of tokens per second with minimal CPU footprint. Check our **[Benchmarks](benchmarks/index.md)** to see it in action.
