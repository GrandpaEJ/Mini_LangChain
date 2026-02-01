# Rust API Reference

While Mini_LangChain is often used via Python or Node.js, the core can also be used directly in Rust projects.

## Crate Structure
- `mini_langchain_core`: The main logic.
    - `llm`: Traits and implementations for providers.
    - `chain`: Orchestration logic.
    - `vectorstore`: Embedding storage and retrieval.
    - `memory`: Stateful session management.

## Example Usage
```rust
use mini_langchain_core::chain::Chain;
use mini_langchain_core::prompt::PromptTemplate;
use mini_langchain_core::providers::openai::OpenAIProvider;

#[tokio::main]
async fn main() {
    let prompt = PromptTemplate::new("Tell me about {thing}");
    let llm = OpenAIProvider::new("your-key".to_string(), "gpt-4".to_string(), None, None, None, None);
    let chain = Chain::new(prompt, Arc::new(llm));

    let result = chain.invoke(HashMap::from([("thing", "Rust")])).await;
    println!("{}", result.unwrap());
}
```

## Internal Traits
Any new provider can be added by implementing the `LLM` trait:
```rust
#[async_trait]
pub trait LLM: Send + Sync {
    async fn generate(&self, prompt: &str) -> anyhow::Result<String>;
}
```
