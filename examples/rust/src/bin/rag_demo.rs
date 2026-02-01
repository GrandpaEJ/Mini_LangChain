use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use mini_langchain_core::chain::LLMChain;
use mini_langchain_core::prompt::PromptTemplate;
use mini_langchain_core::providers::sambanova::SambaNovaProvider;
use mini_langchain_core::loader::{TextLoader, Loader};
use mini_langchain_core::vectorstore::{InMemoryVectorStore, VectorStore};
use mini_langchain_core::embedding::MockEmbeddings;

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- Rust RAG Demo ---");

    // 1. Create dummy data
    let filename = "rust_knowledge.txt";
    std::fs::write(filename, 
        "Rust's ownership model guarantees memory safety without garbage collection.\n\
         Mini LangChain's core logic is implemented in Rust for speed.\n\
         It uses Tokio for async runtime."
    )?;

    // 2. Load
    let loader = TextLoader::new(filename.to_string());
    let docs = loader.load()?;
    println!("Loaded {} documents.", docs.len());

    // 3. Index
    let embeddings = Arc::new(MockEmbeddings);
    let store = InMemoryVectorStore::new(embeddings);
    store.add_documents(&docs).await?;

    // 4. Retrieve
    let query = "Why use Rust?";
    let results = store.similarity_search(query, 1).await?;
    let context = results[0].page_content.clone();
    println!("\nRetrieved Context: {}", context.trim());

    // 5. Generate
    let prompt = PromptTemplate::new(
        "Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        vec!["context".to_string(), "query".to_string()]
    );
    
    let llm = Arc::new(SambaNovaProvider::new(
        None,
        "Meta-Llama-3.1-8B-Instruct".to_string(),
        None, None, None, None, None
    )?);

    let chain = LLMChain::new(prompt, llm);

    let mut inputs = HashMap::new();
    inputs.insert("context".to_string(), context);
    inputs.insert("query".to_string(), query.to_string());

    match chain.call(inputs).await {
        Ok(response) => {
            println!("\nAnswer:");
            println!("{}", response);
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    // Cleanup
    let _ = std::fs::remove_file(filename);

    Ok(())
}
