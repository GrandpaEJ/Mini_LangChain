use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use mini_langchain_core::chain::LLMChain;
use mini_langchain_core::prompt::PromptTemplate;
use mini_langchain_core::providers::sambanova::SambaNovaProvider;

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- Rust Basic Chat Demo ---");

    // 1. Initialize Provider
    // Requires BAM_API_KEY environment variable
    let llm = Arc::new(SambaNovaProvider::new(
        None, // api_key (from env)
        "Meta-Llama-3.1-8B-Instruct".to_string(),
        None, // system_prompt
        None, // temperature
        None, // max_tokens
        None, // top_k
        None, // top_p
    )?);

    // 2. Create Prompt
    let prompt = PromptTemplate::new(
        "Write a haiku about {topic}.",
        vec!["topic".to_string()]
    );

    // 3. Create Chain
    let chain = LLMChain::new(prompt, llm);

    // 4. Invoke
    let mut inputs = HashMap::new();
    inputs.insert("topic".to_string(), "The Rust Borrow Checker".to_string());

    match chain.call(inputs).await {
        Ok(response) => {
            println!("\nResponse:");
            println!("{}", response);
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}
