use async_trait::async_trait;
use anyhow::{Result, anyhow, Context};
use crate::llm::LLM;
use serde_json::json;
use std::env;

pub struct SambaNovaProvider {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl SambaNovaProvider {
    pub fn new(api_key: Option<String>, model: String) -> Self {
        let key = api_key.or_else(|| env::var("SAMBANOVA_API_KEY").ok())
            .expect("SambaNova API Key must be provided or set in SAMBANOVA_API_KEY env var");
            
        Self {
            api_key: key,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LLM for SambaNovaProvider {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let url = "https://api.sambanova.ai/v1/chat/completions";
        
        let body = json!({
            "stream": false,
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant." 
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        let resp = self.client.post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to SambaNova")?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("SambaNova API error: {}", error_text));
        }

        let json_resp: serde_json::Value = resp.json().await
            .context("Failed to parse SambaNova response")?;

        // Extract content from choices[0].message.content
        let content = json_resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow!("Invalid response structure from SambaNova"))?
            .to_string();

        Ok(content)
    }
}
