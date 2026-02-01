use napi_derive::napi;
use std::sync::Arc;
use napi::{Result, Error};
use mini_langchain_core::agent::{AgentExecutor as CoreAgentExecutor};
use mini_langchain_core::llm::LLM;
use crate::llm::{SambaNovaLLM, OpenAILLM, AnthropicLLM, GoogleGenAILLM, OllamaLLM};

#[napi]
pub struct AgentExecutor {
    inner: Arc<CoreAgentExecutor>,
}

use napi::bindgen_prelude::Either;

#[napi]
impl AgentExecutor {
    #[napi(constructor)]
    pub fn new(llm_input: Either<&SambaNovaLLM, Either<&OpenAILLM, Either<&AnthropicLLM, Either<&GoogleGenAILLM, &OllamaLLM>>>>) -> Result<Self> {
        let llm: Arc<dyn LLM> = match llm_input {
            Either::A(samba) => samba.inner.clone(),
            Either::B(rest) => match rest {
                Either::A(openai) => openai.inner.clone(),
                Either::B(rest2) => match rest2 {
                    Either::A(claude) => claude.inner.clone(),
                    Either::B(rest3) => match rest3 {
                        Either::A(gemini) => gemini.inner.clone(),
                        Either::B(ollama) => ollama.inner.clone(),
                    },
                },
            },
        };

        Ok(Self {
            inner: Arc::new(CoreAgentExecutor::new(llm)),
        })
    }

    #[napi]
    pub async fn execute(&self, input: String) -> Result<String> {
        self.inner.execute(&input).await.map_err(|e| Error::from_reason(e.to_string()))
    }
}
