use napi_derive::napi;
use std::sync::Arc;
use napi::{Result, Error};
use mini_langchain_core::agent::{AgentExecutor as CoreAgentExecutor};
use crate::llm::SambaNovaLLM;

#[napi]
pub struct AgentExecutor {
    inner: Arc<CoreAgentExecutor>,
}

#[napi]
impl AgentExecutor {
    #[napi(constructor)]
    pub fn new(llm: &SambaNovaLLM) -> Self {
        // Currently only supporting SambaNovaLLM for Node
        Self {
            inner: Arc::new(CoreAgentExecutor::new(llm.inner.clone())),
        }
    }

    #[napi]
    pub async fn execute(&self, input: String) -> Result<String> {
        self.inner.execute(&input).await.map_err(|e| Error::from_reason(e.to_string()))
    }
}
