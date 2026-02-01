use napi_derive::napi;
use std::sync::{Arc, Mutex};
use mini_langchain_core::prompt::PromptTemplate as CorePromptTemplate;
use mini_langchain_core::chain::LLMChain as CoreLLMChain;
use std::collections::HashMap;
use napi::Result;
use napi::Error;

use crate::llm::SambaNovaLLM;
use crate::memory::ConversationBufferMemory;

#[napi]
pub struct PromptTemplate {
    pub(crate) inner: CorePromptTemplate,
}

#[napi]
impl PromptTemplate {
    #[napi(constructor)]
    pub fn new(template: String, variables: Vec<String>) -> Self {
        Self {
            inner: CorePromptTemplate::new(&template, variables),
        }
    }

    #[napi]
    pub fn format(&self, values: HashMap<String, String>) -> Result<String> {
        self.inner.format(&values).map_err(|e| Error::from_reason(e.to_string()))
    }
}

#[napi]
pub struct Chain {
    inner: Arc<Mutex<Option<CoreLLMChain>>>,
}

#[napi]
impl Chain {
    #[napi(constructor)]
    pub fn new(prompt: &PromptTemplate, llm: &SambaNovaLLM, memory: Option<&ConversationBufferMemory>) -> Self {
        let mut chain = CoreLLMChain::new(prompt.inner.clone(), llm.inner.clone());
        
        if let Some(mem) = memory {
            let core_mem = mem.inner.lock().unwrap().clone();
             chain = chain.with_memory(Arc::new(core_mem));
        }

        Self {
            inner: Arc::new(Mutex::new(Some(chain))),
        }
    }

    #[napi]
    pub async fn invoke(&self, inputs: HashMap<String, String>) -> Result<String> {
        let inner_clone = self.inner.clone();
        
        let chain = {
            let mut guard = inner_clone.lock().unwrap();
            if let Some(chain) = guard.as_ref() {
                chain.clone()
            } else {
                 return Err(Error::from_reason("Chain not initialized".to_string()));
            }
        };

        chain.call(inputs).await.map_err(|e| Error::from_reason(e.to_string()))
    }
}
