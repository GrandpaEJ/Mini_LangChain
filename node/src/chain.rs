use napi_derive::napi;
use std::sync::{Arc, Mutex};
use mini_langchain_core::prompt::PromptTemplate as CorePromptTemplate;
use mini_langchain_core::chain::LLMChain as CoreLLMChain;
use mini_langchain_core::llm::LLM;
use std::collections::HashMap;
use napi::{Result, Error};

use crate::llm::{SambaNovaLLM, OpenAILLM, AnthropicLLM, GoogleGenAILLM, OllamaLLM};
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

use napi::bindgen_prelude::Either;

#[napi]
impl Chain {
    #[napi(constructor)]
    pub fn new(
        prompt: &PromptTemplate, 
        llm_input: Either<&SambaNovaLLM, Either<&OpenAILLM, Either<&AnthropicLLM, Either<&GoogleGenAILLM, &OllamaLLM>>>>, 
        memory: Option<&ConversationBufferMemory>
    ) -> Result<Self> {
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

        let mut chain = CoreLLMChain::new(prompt.inner.clone(), llm);
        
        if let Some(mem) = memory {
            let core_mem = mem.inner.lock().unwrap().clone();
             chain = chain.with_memory(Arc::new(core_mem));
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(Some(chain))),
        })
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
