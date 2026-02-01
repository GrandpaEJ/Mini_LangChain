use napi_derive::napi;
use napi::bindgen_prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use mini_langchain_core::prompt::PromptTemplate as CorePromptTemplate;
use mini_langchain_core::chain::LLMChain as CoreLLMChain;
use mini_langchain_core::providers::sambanova::SambaNovaProvider;
use tokio::sync::Mutex;

#[napi]
pub struct PromptTemplate {
    inner: CorePromptTemplate,
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
pub struct ConversationBufferMemory {
    inner: Arc<std::sync::Mutex<mini_langchain_core::memory::ConversationBufferMemory>>,
}

#[napi]
impl ConversationBufferMemory {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(mini_langchain_core::memory::ConversationBufferMemory::new())),
        }
    }
}

#[napi]
pub struct SambaNovaLLM {
    inner: Arc<SambaNovaProvider>,
}

#[napi]
impl SambaNovaLLM {
    #[napi(constructor)]
    pub fn new(
        model: String,
        api_key: Option<String>,
        system_prompt: Option<String>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
        top_k: Option<u32>,
        top_p: Option<f64>,
    ) -> Result<Self> {
        let provider = SambaNovaProvider::new(
            api_key,
            model,
            system_prompt,
            temperature,
            max_tokens,
            top_k,
            top_p,
        ).map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(provider),
        })
    }
}

use mini_langchain_core::loader::{Loader, TextLoader as CoreTextLoader};
use mini_langchain_core::schema::Document as CoreDocument;

#[napi]
pub struct Document {
    inner: CoreDocument,
}

#[napi]
impl Document {
    #[napi(getter)]
    pub fn page_content(&self) -> String {
        self.inner.page_content.clone()
    }
    
    #[napi(getter)]
    pub fn metadata(&self) -> HashMap<String, String> {
        self.inner.metadata.clone()
    }
}

#[napi]
pub struct TextLoader {
    inner: CoreTextLoader,
}

#[napi]
impl TextLoader {
    #[napi(constructor)]
    pub fn new(file_path: String) -> Self {
        Self {
            inner: CoreTextLoader::new(file_path),
        }
    }

    #[napi]
    pub fn load(&self) -> Result<Vec<Document>> {
        let docs = self.inner.load()
            .map_err(|e| Error::from_reason(e.to_string()))?;
            
        Ok(docs.into_iter().map(|d| Document { inner: d }).collect())
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
        // Currently only supporting SambaNovaLLM for Node bindings for simplicity (type safety)
        // To support generic JS objects (like PyLLMBridge), we'd need more napi glue generic trait impl.
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
        let inner = self.inner.clone();
        
        // Napi async function automatically handles the promise.
        // We verify the chain exists inside the mutex.
        let mut guard = inner.lock().await;
        
        if let Some(chain) = guard.as_mut() {
            chain.call(inputs).await.map_err(|e| Error::from_reason(e.to_string()))
        } else {
             Err(Error::from_reason("Chain not initialized".to_string()))
        }
    }
}
