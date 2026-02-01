use napi_derive::napi;
use std::sync::Arc;
use mini_langchain_core::providers::sambanova::SambaNovaProvider;

#[napi]
#[derive(Clone)]
pub struct SambaNovaLLM {
    pub(crate) inner: Arc<SambaNovaProvider>,
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
        top_p: Option<f64>
    ) -> napi::Result<Self> {
        let provider = SambaNovaProvider::new(api_key, model, system_prompt, temperature, max_tokens, top_k, top_p)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            
        Ok(Self {
            inner: Arc::new(provider),
        })
    }
}
