use napi_derive::napi;
use std::sync::Arc;
use mini_langchain_core::providers::sambanova::SambaNovaProvider;
use mini_langchain_core::providers::openai::OpenAIProvider;
use mini_langchain_core::providers::anthropic::AnthropicProvider;
use mini_langchain_core::providers::google::GoogleGenAIProvider;
use mini_langchain_core::providers::ollama::OllamaProvider;

// --- SambaNova ---
#[napi(js_name = "SambaNovaLLM")]
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
            
        Ok(Self { inner: Arc::new(provider) })
    }
}

// --- OpenAI (and OpenRouter) ---
#[napi(js_name = "OpenAILLM")]
#[derive(Clone)]
pub struct OpenAILLM {
    pub(crate) inner: Arc<OpenAIProvider>,
}

#[napi]
impl OpenAILLM {
    #[napi(constructor)]
    pub fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        system_prompt: Option<String>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Self {
        let provider = OpenAIProvider::new(api_key, model, base_url, system_prompt, temperature, max_tokens);
        Self { inner: Arc::new(provider) }
    }
}

// --- Anthropic ---
#[napi(js_name = "AnthropicLLM")]
#[derive(Clone)]
pub struct AnthropicLLM {
    pub(crate) inner: Arc<AnthropicProvider>,
}

#[napi]
impl AnthropicLLM {
    #[napi(constructor)]
    pub fn new(
        api_key: String,
        model: String,
        system_prompt: Option<String>,
        max_tokens: Option<u32>,
    ) -> Self {
        let provider = AnthropicProvider::new(api_key, model, system_prompt, max_tokens);
        Self { inner: Arc::new(provider) }
    }
}

// --- Google Gemini ---
#[napi(js_name = "GoogleGenAILLM")]
#[derive(Clone)]
pub struct GoogleGenAILLM {
    pub(crate) inner: Arc<GoogleGenAIProvider>,
}

#[napi]
impl GoogleGenAILLM {
    #[napi(constructor)]
    pub fn new(
        api_key: String,
        model: String,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Self {
        let provider = GoogleGenAIProvider::new(api_key, model, temperature, max_tokens);
        Self { inner: Arc::new(provider) }
    }
}

// --- Ollama ---
#[napi(js_name = "OllamaLLM")]
#[derive(Clone)]
pub struct OllamaLLM {
    pub(crate) inner: Arc<OllamaProvider>,
}

#[napi]
impl OllamaLLM {
    #[napi(constructor)]
    pub fn new(
        model: String,
        base_url: Option<String>,
        temperature: Option<f64>,
    ) -> Self {
        let provider = OllamaProvider::new(model, base_url, temperature);
        Self { inner: Arc::new(provider) }
    }
}
