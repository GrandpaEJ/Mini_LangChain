use pyo3::prelude::*;
use std::sync::Arc;
use mini_langchain_core::llm::LLM;
use mini_langchain_core::providers::sambanova::SambaNovaProvider;
use async_trait::async_trait;

// --- Wrapper for Python LLMs ---
pub struct PyLLMBridge {
    pub(crate) py_obj: PyObject,
}

#[async_trait]
impl LLM for PyLLMBridge {
    async fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        let prompt_string = prompt.to_string();
        // Clone the python object reference safely using GIL
        let py_obj = Python::with_gil(|py| self.py_obj.clone_ref(py));
        
        // Use tokio::task::spawn_blocking to acquire GIL without blocking the runtime worker
        let output = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let obj = py_obj.bind(py);
                let args = (prompt_string,);
                let result = obj.call_method1("generate", args)?;
                let s: String = result.extract()?;
                Ok::<String, PyErr>(s)
            })
        }).await??;
        
        Ok(output)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct SambaNovaLLM {
    pub(crate) inner: Arc<SambaNovaProvider>,
}

#[pymethods]
impl SambaNovaLLM {
    #[new]
    #[pyo3(signature = (model, api_key=None, system_prompt=None, temperature=None, max_tokens=None, top_k=None, top_p=None))]
    fn new(
        model: String, 
        api_key: Option<String>,
        system_prompt: Option<String>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
        top_k: Option<u32>,
        top_p: Option<f64>
    ) -> PyResult<Self> {
        let provider = SambaNovaProvider::new(api_key, model, system_prompt, temperature, max_tokens, top_k, top_p)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            
        Ok(Self {
            inner: Arc::new(provider),
        })
    }
}
