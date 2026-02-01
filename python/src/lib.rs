use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use mini_langchain_core::prompt::PromptTemplate as CorePromptTemplate;
use mini_langchain_core::chain::LLMChain as CoreLLMChain;
use mini_langchain_core::cache::InMemoryCache as CoreInMemoryCache;
use mini_langchain_core::llm::LLM;
use async_trait::async_trait;

// --- Wrapper for Python LLMs ---
struct PyLLMBridge {
    py_obj: PyObject,
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

// --- PyO3 Classes ---

#[pyclass]
struct PromptTemplate {
    inner: CorePromptTemplate,
}

#[pymethods]
impl PromptTemplate {
    #[new]
    fn new(template: String, variables: Vec<String>) -> Self {
        Self {
            inner: CorePromptTemplate::new(&template, variables),
        }
    }

    fn format(&self, values: HashMap<String, String>) -> PyResult<String> {
        self.inner.format(&values).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
struct InMemoryCache {
    inner: Arc<CoreInMemoryCache>,
}

#[pymethods]
impl InMemoryCache {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(CoreInMemoryCache::new()),
        }
    }
}

#[pyclass]
struct Chain {
    inner: Arc<Mutex<Option<CoreLLMChain>>>,
}

use mini_langchain_core::providers::sambanova::SambaNovaProvider;

// ... (Existing Imports)

#[pyclass]
struct SambaNovaLLM {
    inner: Arc<SambaNovaProvider>,
}

#[pymethods]
impl SambaNovaLLM {
    #[new]
    #[pyo3(signature = (model, api_key=None))]
    fn new(model: String, api_key: Option<String>) -> Self {
        Self {
            inner: Arc::new(SambaNovaProvider::new(api_key, model)),
        }
    }
}

// Implement LLM trait for the wrapper so it can be used in Chain
// Wait, CoreLLMChain takes Arc<dyn LLM>. 
// We need to bridge this.
// Actually, `SambaNovaProvider` implements `LLM`. 
// The python `Chain` needs to accept `SambaNovaLLM` as `llm_model` arg.
// Currently `Chain::new` takes `PyObject`.
// If `llm_model` is a `SambaNovaLLM` instance, we can extract the inner Arc.
// BUT `Chain::new` is generic on the python side (takes `PyObject`).

// We need to modify `Chain::new` to check if `llm_model` is a `SambaNovaLLM`
// If so, use its inner provider directly (Pure Rust Path! FAST!)
// If not, fall back to `PyLLMBridge`.

#[pymethods]
impl Chain {
    #[new]
    fn new(py: Python<'_>, prompt: &PromptTemplate, llm_model: PyObject) -> PyResult<Self> {
        // Try to extract SambaNovaLLM
        let llm: Arc<dyn LLM> = if let Ok(samba) = llm_model.extract::<SambaNovaLLM>(py) {
             samba.inner.clone()
        } else {
             // Fallback to Python Bridge
             Arc::new(PyLLMBridge { py_obj: llm_model })
        };
        
        let chain = CoreLLMChain::new(prompt.inner.clone(), llm);
        
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(chain))),
        })
    }
// ...

    fn set_cache(&self, cache: &InMemoryCache) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        if let Some(chain) = guard.take() {
            let new_chain = chain.with_cache(cache.inner.clone());
            *guard = Some(new_chain);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Chain not initialized"))
        }
    }

    #[pyo3(signature = (inputs))]
    fn invoke(&self, py: Python<'_>, inputs: HashMap<String, String>) -> PyResult<String> {
        let inner_clone = self.inner.clone();
        
        let result: Result<String, String> = py.allow_threads(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                let mut chain_opt = inner_clone.lock().unwrap();
                if let Some(chain) = chain_opt.as_mut() {
                    chain.call(inputs).await.map_err(|e| e.to_string())
                } else {
                     Err("Chain not initialized".to_string())
                }
            })
        });

        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

#[pymodule]
fn mini_langchain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PromptTemplate>()?;
    m.add_class::<InMemoryCache>()?;
    m.add_class::<Chain>()?;
    Ok(())
}
