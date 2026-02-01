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

use mini_langchain_core::memory::{ConversationBufferMemory as CoreBufferMemory, Memory};

#[pyclass]
struct ConversationBufferMemory {
    inner: Arc<Mutex<CoreBufferMemory>>, // Wrapper for Python safety
}

#[pymethods]
impl ConversationBufferMemory {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CoreBufferMemory::new())),
        }
    }
}

use mini_langchain_core::loader::{Loader, TextLoader as CoreTextLoader};
use mini_langchain_core::schema::Document as CoreDocument;

#[pyclass]
struct Document {
    inner: CoreDocument,
}

#[pymethods]
impl Document {
    #[getter]
    fn page_content(&self) -> String {
        self.inner.page_content.clone()
    }
    
    #[getter]
    fn metadata(&self) -> HashMap<String, String> {
        self.inner.metadata.clone()
    }
}

#[pyclass]
struct TextLoader {
    inner: CoreTextLoader,
}

#[pymethods]
impl TextLoader {
    #[new]
    fn new(file_path: String) -> Self {
        Self {
            inner: CoreTextLoader::new(file_path),
        }
    }

    fn load(&self) -> PyResult<Vec<Document>> {
        let docs = self.inner.load()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            
        Ok(docs.into_iter().map(|d| Document { inner: d }).collect())
    }
}

#[pyclass]
struct Chain {
    inner: Arc<Mutex<Option<CoreLLMChain>>>,
}

use mini_langchain_core::providers::sambanova::SambaNovaProvider;

// ... (Existing Imports)

#[pyclass]
#[derive(Clone)]
struct SambaNovaLLM {
    inner: Arc<SambaNovaProvider>,
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
    #[pyo3(signature = (prompt, llm_model, memory=None))]
    fn new(py: Python<'_>, prompt: &PromptTemplate, llm_model: PyObject, memory: Option<&ConversationBufferMemory>) -> PyResult<Self> {
        // Try to extract SambaNovaLLM
        let llm: Arc<dyn LLM> = if let Ok(samba) = llm_model.extract::<SambaNovaLLM>(py) {
             samba.inner.clone()
        } else {
             // Fallback to Python Bridge
             Arc::new(PyLLMBridge { py_obj: llm_model })
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

use mini_langchain_core::token::TokenCounter;

#[pyclass]
struct TokenCalculator;

#[pymethods]
impl TokenCalculator {
    #[staticmethod]
    fn count(text: &str) -> usize {
        TokenCounter::count(text)
    }

    #[staticmethod]
    fn estimate_cost(text: &str, rate_per_1k: f64) -> f64 {
        TokenCounter::estimate_cost(text, rate_per_1k)
    }
}

#[pymodule]
fn mini_langchain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PromptTemplate>()?;
    m.add_class::<InMemoryCache>()?;
    m.add_class::<Chain>()?;
    m.add_class::<SambaNovaLLM>()?;
    m.add_class::<ConversationBufferMemory>()?;
    m.add_class::<Document>()?;
    m.add_class::<TextLoader>()?;
    m.add_class::<TokenCalculator>()?;
    Ok(())
}
