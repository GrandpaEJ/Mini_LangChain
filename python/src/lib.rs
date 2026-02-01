use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;
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
        Python::with_gil(|py| {
            let obj = self.py_obj.as_ref(py); // as_ref(py) -> &PyAny
            // Assuming the python object has a `generate(prompt)` method
            let args = (prompt,);
            let result = obj.call_method1("generate", args)?;
            let s: String = result.extract()?;
            Ok(s)
        })
        .map_err(|e| anyhow::anyhow!("Python LLM error: {}", e))
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
    inner: Arc<tokio::sync::Mutex<Option<CoreLLMChain>>>, // Wrapped in option/mutex to allow construction
    // CoreLLMChain needs to be Sync/Send.
}

#[pymethods]
impl Chain {
    #[new]
    fn new(prompt: &PromptTemplate, llm_model: PyObject) -> Self {
        // Here we build the chain.
        // 1. Wrap the Python LLM object
        let bridge = PyLLMBridge { py_obj: llm_model };
        let llm = Arc::new(bridge);
        
        let chain = CoreLLMChain::new(prompt.inner.clone(), llm);
        
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(Some(chain))),
        }
    }

    fn set_cache(&self, cache: &InMemoryCache) -> PyResult<()> {
        // This is tricky because CoreLLMChain consumes self in `with_cache`.
        // We'd need to take it out of the mutex, modify it, put it back.
        // For simplicity now, let's just make `with_cache` take &mut or use a builder pattern that returns the Chain.
        // Or we just re-construct. 
        // Actually, let's change `CoreLLMChain` to have a `set_cache` method that takes &mut self, 
        // or just re-wrap here.
        
        // Simpler approach for now: Chain is immutable once built in Rust? 
        // Or let's use the current `with_cache` (consumes self).
        // We can do this in the constructor, OR provide a separate method. 
        // Let's assume we can't easily modify the chain after creation in this naive binding.
        // WAIT: I can just implement `set_cache` on the Py wrapper that requires it be called before `invoke`?
        // Let's implement `with_cache` on the constructor or a builder.
        // Actually, let's just create a `ChainBuilder`? No, simpler.
        
        // Let's support `chain = Chain(prompt, llm, cache=None)`
        Ok(())
    }

    #[pyo3(signature = (inputs))]
    fn invoke(&self, py: Python<'_>, inputs: HashMap<String, String>) -> PyResult<PyObject> {
        let inner_clone = self.inner.clone();
        
        // We need to run the async Rust code.
        // We can use pyo3_asyncio (or similar pattern).
        // For simplicity, let's use a blocking implementation in Python if we don't want to drag in pyo3-asyncio yet.
        // BUT the LLM trait is async. So we MUST run a runtime.
        
        let future = async move {
            let mut guard = inner_clone.lock().await;
            if let Some(chain) = guard.as_ref() {
                chain.call(inputs).await.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err("Chain not initialized"))
            }
        };

        // Execute sync for now using a local runtime (not efficient but easier for "Mini")
        // OR better: use `pyo3_asyncio`. 
        // Let's try to just block_on using tokio (careful with GIL).
        
        let result = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)?;
            
        Ok(result.into_py(py))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn mini_langchain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PromptTemplate>()?;
    m.add_class::<InMemoryCache>()?;
    m.add_class::<Chain>()?;
    Ok(())
}
