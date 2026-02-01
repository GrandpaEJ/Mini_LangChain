use napi_derive::napi;
use std::sync::Arc;
use std::collections::HashMap;
use napi::Result;
use napi::Error;
use mini_langchain_core::loader::{Loader, TextLoader as CoreTextLoader};
use mini_langchain_core::schema::Document as CoreDocument;
use mini_langchain_core::embedding::{Embeddings, MockEmbeddings as CoreMockEmbeddings};
use mini_langchain_core::vectorstore::{VectorStore, InMemoryVectorStore as CoreInMemoryVectorStore};

#[napi]
pub struct Document {
    pub(crate) inner: CoreDocument,
}

#[napi]
impl Document {
    #[napi(constructor)]
    pub fn new(page_content: String, metadata: Option<HashMap<String, String>>) -> Self {
        let mut doc = CoreDocument::new(page_content);
        if let Some(meta) = metadata {
            doc.metadata = meta;
        }
        Self { inner: doc }
    }

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
pub struct MockEmbeddings {
    pub(crate) inner: Arc<CoreMockEmbeddings>,
}

#[napi]
impl MockEmbeddings {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CoreMockEmbeddings),
        }
    }

    #[napi]
    pub async fn embed_query(&self, text: String) -> Result<Vec<f64>> {
        // Napi doesn't support f32 directly in arrays often, or prefers f64 in JS.
        // Rust vec is f32. conversion needed.
        let  res = self.inner.embed_query(&text).await.map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(res.into_iter().map(|v| v as f64).collect())
    }
}

#[napi]
pub struct InMemoryVectorStore {
    inner: Arc<CoreInMemoryVectorStore>,
}

#[napi]
impl InMemoryVectorStore {
    #[napi(constructor)]
    pub fn new(embeddings: &MockEmbeddings) -> Self {
        Self {
            inner: Arc::new(CoreInMemoryVectorStore::new(embeddings.inner.clone())),
        }
    }

    #[napi]
    pub async fn add_documents(&self, docs: Vec<&Document>) -> Result<Vec<String>> {
         // Need to clone inner documents
         let core_docs: Vec<CoreDocument> = docs.iter().map(|d| d.inner.clone()).collect();
         self.inner.add_documents(&core_docs).await.map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn similarity_search(&self, query: String, k: u32) -> Result<Vec<Document>> {
         let results = self.inner.similarity_search(&query, k as usize).await
            .map_err(|e| Error::from_reason(e.to_string()))?;
            
         Ok(results.into_iter().map(|d| Document { inner: d }).collect())
    }
}
