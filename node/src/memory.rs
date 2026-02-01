use napi_derive::napi;
use std::sync::{Arc, Mutex};
use mini_langchain_core::memory::{ConversationBufferMemory as CoreBufferMemory};

#[napi]
pub struct ConversationBufferMemory {
    pub(crate) inner: Arc<Mutex<CoreBufferMemory>>,
}

#[napi]
impl ConversationBufferMemory {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CoreBufferMemory::new())),
        }
    }
}
