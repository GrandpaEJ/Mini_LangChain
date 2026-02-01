#![deny(clippy::all)]

pub mod llm;
pub mod chain;
pub mod memory;
pub mod rag;
pub mod agent;

// Re-export common structs if needed, or let Napi verify logic handle it.
// Actually, Napi automatic discovery might need these pubs to be 'use'd or declared inside modules.
// But `lib.rs` is entry.
// However, Napi usually generates `index.js` based on what is #[napi] exported.
// Since modules have #[napi] items, they should be discovered if they are pub mod?
// Let's verify `napi` crate behavior. Usually yes.
