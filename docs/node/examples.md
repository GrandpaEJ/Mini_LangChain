# Node.js Examples

Learn how to use Mini_LangChain in your Node.js applications.

## Simple Completion
```javascript
const { Chain, PromptTemplate, SambaNovaLLM } = require('mini-langchain');

async function main() {
  const llm = new SambaNovaLLM("llama3-70b");
  const prompt = new PromptTemplate("Explain {topic} in 5 words.");
  const chain = new Chain(prompt, llm);

  const res = await chain.invoke({ topic: "Async functions" });
  console.log(res);
}

main();
```

## RAG Workflow
```javascript
const { 
  TextLoader, 
  InMemoryVectorStore, 
  MockEmbeddings, 
  Chain, 
  PromptTemplate,
  OpenAILLM 
} = require('mini-langchain');

async function rag() {
  // 1. Load data
  const loader = new TextLoader("./knowledge.txt");
  const docs = loader.load();

  // 2. Index in Rust
  const store = new InMemoryVectorStore(new MockEmbeddings());
  await store.addDocuments(docs);

  // 3. Search and Answer
  const retrieved = await store.similaritySearch("specific topic", 1);
  const context = retrieved[0].pageContent;

  const chain = new Chain(
    new PromptTemplate("Context: {context}\n\nQuestion: {query}"),
    new OpenAILLM("key", "gpt-4")
  );

  console.log(await chain.invoke({ context, query: "summary" }));
}
```
