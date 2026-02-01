const fs = require('fs');
const {
    Chain,
    PromptTemplate,
    SambaNovaLLM,
    TextLoader,
    InMemoryVectorStore,
    MockEmbeddings,
    Document
} = require('../../node/index');

// Usage: node rag_demo.js

async function main() {
    console.log("--- RAG Demo ---");

    // 1. Setup Data
    const filename = "node_knowledge.txt";
    fs.writeFileSync(filename, `
  Node.js bindings for Mini LangChain use napi-rs.
  They provide a Promise-based API for async operations.
  You can use standard JS classes like Chain and PromptTemplate.
  `);

    // 2. Load
    const loader = new TextLoader(filename);
    const docs = loader.load();
    console.log(`Loaded ${docs.length} documents.`);

    // 3. Index
    const vectorStore = new InMemoryVectorStore(new MockEmbeddings());
    // Note: addDocuments might be async depending on implementation
    await vectorStore.addDocuments(docs);

    // 4. Retrieve
    const query = "How are Node.js bindings implemented?";
    const results = await vectorStore.similaritySearch(query, 1);
    const context = results[0].pageContent;
    console.log(`\nRetrieved Context: ${context.trim()}`);

    // 5. Generate
    const prompt = new PromptTemplate(
        "Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    );
    const llm = new SambaNovaLLM("Meta-Llama-3.1-8B-Instruct");
    const chain = new Chain(prompt, llm);

    const answer = await chain.invoke({ context, query });
    console.log("\nAnswer:");
    console.log(answer);

    // Cleanup
    fs.unlinkSync(filename);
}

main();
