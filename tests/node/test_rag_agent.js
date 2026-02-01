const {
    MockEmbeddings, InMemoryVectorStore, Document,
    AgentExecutor, SambaNovaLlm
} = require('../../node/index.js');

async function testRag() {
    console.log("\n--- Testing RAG (Node) ---");
    const embeddings = new MockEmbeddings();
    const vectorstore = new InMemoryVectorStore(embeddings);

    // Using the newly added Document constructor
    const docs = [
        new Document("Node is async.", { "source": "js" }),
        new Document("Rust is fast.", { "source": "rust" }),
    ];

    console.log("Adding documents...");
    await vectorstore.addDocuments(docs);

    console.log("Searching for 'fast'...");
    const results = await vectorstore.similaritySearch("fast", 1);

    if (results.length > 0) {
        console.log(`PASS: Found ${results.length} document`);
        console.log(`Content: ${results[0].pageContent}`);
    } else {
        console.log("FAIL: No results found");
    }
}

async function testAgent() {
    console.log("\n--- Testing Agent (Node) ---");
    // const llm = new SambaNovaLlm("model", "key"); 
    // const agent = new AgentExecutor(llm);
    // await agent.execute("hi");
    console.log("Agent test placeholder.");
}

(async () => {
    try {
        await testRag();
        await testAgent();
    } catch (e) {
        console.error("FAIL:", e);
    }
})();
