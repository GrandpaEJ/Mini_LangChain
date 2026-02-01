const { Chain, PromptTemplate, SambaNovaLlm: SambaNovaLLM, ConversationBufferMemory, TextLoader } = require('../../node/index.js');
const path = require('path');

async function testMemory() {
    console.log("\n--- Testing Memory (Node) ---");
    const memory = new ConversationBufferMemory();

    // Mocking LLM by using a real one but we just check if it runs without error for now
    // Or we create a Mock if we extended the bindings to accept JS objects (we didn't yet for Node).
    // So we use a dummy SambaNovaLLM but won't invoke it properly if we don't have a key.
    // Actually, let's just Instantiate connection.
    const llm = new SambaNovaLLM("Meta-Llama-3.1-8B-Instruct", "dummy-key");
    const tmpl = new PromptTemplate("History: {history} Input: {input}", ["history", "input"]);

    console.log("Creating Chain with Memory...");
    const chain = new Chain(tmpl, llm, memory);
    console.log("PASS: Chain created with Memory");
}

async function testLoader() {
    console.log("\n--- Testing TextLoader (Node) ---");
    const filePath = path.join(__dirname, "../items.txt");
    const loader = new TextLoader(filePath);
    const docs = loader.load();

    if (docs.length > 0) {
        console.log(`PASS: Loaded ${docs.length} document`);
        console.log(`Content: ${docs[0].pageContent.trim()}`);
    } else {
        console.log("FAIL: No documents loaded");
    }
}

(async () => {
    try {
        await testMemory();
        await testLoader();
    } catch (e) {
        console.error("FAIL:", e);
    }
})();
