const { Chain, PromptTemplate, SambaNovaLLM, OpenAILLM } = require('../../node/index');

// Usage: node basic_chat.js

async function main() {
    console.log("--- Basic Chat Demo ---");

    // 1. Initialize LLM
    // Ensure BAM_API_KEY env var is set
    const llm = new SambaNovaLLM("Meta-Llama-3.1-8B-Instruct");

    // 2. Create Prompt
    const prompt = new PromptTemplate("Draft a tweet about {topic} with hashtags.");

    // 3. Create Chain
    const chain = new Chain(prompt, llm);

    // 4. Invoke
    try {
        const res = await chain.invoke({ topic: "Rust Programming" });
        console.log("\nGenerated Tweet:");
        console.log(res);
    } catch (e) {
        console.error("Error:", e);
    }
}

main();
