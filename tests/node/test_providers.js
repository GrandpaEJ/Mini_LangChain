const {
    Chain, AgentExecutor, PromptTemplate,
    SambaNovaLLM, OpenAILLM, AnthropicLLM, GoogleGenAILLM, OllamaLLM
} = require('../../node/index.js');
const assert = require('assert');

console.log("--- Testing Node.js Providers Initialization ---");

const prompt = new PromptTemplate("Hello {input}", ["input"]);

// 1. SambaNova
try {
    const llm = new SambaNovaLLM("Meta-Llama-3", "key");
    assert(llm, "SambaNovaLLM should be created");
    const chain = new Chain(prompt, llm);
    assert(chain, "Chain should accept SambaNovaLLM");
    console.log("PASS: SambaNovaLLM");
} catch (e) {
    console.error("FAIL: SambaNovaLLM", e);
    process.exit(1);
}

// 2. OpenAI
try {
    const llm = new OpenAILLM("sk-key", "gpt-4");
    assert(llm, "OpenAILLM should be created");
    const chain = new Chain(prompt, llm);
    assert(chain, "Chain should accept OpenAILLM");
    // Test Agent acceptance
    const agent = new AgentExecutor(llm);
    assert(agent, "Agent should accept OpenAILLM");
    console.log("PASS: OpenAILLM");
} catch (e) {
    console.error("FAIL: OpenAILLM", e);
    process.exit(1);
}

// 3. Anthropic
try {
    const llm = new AnthropicLLM("sk-ant-key", "claude-3");
    assert(llm, "AnthropicLLM should be created");
    const chain = new Chain(prompt, llm);
    assert(chain, "Chain should accept AnthropicLLM");
    console.log("PASS: AnthropicLLM");
} catch (e) {
    console.error("FAIL: AnthropicLLM", e);
    process.exit(1);
}

// 4. Google
try {
    const llm = new GoogleGenAILLM("key", "gemini-pro");
    assert(llm, "GoogleGenAILLM should be created");
    const chain = new Chain(prompt, llm);
    assert(chain, "Chain should accept GoogleGenAILLM");
    console.log("PASS: GoogleGenAILLM");
} catch (e) {
    console.error("FAIL: GoogleGenAILLM", e);
    process.exit(1);
}

// 5. Ollama
try {
    const llm = new OllamaLLM("llama3");
    assert(llm, "OllamaLLM should be created");
    const chain = new Chain(prompt, llm);
    assert(chain, "Chain should accept OllamaLLM");
    console.log("PASS: OllamaLLM");
} catch (e) {
    console.error("FAIL: OllamaLLM", e);
    process.exit(1);
}

console.log("--- All Node.js Providers Passed ---");
