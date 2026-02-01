const { PromptTemplate, Chain, SambaNovaLlm } = require('../../node/index.js');
const fs = require('fs');

const path = require('path');
const envPath = path.resolve(__dirname, '../../.env');

// Load .env if present
if (fs.existsSync(envPath)) {
    const envConfig = fs.readFileSync(envPath, 'utf8');
    envConfig.split('\n').forEach(line => {
        const [key, val] = line.split('=');
        if (key && val) {
            process.env[key.trim()] = val.trim();
            // Fix typo if present
            if (key.trim() === "SAMBANOVA_API_KAY") {
                process.env["SAMBANOVA_API_KEY"] = val.trim();
            }
        }
    });
}

async function main() {
    console.log("--- Mini LangChain Node.js Test ---");

    try {
        // 1. Template
        const tmpl = new PromptTemplate("Hello {name}, how are you?", ["name"]);
        console.log("Template created.");

        // 2. LLM
        // Note: napi-rs handling of Options can be tricky if not strictly typed or if passing undefined.
        // We pass explicit values or undefined.
        const llm = new SambaNovaLlm(
            "Meta-Llama-3.1-8B-Instruct",
            process.env.SAMBANOVA_API_KEY || undefined,
            "You are a helpful assistant.",
            0.7,
            100,
            50,
            0.9
        );
        console.log("LLM created.");

        // 3. Chain
        const chain = new Chain(tmpl, llm);
        console.log("Chain created.");

        // 4. Invoke
        console.log("Invoking chain...");
        // Pass a Map or Object? Our Rust impl expects HashMap<String, String>.
        // napi-rs converts JS Objects to HashMap automatically.
        const input = { "name": "Node.js User" };

        const result = await chain.invoke(input);
        console.log("\nResult:", result);
        console.log("\nSUCCESS: Node.js bindings working!");

    } catch (e) {
        console.error("Error:", e);
        process.exit(1);
    }
}

main();
