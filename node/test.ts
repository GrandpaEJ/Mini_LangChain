import { PromptTemplate, Chain, SambaNovaLlm } from './index';

// Simple Type Check Test
// This file is used to verify that index.d.ts is correctly picked up by tsc

async function main() {
    console.log("--- Mini LangChain TS Type Check ---");

    const tmpl: PromptTemplate = new PromptTemplate("Hello {name}", ["name"]);

    // Check optional types
    const llm: SambaNovaLlm = new SambaNovaLlm("Meta-Llama-3.1-8B-Instruct");

    const chain: Chain = new Chain(tmpl, llm);

    // Explicit type for input map
    const input: Record<string, string> = { "name": "TypeScript" };

    // This part effectively runs logic but primary purpose here is compilation check
    if (process.env.RUN_TS) {
        try {
            const res = await chain.invoke(input);
            console.log("TS Run Result:", res);
        } catch (e) {
            console.error(e);
        }
    }

    console.log("Types confirmed.");
}

main();
