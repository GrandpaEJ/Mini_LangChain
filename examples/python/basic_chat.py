import os
import asyncio
from mini_langchain import Chain, PromptTemplate, OpenAI, SambaNovaLLM

# Make sure to set your API keys in environment variables:
# export BAM_API_KEY="your-sambanova-key"
# export OPENAI_API_KEY="your-openai-key"

async def main():
    # 1. Initialize LLM
    # You can swap this with OpenAILLM, AnthropicLLM, etc.
    llm = SambaNovaLLM(model="Meta-Llama-3.1-8B-Instruct")
    # llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # 2. Create a prompt template
    prompt = PromptTemplate("Explain the concept of {topic} in a {style} style.")

    # 3. Create a Chain
    chain = Chain(prompt, llm)

    # 4. Invoke the chain
    print("Running chain...")
    response = await chain.invoke({"topic": "Quantum Entanglement", "style": "pirate"})
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
