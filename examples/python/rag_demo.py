import asyncio
import os
from mini_langchain import (
    Chain, 
    PromptTemplate, 
    OpenAI, 
    SambaNovaLLM,
    TextLoader,
    InMemoryVectorStore,
    MockEmbeddings
)

async def main():
    print("--- RAG Demo ---")
    
    # 1. Create a dummy knowledge base file
    with open("knowledge.txt", "w") as f:
        f.write("Mini LangChain is a high-performance LLM orchestration library.\n")
        f.write("It is written in Rust and provides bindings for Python and Node.js.\n")
        f.write("It supports many providers including SambaNova, OpenAI, and Anthropic.\n")

    # 2. Load Documents
    loader = TextLoader("knowledge.txt")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # 3. Index Documents (using MockEmbeddings for demo purposes)
    # In a real app, use OpenAIEmbeddings or similar
    embeddings = MockEmbeddings() 
    vector_store = InMemoryVectorStore(embeddings)
    await vector_store.add_documents(docs)

    # 4. Retrieve context
    query = "What languages does Mini LangChain support?"
    results = await vector_store.similarity_search(query, k=1)
    context = results[0].page_content
    print(f"\nRetrieved Context: {context}")

    # 5. Generate Answer
    prompt = PromptTemplate(
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    # Using SambaNova for generation
    llm = SambaNovaLLM(model="Meta-Llama-3.1-8B-Instruct")
    chain = Chain(prompt, llm)

    response = await chain.invoke({
        "context": context,
        "question": query
    })

    print("\nAnswer:")
    print(response)

    # Cleanup
    os.remove("knowledge.txt")

if __name__ == "__main__":
    asyncio.run(main())
