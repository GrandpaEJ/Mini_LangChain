# Python Examples

Here are some common patterns for using Mini_LangChain in Python.

## Basic LLM Chain
```python
from mini_langchain import Chain, PromptTemplate, SambaNovaLLM

# Fast inference with SambaNova
llm = SambaNovaLLM(model="llama3-70b")
prompt = PromptTemplate("Explain {concept} in one sentence.")
chain = Chain(prompt, llm)

print(chain.invoke({"concept": "Recursion"}))
```

## Simple RAG (Retrieval Augmented Generation)
```python
import asyncio
from mini_langchain import Document, InMemoryVectorStore, MockEmbeddings

async def run_rag():
    # 1. Create Documents
    docs = [
        Document(page_content="Mini_LangChain is built in Rust."),
        Document(page_content="It supports Python and Node.js.")
    ]
    
    # 2. Setup Vector Store
    embeddings = MockEmbeddings()
    vectorstore = InMemoryVectorStore(embeddings)
    
    # 3. Add and Search
    await vectorstore.add_documents(docs)
    results = await vectorstore.similarity_search("What is it built in?", k=1)
    
    for doc in results:
        print(f"Result: {doc.page_content}")

asyncio.run(run_rag())
```

## Conversation with Memory
```python
from mini_langchain import Chain, PromptTemplate, OpenAILLM, ConversationBufferMemory

memory = ConversationBufferMemory()
llm = OpenAILLM(api_key="...", model="gpt-3.5-turbo")
prompt = PromptTemplate("Previous: {history}\nUser: {input}")

chain = Chain(prompt, llm, memory=memory)

# First interaction
chain.invoke({"input": "My name is Alice."})

# Second interaction (remembers name)
print(chain.invoke({"input": "What is my name?"}))
```
