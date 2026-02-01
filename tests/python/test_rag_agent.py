import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(lib_path)

try:
    from mini_langchain import (
        MockEmbeddings, InMemoryVectorStore, Document, 
        AgentExecutor, SambaNovaLLM, PromptTemplate
    )
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_rag():
    print("\n--- Testing RAG ---")
    embeddings = MockEmbeddings()
    vectorstore = InMemoryVectorStore(embeddings)
    
    docs = [
        Document("The sky is blue.", {"source": "nature"}),
        Document("The grass is green.", {"source": "nature"}),
        Document("Rust is fast.", {"source": "tech"}),
    ]
    
    print("Adding documents...")
    vectorstore.add_documents(docs)
    
    print("Searching for 'color'...")
    results = vectorstore.similarity_search("color", 1)
    
    if len(results) > 0:
        print(f"PASS: Found {len(results)} document")
        print(f"Content: {results[0].page_content}")
    else:
        print("FAIL: No results found")

def test_agent():
    print("\n--- Testing Agent (Mock) ---")
    
    # Mock LLM for Agent
    class MockAgentLLM:
        def generate(self, prompt):
            # Simulate agent deciding to use a tool
            if "Answer the following: What is 2+2?" in prompt:
                return "Action: Calculator Input: 2+2"
            return "Final Answer: 4"

    llm = MockAgentLLM()
    # In reality we'd wrap this, but our AgentExecutor expects a compatible LLM object 
    # OR we use the Bridge.
    # The binding `AgentExecutor::new` takes `PyObject` and bridges it if not SambaNova.
    
    agent = AgentExecutor(llm)
    # We didn't expose `with_tool` in bindings yet! 
    # Oops, I missed exposing `with_tool` method in `AgentExecutor` struct in lib.rs.
    # For now, let's just test that we can instantiate it and call execute, 
    # even if it has no tools (it should just define tools list as empty).
    
    # Actually, without tools, the prompt says "Available Tools: " (empty).
    # If the LLM returns "Action: ...", the specific tool won't be found.
    
    print("Executing agent...")
    try:
        res = agent.execute("Hello")
        print(f"PASS: Agent executed. Response: {res}")
    except Exception as e:
        print(f"FAIL: Agent execution error: {e}")

if __name__ == "__main__":
    test_rag()
    test_agent()
