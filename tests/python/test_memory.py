import sys
import os

# Ensure we can find the library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(lib_path)

try:
    from mini_langchain import Chain, PromptTemplate, SambaNovaLLM, ConversationBufferMemory, TextLoader, Document
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_memory():
    print("\n--- Testing Memory ---")
    memory = ConversationBufferMemory()
    
    # Mock LLM for cost saving
    class MockLLM:
        def generate(self, prompt):
            return f"Processed: {prompt}"

    llm = MockLLM()
    tmpl = PromptTemplate("History: {history} Input: {input}", ["history", "input"])
    
    # Pass memory to chain
    chain = Chain(tmpl, llm, memory=memory)
    
    # First invoke
    print("Invoking 1...")
    res1 = chain.invoke({"input": "Hi 1"})
    print(f"Result 1: {res1}")

    # Second invoke (should see history)
    print("Invoking 2...")
    res2 = chain.invoke({"input": "Hi 2"})
    print(f"Result 2: {res2}")
    
    if "Human: Hi 1" in res2:
        print("PASS: Memory preserved context")
    else:
        print("FAIL: Memory did not preserve context")

def test_loader():
    print("\n--- Testing TextLoader ---")
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../items.txt"))
    loader = TextLoader(file_path)
    docs = loader.load()
    
    if len(docs) > 0:
        print(f"PASS: Loaded {len(docs)} document")
        print(f"Content: {docs[0].page_content.strip()}")
    else:
        print("FAIL: No documents loaded")

if __name__ == "__main__":
    test_memory()
    test_loader()
