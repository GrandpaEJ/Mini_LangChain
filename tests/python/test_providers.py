import unittest
from mini_langchain import (
    Chain, AgentExecutor, PromptTemplate,
    SambaNovaLLM, OpenAILLM, AnthropicLLM, GoogleGenAILLM, OllamaLLM
)

class TestProviders(unittest.TestCase):
    def setUp(self):
        self.prompt = PromptTemplate("Hello {input}", ["input"])

    def test_sambanova_initialization(self):
        # API Key optional in some contexts or mock
        llm = SambaNovaLLM("Meta-Llama-3-8B", "test-key")
        chain = Chain(self.prompt, llm)
        self.assertIsNotNone(chain)

    def test_openai_initialization(self):
        llm = OpenAILLM("sk-test", "gpt-4o")
        chain = Chain(self.prompt, llm)
        self.assertIsNotNone(chain)
        # Check Agent
        agent = AgentExecutor(llm)
        self.assertIsNotNone(agent)

    def test_anthropic_initialization(self):
        llm = AnthropicLLM("sk-ant-test", "claude-3-opus")
        chain = Chain(self.prompt, llm)
        self.assertIsNotNone(chain)

    def test_google_initialization(self):
        llm = GoogleGenAILLM("AIza-test", "gemini-pro")
        chain = Chain(self.prompt, llm)
        self.assertIsNotNone(chain)

    def test_ollama_initialization(self):
        llm = OllamaLLM("llama3")
        chain = Chain(self.prompt, llm)
        self.assertIsNotNone(chain)

    def test_open_router_via_openai(self):
        # OpenRouter uses OpenAI client with custom URL
        llm = OpenAILLM("sk-or-test", "nous/hermes", "https://openrouter.ai/api/v1")
        chain = Chain(self.prompt, llm)
        self.assertIsNotNone(chain)
    
if __name__ == '__main__':
    print("Running Provider Tests...")
    unittest.main()
