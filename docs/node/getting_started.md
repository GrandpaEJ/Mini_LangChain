# Getting Started with Node.js

## Installation

```bash
npm install mini-langchain-node
```

## Basic Usage

### 1. Templates

```javascript
const { PromptTemplate } = require('mini-langchain-node');

// Whitespace is automatically minified
const tmpl = new PromptTemplate("  Hello {name}   ", ["name"]);
```

### 2. Connect to LLM

```javascript
const { SambaNovaLlm } = require('mini-langchain-node');

const llm = new SambaNovaLlm(
    "Meta-Llama-3.1-8B-Instruct",
    process.env.SAMBANOVA_API_KEY, 
    "You are a helpful assistant." // System Prompt
);
```

### 3. Chains

```javascript
const { Chain } = require('mini-langchain-node');

const chain = new Chain(tmpl, llm);

(async () => {
    const res = await chain.invoke({ "name": "Node User" });
    console.log(res);
})();
```
