# LangGraph Practice

This repository contains a collection of Jupyter Notebooks and Python scripts demonstrating the usage of [LangGraph](https://python.langchain.com/docs/langgraph/), a library for building stateful, multi-actor applications with LLMs.

## Project Structure
The `src/` directory is organized into several modules, each focusing on different features, architectures, and design patterns of LangGraph:

### 1. Basics (`1-langgraph-basic/`)
- **`01_simple_langgraph.ipynb`**: Introduction to creating simple state graphs.
- **`02_chatbot.ipynb`**: Building a basic conversational chatbot using LangGraph.

### 2. Components (`2-langgraph-components/`)
- **`01-state-schema-with-dataclasses.ipynb`**: Using dataclasses for state management.
- **`02-pydantic.ipynb`**: Using Pydantic for structured state schemas.
- **`03-chain-using-langgraph.ipynb`**: Implementing standard LangChain chains within a graph.
- **`04-chatbot-with-multiple-tools.ipynb`**: Enhancing the chatbot with tool calling capabilities.
- **`05-ReAct-Architecture.ipynb`**: Implementing the ReAct (Reason + Act) pattern.
- **`06-stream-in-langgraph.ipynb`**: Handling streaming outputs from the graph.

### 3. Debugging (`3-debugging/`)
- **`01-openai-agent.py`**: A Python script demonstrating an OpenAI agent, ready to be used with LangGraph Studio for visual debugging.
- Includes `langgraph.json` configuration for LangGraph CLI/Studio.

### 4. Workflows (`4-workflow/`)
- **`01-prompt-chaining.ipynb`**: Chaining multiple prompts sequentially.
- **`02-parallelization-langgraph.ipynb`**: Executing tasks in parallel within the graph.
- **`03-routing.ipynb`**: Dynamic routing based on LLM decisions.
- **`04-orchestrator.ipynb`**: Building an orchestrator pattern.
- **`05-orchestrate-blog-post.ipynb`**: A practical example of orchestrating a blog post creation.
- **`06-evaluator-optimizer.ipynb`**: Implementing an evaluator-optimizer loop for self-correction.

### 5. Human-in-the-Loop (`5-human-in-the-loop/`)
- **`01-human-in-the-loop.ipynb`**: Adding human approval and intervention steps into the workflow.

### 6. RAG (Retrieval-Augmented Generation) (`6-rag/`)
- **`01-agentic-rag.ipynb`**: Agent-driven RAG implementation.
- **`02-corrective-rag.ipynb`**: Implementing Corrective RAG (CRAG).
- **`03-adaptive-rag.ipynb`**: Implementing Adaptive RAG.

## Getting Started

### Prerequisites
- Python 3.13 or higher.
- API keys for services used in the notebooks (e.g., OpenAI, Groq, Tavily) as required.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd langgraph_practice
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install dependencies:**
   The project dependencies are managed via `pyproject.toml` and `uv.lock`. You can install them using `pip`:
   ```bash
   pip install -e .
   ```
   Or, if you use `uv`:
   ```bash
   uv pip install -e .
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage
Navigate to the desired module within the `src/` directory and open the Jupyter Notebooks to explore the concepts:
```bash
jupyter notebook
```

For the debugging module (`src/3-debugging`), you can use the LangGraph CLI to launch LangGraph Studio and visually interact with the agent:
```bash
cd src/3-debugging
langgraph dev
```

---

*This project is designed for practice and learning advanced LangChain and LangGraph techniques.*
