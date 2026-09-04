# Agentic RAG System

A production-grade Retrieval-Augmented Generation pipeline built with LangGraph, featuring adaptive routing, tool-calling agents, query decomposition, relevance evaluation, and self-correcting retrieval loops.

## Architecture Overview

```text
START
  └─► route_question
        ├─► [needs retrieval] check_decomposition
        │     ├─► [complex] decompose_query ──► agent
        │     └─► [simple]                 ──► agent
        │               └─► tools ──► check_retrieval_limit
        │                         ├─► [limit reached] collect_tool_output
        │                         └─► [under limit]   agent (loop)
        │                                   └─► collect_tool_output
        │                                         └─► evaluate_docs
        │                                               ├─► [relevant]     build_prompt
        │                                               └─► [not relevant] rewrite_query
        │                                                         ├─► [< 3 rewrites] agent
        │                                                         └─► [≥ 3 rewrites]  build_prompt
        └─► [no retrieval needed] build_prompt
                                        └─► generate ──► END
```

## Features

- **Adaptive query routing:** Classifies each query and routes it to retrieval or direct generation, avoiding unnecessary retrieval for general-knowledge questions.
- **Query decomposition:** Detects multi-part questions and splits them into focused sub-queries, each handled by a separate retrieval call.
- **Dual-tool agent:** An LLM-powered agent can use both tools:
  - `vector_store_search` searches a local ChromaDB vector store backed by a PDF document.
  - `web_search` searches the web through Tavily for current or real-time information.
- **Retrieval loop with step limit:** The agent can call tools iteratively up to `max_retrieval_steps`, preventing runaway loops.
- **Document relevance evaluation:** Each retrieved passage is individually scored for relevance before it is used in generation.
- **Automatic query rewriting:** If retrieved documents are not relevant, the query is rewritten and retrieval is retried up to three times before generation falls back to general knowledge.
- **Structured prompt construction:** The final generation prompt is built dynamically from vector-store and web results, including source URLs for web results.

## Tech Stack

| Component | Library |
| --- | --- |
| Graph orchestration | `langgraph` |
| LLM | `langchain-openai` using GPT-4o-mini |
| Embeddings | `langchain-openai` using `text-embedding-3-small` |
| Vector store | `langchain-chroma` |
| Web search | `tavily-python` |
| Document loading | `langchain-community` with `PyPDFLoader` |
| Text splitting | `langchain-text-splitters` |
| Structured outputs | `pydantic` |

## Setup

### 1. Install dependencies

```bash
pip install langgraph langchain langchain-openai langchain-chroma langchain-community \
            langchain-text-splitters tavily-python pydantic python-dotenv
```

### 2. Configure environment variables

Create a `.env` file in the project root, or in its parent directory depending on your layout:

```dotenv
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Add your PDF

Place the source document at:

```text
./Documents/evs_oil_price_shock.pdf
```

The document is split into chunks with a size of 1,000 and no overlap, then indexed into `./chroma_db` on first run.

## Usage

### Basic invocation

```python
result = graph.invoke({
    "query": "Your question here",
    "messages": [],
    "retrieved_docs": [],
    "relevant_docs": [],
    "context": "",
    "constructed_prompt": "",
    "retrieval_count": 0,
    "max_retrieval_steps": 3,
    "rewrite_count": 0,
    "rewritten_query": "",
})

print(result["generation"].content)
```

### Example: Simple document query

```python
query = (
    "According to the report, by how many million barrels per day could a "
    "300-million EV fleet displace oil demand by 2030?"
)
result = graph.invoke({"query": query, "messages": [], ...})
```

### Example: Multi-part query

```python
query = (
    "How much have battery pack costs fallen over the past decade, "
    "and what is the current weather in Delhi?"
)
result = graph.invoke({
    "query": query,
    "messages": [],
    "max_retrieval_steps": 6,
    ...
})
```

Use a higher `max_retrieval_steps` value, such as 6 or more, when multi-part queries are expected.

## State Schema

| Field | Type | Description |
| --- | --- | --- |
| `query` | `str` | Active query, which may be rewritten or decomposed. |
| `messages` | `list` | LangGraph message history for the agent. |
| `retrieved_docs` | `list[Document]` | All documents returned by tools. |
| `relevant_docs` | `list[Document]` | Documents that passed relevance evaluation. |
| `context` | `str` | Concatenated text passed to the generator. |
| `constructed_prompt` | `str` | Final system prompt for generation. |
| `generation` | `str` | Model response. |
| `need_retrieval` | `bool` | Whether the router decided to retrieve. |
| `retrieval_count` | `int` | Number of tool-call rounds so far. |
| `max_retrieval_steps` | `int` | Hard ceiling on retrieval rounds. |
| `is_relevant` | `bool` | Whether retrieved documents passed evaluation. |
| `rewritten_query` | `str` | Query after rewriting, when triggered. |
| `rewrite_count` | `int` | Number of rewrite attempts so far. |
| `needs_decomposition` | `bool` | Whether the query was split into sub-queries. |

## Key Design Decisions

### Why a ToolNode agent instead of a fixed retriever?

The agent decides at runtime whether to use the vector store, the web, or both. It can also call tools multiple times within the step limit, which handles decomposed multi-step queries without hardcoding a retrieval path.

### Why per-document relevance evaluation?

Filtering at the document level avoids discarding partially useful retrieval results and gives the rewrite loop a cleaner signal about what failed.

### Why separate `build_prompt` and `generate` nodes?

Decoupling prompt construction from generation makes the final prompt inspectable in state through `constructed_prompt`, which is useful for debugging and evaluation.

### Fallback behavior

After three failed rewrite-retrieve cycles, `build_prompt` emits `FALLBACK` as the constructed prompt. The `generate` node then answers from the LLM's general knowledge instead of returning an error.

## Project Structure

```text
.
├── Documents/
│   └── evs_oil_price_shock.pdf   # Source document
├── chroma_db/                    # Persisted vector store
├── A_Rag.ipynb                   # Main notebook
├── .env                          # API keys; do not commit
└── README.md
```

## Limitations

- The vector store is scoped to a single PDF. Supporting multiple documents requires changes to the loader and the agent's system prompt.
- `max_retrieval_steps` must be increased for decomposed queries; a value of 6 or more is recommended for multi-part questions.
- Relevance evaluation makes one LLM call per retrieved document. Large `k` values therefore add latency and cost.