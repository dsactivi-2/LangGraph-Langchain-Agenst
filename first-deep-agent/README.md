# Deep Agents Template

Deployment template for a deep agent built with `create_deep_agent(...)`.

## What this template gives you

- A deployable deep agent graph at `src/deep_agent/graph.py`.
- A multi-agent fleet at `src/deep_agent/fleet.py`, exposed as separate LangGraph graph IDs.
- Explicit workflow prompt (plan, delegate, critique, finalize).
- Two predefined sub-agents (`researcher`, `critic`).
- Human-in-the-loop interrupts on `execute` and `write_file`.
- A `uv`-managed local workflow with a small `Makefile` wrapper and starter tests.

## Agent fleet

The deployment exposes these graph IDs:

| Graph ID | Purpose |
| --- | --- |
| `agent` | Main production-oriented Deep Agent. |
| `react_agent` | Simple LangChain tool-calling baseline agent. |
| `retrieval_agent` | Project-context retrieval agent. |
| `rag_research` | RAG-style research agent with allowed source fetching. |
| `downloading_agents` | Deep Agents packaging/downloading advisor. |
| `deploy_mcp_docs_agent` | Docs-focused Deep Agent for LangChain/LangGraph/Deep Agents. |
| `deploy_coding_agent` | Guarded coding and deployment planning Deep Agent. |
| `async_subagent_server` | Async subagent server architecture advisor. |
| `deep_research` | Deep Research style agent with research and critique subagents. |
| `llm_wiki` | Persistent project wiki maintainer using `/memories/wiki/`. |

## Prerequisites

- An API key for your model provider (Ollama Cloud by default)
- A [LangSmith](https://smith.langchain.com/) account (Plus plan or higher) to deploy

## Model routing

Default model assignment:

| Role | Model |
| --- | --- |
| Main/default agents | `ollama:deepseek-v4-flash:cloud` |
| Long-context research | `ollama:deepseek-v4-flash:cloud` |
| Coding/deploy planning | `ollama:kimi-k2.7-code:cloud` |
| Critic/review subagents | `ollama:kimi-k2.7-code:cloud` |

These can be changed with `DEEP_AGENT_MODEL`, `DEEP_AGENT_RESEARCH_MODEL`,
`DEEP_AGENT_CODING_MODEL`, and `DEEP_AGENT_CRITIC_MODEL`.

## Human-approved web research

`deep_research` has a separate `fetch_approved_web_url` tool for selected
external domains such as ElevenLabs, OpenAI, Ollama, LiveKit, Telnyx, and
Deepgram. This tool is configured with human-in-the-loop approval, so the run
pauses before the external fetch executes. Do not put real passwords, API keys,
or private secrets into chat messages.

The approved domain list is controlled by `APPROVED_WEB_DOMAINS`.

## Quickstart

1. Sync the project and configure environment:

```bash
uv sync
cp .env.example .env
```

2. Start the dev server:

```bash
uv run langgraph dev
```

3. Deploy to LangSmith:

```bash
uv run langgraph deploy
```

See the [CLI docs](https://docs.langchain.com/langsmith/cli#deploy) for deploy options.

To set up CI instead, push this repo to GitHub and configure your deployment through the LangSmith UI.

## Tests and lint

```bash
make test
make integration-tests
make lint
make format
```

Integration tests are skipped unless a configured model-provider key is set.

## Reference docs

- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents quickstart: https://docs.langchain.com/oss/python/deepagents/quickstart
- LangSmith CLI: https://docs.langchain.com/langsmith/cli
