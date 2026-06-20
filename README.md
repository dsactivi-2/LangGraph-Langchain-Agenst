# LangGraph LangChain Agents

This repository contains the LangChain, LangGraph, and Deep Agents project setup.

## Projects

- `first-deep-agent/` - LangGraph Cloud deployment with the Deep Agents fleet.
- `agent-chat-ui/` - Local Agent Chat UI connected to the LangGraph deployment.

## Current Status

- LangGraph Cloud deployment is live in the EU region.
- Ten graph IDs / agents are deployed.
- Ollama Cloud model routing is configured.
- Human-approved web research is available for `deep_research`.
- The local chat UI can select and test all deployed agents.

## Security

Real secrets stay in local `.env` files and are not committed. Use `.env.example`
files as templates.
