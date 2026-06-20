export type AgentOption = {
  id: string;
  name: string;
  description: string;
};

export const AGENT_OPTIONS: AgentOption[] = [
  {
    id: "agent",
    name: "Main Deep Agent",
    description: "General LangGraph/LangChain server assistant",
  },
  {
    id: "deploy_mcp_docs_agent",
    name: "Docs MCP Agent",
    description: "LangChain, LangGraph, Deep Agents docs research",
  },
  {
    id: "deploy_coding_agent",
    name: "Coding Deploy Agent",
    description: "Code and deployment planning with guardrails",
  },
  {
    id: "deep_research",
    name: "Deep Research",
    description: "Research planning, source review, synthesis",
  },
  {
    id: "llm_wiki",
    name: "LLM Wiki",
    description: "Persistent project wiki and reusable notes",
  },
  {
    id: "rag_research",
    name: "RAG Research",
    description: "Retrieval plus allowed source fetching",
  },
  {
    id: "retrieval_agent",
    name: "Retrieval Agent",
    description: "Project-context lookup and answers",
  },
  {
    id: "react_agent",
    name: "React Agent",
    description: "Simple tool-calling baseline",
  },
  {
    id: "downloading_agents",
    name: "Downloading Agents",
    description: "Package and share file-based agents",
  },
  {
    id: "async_subagent_server",
    name: "Async Subagent Server",
    description: "Remote subagent/server architecture advisor",
  },
];

export function getAgentOption(id?: string | null) {
  return AGENT_OPTIONS.find((agent) => agent.id === id);
}
