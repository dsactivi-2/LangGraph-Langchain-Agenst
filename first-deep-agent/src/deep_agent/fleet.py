"""Agent fleet graphs for the LangGraph/LangChain server project."""

from __future__ import annotations

import os
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.agents import create_agent
from langchain_core.tools import tool

DEFAULT_MODEL = os.getenv("DEEP_AGENT_MODEL", "ollama:deepseek-v4-flash:cloud")
RESEARCH_MODEL = os.getenv("DEEP_AGENT_RESEARCH_MODEL", DEFAULT_MODEL)
CODING_MODEL = os.getenv("DEEP_AGENT_CODING_MODEL", "ollama:kimi-k2.7-code:cloud")
CRITIC_MODEL = os.getenv("DEEP_AGENT_CRITIC_MODEL", CODING_MODEL)
MEMORY_NAMESPACE = os.getenv("DEEP_AGENT_MEMORY_NAMESPACE", "first-deep-agent")

ALLOWED_FETCH_DOMAINS = {
    "docs.langchain.com",
    "raw.githubusercontent.com",
    "github.com",
}
DEFAULT_APPROVED_WEB_DOMAINS = {
    "docs.elevenlabs.io",
    "elevenlabs.io",
    "elevenlabs.com",
    "openai.com",
    "platform.openai.com",
    "ollama.com",
    "docs.livekit.io",
    "livekit.io",
    "developers.telnyx.com",
    "telnyx.com",
    "developers.deepgram.com",
    "deepgram.com",
}

PROJECT_CONTEXT = [
    {
        "title": "Current deployment",
        "text": (
            "The project deploys a Deep Agent to LangSmith/LangGraph Cloud EU. "
            "The existing deployment contains a production-oriented Deep Agent, "
            "LangSmith tracing, OpenAI and Ollama provider configuration, "
            "StoreBackend memory under /memories/, and TTL rules for store and checkpointer."
        ),
    },
    {
        "title": "Best-practice direction",
        "text": (
            "Prefer one LangGraph deployment with multiple graph IDs and default assistants "
            "before creating many separate deployments. Use LangSmith Studio to inspect runs, "
            "threads, assistants, memory, and traces."
        ),
    },
    {
        "title": "Safety boundary",
        "text": (
            "Cloud agents in this fleet do not receive direct shell, local filesystem, or VPS access. "
            "They can plan, research, draft commands, explain deployment flows, and use their virtual "
            "Deep Agents filesystem. Real infrastructure changes should go through audited preflight "
            "and human approval."
        ),
    },
    {
        "title": "Deep Agents examples",
        "text": (
            "Relevant official examples include deploy-mcp-docs-agent, deploy-coding-agent, "
            "async-subagent-server, deep_research, llm-wiki, and downloading_agents. "
            "This fleet adapts their roles into one LangGraph deployment."
        ),
    },
]

DOC_SOURCES = [
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/deepagents/overview",
    "https://docs.langchain.com/langsmith/studio",
    "https://docs.langchain.com/langsmith/assistants",
    "https://docs.langchain.com/langsmith/cli",
    "https://docs.langchain.com/llms.txt",
]


def _compact(text: str, limit: int = 8000) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n\n[truncated]"


def _agent_memory_namespace(agent_key: str):
    def namespace(_runtime) -> tuple[str, ...]:
        return (MEMORY_NAMESPACE, agent_key, "memories")

    return namespace


def _build_backend(agent_key: str):
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=_agent_memory_namespace(agent_key)),
        },
    )


def _approved_web_domains() -> set[str]:
    configured = os.getenv("APPROVED_WEB_DOMAINS", "")
    domains = {
        domain.strip().lower() for domain in configured.split(",") if domain.strip()
    }
    return domains or DEFAULT_APPROVED_WEB_DOMAINS


def _is_allowed_domain(hostname: str, allowed_domains: set[str]) -> bool:
    hostname = hostname.lower().strip(".")
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in allowed_domains
    )


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def retrieve_project_context(query: str, k: int = 3) -> str:
    """Retrieve project notes relevant to a query."""
    terms = {term.lower() for term in query.split() if len(term) > 2}
    scored: list[tuple[int, str]] = []
    for item in PROJECT_CONTEXT:
        haystack = f"{item['title']} {item['text']}".lower()
        score = sum(1 for term in terms if term in haystack)
        scored.append((score, f"{item['title']}: {item['text']}"))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    selected = [
        text for score, text in scored[: max(1, min(k, len(scored)))] if score > 0
    ]
    if not selected:
        selected = [text for _, text in scored[: max(1, min(k, len(scored)))]]
    return "\n\n".join(selected)


@tool
def list_langchain_doc_sources(topic: str = "overview") -> str:
    """List useful LangChain, LangGraph, Deep Agents, and LangSmith documentation URLs."""
    return "\n".join(f"- {url}" for url in DOC_SOURCES)


@tool
def fetch_allowed_url(url: str, max_chars: int = 8000) -> str:
    """Fetch text from an allowed LangChain/GitHub URL for grounded research."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_FETCH_DOMAINS:
        return "Fetch blocked. Allowed HTTPS domains: " + ", ".join(
            sorted(ALLOWED_FETCH_DOMAINS)
        )

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "langgraph-agent-fleet/0.1"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(max(1024, min(max_chars, 20000)))
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"Fetch failed: {exc}"

    text = raw.decode("utf-8", errors="replace")
    return _compact(text, limit=max(1024, min(max_chars, 12000)))


@tool
def fetch_approved_web_url(url: str, reason: str, max_chars: int = 8000) -> str:
    """Fetch an approved external HTTPS URL after explicit human approval."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    allowed_domains = _approved_web_domains()
    if parsed.scheme != "https" or not hostname:
        return "Fetch blocked. Only absolute HTTPS URLs are allowed."
    if not _is_allowed_domain(hostname, allowed_domains):
        return "Fetch blocked. Approved domains: " + ", ".join(sorted(allowed_domains))
    if not reason.strip():
        return "Fetch blocked. A short reason is required for approval review."

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "langgraph-approved-research/0.1"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get("content-type", "")
            raw = response.read(max(1024, min(max_chars, 20000)))
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"Fetch failed: {exc}"

    text = raw.decode("utf-8", errors="replace")
    return (
        f"Source: {url}\n"
        f"Reason: {reason}\n"
        f"Content-Type: {content_type}\n\n"
        f"{_compact(text, limit=max(1024, min(max_chars, 12000)))}"
    )


@tool
def validate_downloadable_agent_layout(file_listing: str) -> str:
    """Check whether a file listing looks like a downloadable Deep Agents package."""
    files = {
        line.strip().strip("/") for line in file_listing.splitlines() if line.strip()
    }
    has_agents_md = "AGENTS.md" in files or ".deepagents/AGENTS.md" in files
    has_skill = any(
        part.endswith("/SKILL.md") or part == "skills/SKILL.md" for part in files
    )
    has_agent_json = "agent.json" in files or ".deepagents/agent.json" in files

    checks = [
        ("AGENTS.md instructions", has_agents_md),
        ("optional agent.json metadata", has_agent_json),
        ("optional skills/*/SKILL.md", has_skill),
    ]
    lines = [f"{'PASS' if ok else 'WARN'} {name}" for name, ok in checks]
    if has_agents_md:
        lines.append(
            "Package can be treated as a basic file-based Deep Agents package."
        )
    else:
        lines.append(
            "Add AGENTS.md before sharing this as a downloadable agent package."
        )
    return "\n".join(lines)


def _prompt(title: str, body: str) -> str:
    return textwrap.dedent(
        f"""
        You are {title}.

        {body}

        General rules:
        - Be practical, concise, and explicit about uncertainty.
        - Prefer official LangChain/LangGraph/Deep Agents/LangSmith documentation.
        - Never invent secrets, credentials, deployment IDs, or account data.
        - For production or infrastructure changes, provide a discovery/preflight plan first.
        """
    ).strip()


react_agent = create_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, add_numbers, multiply_numbers, retrieve_project_context],
    system_prompt=_prompt(
        "a React-style tool-calling LangChain agent",
        (
            "Use a simple think-act-observe loop. Call tools when arithmetic, time, "
            "or project facts are needed. This graph is the minimal baseline agent."
        ),
    ),
    name="react_agent",
)

retrieval_agent = create_agent(
    model=DEFAULT_MODEL,
    tools=[retrieve_project_context, list_langchain_doc_sources],
    system_prompt=_prompt(
        "a retrieval-focused LangChain agent",
        (
            "Answer from retrieved project context first. If the context is insufficient, "
            "say what source should be added to the knowledge base."
        ),
    ),
    name="retrieval_agent",
)

rag_research = create_agent(
    model=DEFAULT_MODEL,
    tools=[retrieve_project_context, list_langchain_doc_sources, fetch_allowed_url],
    system_prompt=_prompt(
        "a RAG research LangChain agent",
        (
            "Combine retrieval with live source fetching from allowed documentation URLs. "
            "Use retrieved context to decide what to fetch, then summarize grounded findings."
        ),
    ),
    name="rag_research",
)

downloading_agents = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, validate_downloadable_agent_layout],
    backend=_build_backend("downloading_agents"),
    system_prompt=_prompt(
        "a Deep Agents packaging advisor based on the downloading_agents example",
        (
            "Help package agents as folders or zip archives. Explain AGENTS.md, skills, "
            "subagents, agent.json, and what should or should not be included. Use "
            "/memories/ only for reusable packaging rules, never for secrets."
        ),
    ),
    interrupt_on={"write_file": True, "edit_file": True},
    name="downloading_agents",
)

deploy_mcp_docs_agent = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, list_langchain_doc_sources, fetch_allowed_url],
    backend=_build_backend("deploy_mcp_docs_agent"),
    system_prompt=_prompt(
        "a Deep Agents documentation researcher for LangChain, LangGraph, and Deep Agents",
        (
            "Before answering framework or deployment questions, identify relevant official docs, "
            "fetch allowed sources when useful, and separate confirmed facts from assumptions."
        ),
    ),
    subagents=[
        {
            "name": "docs-researcher",
            "description": "Research official docs and return compact evidence.",
            "system_prompt": "Use official docs first. Return URLs, facts, and uncertainty.",
            "tools": [list_langchain_doc_sources, fetch_allowed_url],
        }
    ],
    name="deploy_mcp_docs_agent",
)

deploy_coding_agent = create_deep_agent(
    model=CODING_MODEL,
    tools=[utc_now, retrieve_project_context],
    backend=_build_backend("deploy_coding_agent"),
    system_prompt=_prompt(
        "a guarded Deep Agents coding and deployment planner",
        (
            "Plan code changes, tests, deployment steps, and rollback checks. You may use the "
            "virtual Deep Agents filesystem for drafts, but you do not have direct shell or VPS "
            "control in this cloud graph. Require human approval before real infrastructure changes."
        ),
    ),
    subagents=[
        {
            "name": "code-reviewer",
            "description": "Review plans and code-change proposals for risks.",
            "model": CRITIC_MODEL,
            "system_prompt": "Find missing tests, unsafe deploy assumptions, and rollback gaps.",
            "tools": [retrieve_project_context],
        }
    ],
    interrupt_on={"write_file": True, "edit_file": True},
    name="deploy_coding_agent",
)

async_subagent_server = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, retrieve_project_context],
    backend=_build_backend("async_subagent_server"),
    system_prompt=_prompt(
        "an async subagent server architect",
        (
            "Help design self-hosted Agent Protocol subagent servers and supervisors. "
            "Explain when to split a subagent into a remote service, what API boundaries matter, "
            "and how to test async delegation."
        ),
    ),
    name="async_subagent_server",
)

deep_research = create_deep_agent(
    model=RESEARCH_MODEL,
    tools=[
        utc_now,
        list_langchain_doc_sources,
        fetch_allowed_url,
        fetch_approved_web_url,
    ],
    backend=_build_backend("deep_research"),
    system_prompt=_prompt(
        "a Deep Research agent",
        (
            "Plan research, delegate focused source review when helpful, keep intermediate notes "
            "in temporary files, and synthesize final answers with citations or source URLs. "
            "For non-LangChain external sources, first explain the exact URL and reason, then "
            "call fetch_approved_web_url so the human can approve or reject the access. Never "
            "ask for real passwords or API keys in chat."
        ),
    ),
    subagents=[
        {
            "name": "researcher",
            "description": "Fetch and summarize sources for one focused research question.",
            "model": RESEARCH_MODEL,
            "system_prompt": "Return concise evidence with source URLs and caveats.",
            "tools": [
                list_langchain_doc_sources,
                fetch_allowed_url,
                fetch_approved_web_url,
            ],
            "interrupt_on": {"fetch_approved_web_url": True},
        },
        {
            "name": "critic",
            "description": "Check research conclusions for unsupported claims.",
            "model": CRITIC_MODEL,
            "system_prompt": "Flag weak sourcing, overclaims, and missing counterevidence.",
            "tools": [utc_now],
        },
    ],
    interrupt_on={"fetch_approved_web_url": True},
    name="deep_research",
)

llm_wiki = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, retrieve_project_context, fetch_allowed_url],
    backend=_build_backend("llm_wiki"),
    system_prompt=_prompt(
        "a persistent LLM wiki maintainer",
        (
            "Maintain durable project knowledge under /memories/wiki/. For each useful finding, "
            "write short, factual wiki notes with source hints. For questions, read relevant wiki "
            "notes first, then fetch allowed sources only when needed."
        ),
    ),
    interrupt_on={"write_file": True, "edit_file": True},
    name="llm_wiki",
)


ALL_GRAPHS = {
    "react_agent": react_agent,
    "retrieval_agent": retrieval_agent,
    "rag_research": rag_research,
    "downloading_agents": downloading_agents,
    "deploy_mcp_docs_agent": deploy_mcp_docs_agent,
    "deploy_coding_agent": deploy_coding_agent,
    "async_subagent_server": async_subagent_server,
    "deep_research": deep_research,
    "llm_wiki": llm_wiki,
}
