"""Agent fleet graphs for the LangGraph/LangChain server project."""

from __future__ import annotations

import os
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import json
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
PACKAGE_VERSION = os.getenv("AGENT_PACKAGE_VERSION", "0.2.0")

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

PRODUCTION_RULES = """
Production operating rules:
- Start by identifying the user's goal, the active graph role, and the evidence you have.
- Prefer tool-backed facts over memory or assumptions. If evidence is missing, say what is missing.
- Do not claim local filesystem, shell, VPS, browser, or private repository access from this cloud graph.
- Never ask the user to paste secrets. Refer to secret names only.
- For deploys, writes, account changes, billing, DNS, or infrastructure: produce discovery, preflight, dry-run, approval, apply, and post-check steps.
- Stop after one repeated tool failure. Explain the failure and the exact next corrective action instead of retrying the same call.
- Keep durable memory under /memories/ short, factual, and reusable. Do not store transient package output there.
- When writing files through Deep Agents tools, content must be plain text. JSON/YAML must be serialized as a string first.
- Keep final answers concise and action-oriented; include uncertainty and residual risk.
""".strip()

PRODUCTION_READINESS_CRITERIA = [
    "clear role and scope",
    "explicit tool boundaries",
    "source-grounded answers when facts matter",
    "safe handling of secrets and infrastructure actions",
    "loop prevention after repeated tool errors",
    "memory isolation under /memories/",
    "subagent use only for focused work with complete instructions",
    "human approval for external web fetches or file writes",
]

PACKAGEABLE_AGENT_CATALOG = [
    {
        "graph_id": "agent",
        "label": "Main Deep Agent",
        "package_fit": "production-ready",
        "primary_model": DEFAULT_MODEL,
        "subagents": ["researcher", "critic"],
        "notes": "General production-oriented Deep Agent with planning, memory, researcher, and critic subagents.",
    },
    {
        "graph_id": "react_agent",
        "label": "React Agent",
        "package_fit": "production-starter",
        "primary_model": DEFAULT_MODEL,
        "subagents": [],
        "notes": "Small LangChain tool-calling baseline for arithmetic, time, and project-context questions.",
    },
    {
        "graph_id": "retrieval_agent",
        "label": "Retrieval Agent",
        "package_fit": "production-starter",
        "primary_model": DEFAULT_MODEL,
        "subagents": [],
        "notes": "Project-context retrieval agent. Production limitation: current retrieval source is curated project notes, not yet a vector index.",
    },
    {
        "graph_id": "rag_research",
        "label": "RAG Research Agent",
        "package_fit": "production-ready-for-docs",
        "primary_model": DEFAULT_MODEL,
        "subagents": [],
        "notes": "Retrieval plus allowed official-doc fetching for docs-grounded research.",
    },
    {
        "graph_id": "downloading_agents",
        "label": "Downloading Agents Packaging Advisor",
        "package_fit": "production-ready-for-manifests",
        "primary_model": CODING_MODEL,
        "subagents": [],
        "notes": "Packaging advisor that can inspect known project agents and draft package manifests. It does not create downloadable ZIP files in cloud mode.",
    },
    {
        "graph_id": "deploy_mcp_docs_agent",
        "label": "Deploy MCP Docs Agent",
        "package_fit": "production-ready",
        "primary_model": DEFAULT_MODEL,
        "subagents": ["docs-researcher", "docs-critic"],
        "notes": "Official-docs researcher for LangChain, LangGraph, Deep Agents, and LangSmith deployment.",
    },
    {
        "graph_id": "deploy_coding_agent",
        "label": "Deploy Coding Agent",
        "package_fit": "production-ready-for-planning",
        "primary_model": CODING_MODEL,
        "subagents": ["code-reviewer", "release-manager"],
        "notes": "Guarded coding and deployment planner. It drafts plans and checks but cannot directly change real infrastructure from cloud mode.",
    },
    {
        "graph_id": "async_subagent_server",
        "label": "Async Subagent Server Architect",
        "package_fit": "production-ready-for-architecture",
        "primary_model": DEFAULT_MODEL,
        "subagents": ["api-reviewer"],
        "notes": "Architecture advisor for remote subagent servers, API boundaries, queues, and observability.",
    },
    {
        "graph_id": "deep_research",
        "label": "Deep Research",
        "package_fit": "production-ready",
        "primary_model": RESEARCH_MODEL,
        "subagents": ["researcher", "critic"],
        "notes": "Research planner with source-review and critic subagents plus approved external web fetch.",
    },
    {
        "graph_id": "llm_wiki",
        "label": "LLM Wiki",
        "package_fit": "production-ready-for-memory-notes",
        "primary_model": DEFAULT_MODEL,
        "subagents": ["wiki-auditor"],
        "notes": "Persistent project wiki maintainer for short factual notes under /memories/wiki/.",
    },
]

# Backward-compatible alias for existing callers and old threads.
PACKAGABLE_AGENT_CATALOG = PACKAGEABLE_AGENT_CATALOG


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
def list_agent_fleet() -> str:
    """List all graph agents in this deployment with models, subagents, and readiness status."""
    return json.dumps(PACKAGEABLE_AGENT_CATALOG, ensure_ascii=False, indent=2)


@tool
def agent_readiness_report(graph_id: str = "all") -> str:
    """Return a production-readiness report for one graph ID or the whole fleet."""
    selected = (
        PACKAGEABLE_AGENT_CATALOG
        if graph_id == "all"
        else [item for item in PACKAGEABLE_AGENT_CATALOG if item["graph_id"] == graph_id]
    )
    if not selected:
        known = ", ".join(item["graph_id"] for item in PACKAGEABLE_AGENT_CATALOG)
        return f"Unknown graph_id '{graph_id}'. Known graph IDs: {known}"

    report = {
        "readiness_criteria": PRODUCTION_READINESS_CRITERIA,
        "agents": selected,
        "global_limitations": [
            "Cloud graphs do not have direct Mac/VPS shell access.",
            "retrieval_agent currently uses curated project notes, not a full vector database.",
            "downloadable ZIP export requires an external build/export step outside the cloud graph.",
            "Cloud deployment updates require a LangSmith key with deployment permissions.",
        ],
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


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


@tool
def list_packagable_agents() -> str:
    """List the graph agents currently deployed in this project and their packaging readiness."""
    return json.dumps(PACKAGEABLE_AGENT_CATALOG, ensure_ascii=False, indent=2)


@tool
def list_packageable_agents() -> str:
    """List the graph agents currently deployed in this project and their packaging readiness."""
    return json.dumps(PACKAGEABLE_AGENT_CATALOG, ensure_ascii=False, indent=2)


@tool
def build_agent_package_manifest(graph_id: str) -> str:
    """Create a JSON manifest string for one known graph agent package."""
    match = next(
        (item for item in PACKAGEABLE_AGENT_CATALOG if item["graph_id"] == graph_id),
        None,
    )
    if match is None:
        known = ", ".join(item["graph_id"] for item in PACKAGEABLE_AGENT_CATALOG)
        return f"Unknown graph_id '{graph_id}'. Known graph IDs: {known}"

    manifest = {
        "name": match["label"],
        "graph_id": match["graph_id"],
        "version": PACKAGE_VERSION,
        "package_type": "langgraph-agent",
        "status": match["package_fit"],
        "primary_model": match["primary_model"],
        "subagents": match["subagents"],
        "description": match["notes"],
        "recommended_files": [
            "AGENTS.md",
            "agent.json",
            "README.md",
            "requirements.txt or pyproject.toml",
            "src/",
            "tests/",
        ],
        "exclude": [
            ".env",
            ".env.*",
            ".venv/",
            "node_modules/",
            "__pycache__/",
            ".pytest_cache/",
            ".ruff_cache/",
            "API keys, tokens, and private credentials",
        ],
        "preflight_checks": [
            "validate langgraph.json graph entry exists",
            "run unit tests",
            "run lint",
            "verify environment variable names without printing values",
            "create a rollback note with previous deployment revision",
        ],
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2)


def _prompt(title: str, body: str) -> str:
    return textwrap.dedent(
        f"""
        You are {title}.

        {body}

        General rules:
        {PRODUCTION_RULES}

        Role-specific rules:
        - Be practical, concise, and explicit about uncertainty.
        - Prefer official LangChain/LangGraph/Deep Agents/LangSmith documentation.
        - Never invent secrets, credentials, deployment IDs, or account data.
        - For production or infrastructure changes, provide a discovery/preflight plan first.
        """
    ).strip()


react_agent = create_agent(
    model=DEFAULT_MODEL,
    tools=[
        utc_now,
        add_numbers,
        multiply_numbers,
        retrieve_project_context,
        list_agent_fleet,
        agent_readiness_report,
    ],
    system_prompt=_prompt(
        "a React-style tool-calling LangChain agent",
        (
            "Use a simple think-act-observe loop with a maximum of three tool calls unless "
            "the user explicitly asks for more. Call tools for arithmetic, time, project facts, "
            "fleet status, or readiness checks. If a requested action belongs to a specialized "
            "agent, name the better graph instead of pretending to do that work here."
        ),
    ),
    name="react_agent",
)

retrieval_agent = create_agent(
    model=DEFAULT_MODEL,
    tools=[
        retrieve_project_context,
        list_langchain_doc_sources,
        list_agent_fleet,
        agent_readiness_report,
    ],
    system_prompt=_prompt(
        "a retrieval-focused LangChain agent",
        (
            "Answer from retrieved project context first and label answers as 'confirmed from "
            "project notes' or 'not in current knowledge base'. If the context is insufficient, "
            "say exactly which source should be indexed next. Do not claim full RAG/vector-DB "
            "coverage until an external index is attached."
        ),
    ),
    name="retrieval_agent",
)

rag_research = create_agent(
    model=DEFAULT_MODEL,
    tools=[
        retrieve_project_context,
        list_langchain_doc_sources,
        fetch_allowed_url,
        list_agent_fleet,
        agent_readiness_report,
    ],
    system_prompt=_prompt(
        "a RAG research LangChain agent",
        (
            "Combine retrieval with live source fetching from allowed documentation URLs. "
            "Use retrieved context to decide what to fetch, then summarize grounded findings. "
            "Cite fetched URLs. If a fetch is blocked or irrelevant, stop after one attempt and "
            "explain the missing source."
        ),
    ),
    name="rag_research",
)

downloading_agents = create_deep_agent(
    model=CODING_MODEL,
    tools=[
        utc_now,
        validate_downloadable_agent_layout,
        list_packagable_agents,
        list_packageable_agents,
        list_agent_fleet,
        agent_readiness_report,
        build_agent_package_manifest,
    ],
    backend=_build_backend("downloading_agents"),
    system_prompt=_prompt(
        "a Deep Agents packaging advisor based on the downloading_agents example",
        (
            "Help package agents as folders or zip archives. Use list_packagable_agents "
            "to inspect the agents available in this deployed project; do not infer project "
            "contents from ls('/') because the Deep Agents filesystem is virtual and may only "
            "contain /memories/. Explain AGENTS.md, skills, subagents, agent.json, and what "
            "should or should not be included. If you write JSON files with write_file, the "
            "content argument must always be a JSON string, never a Python dict/object. If a "
            "write_file call fails once, stop retrying the same call and explain the fix. Use "
            "/memories/ only for short reusable packaging rules, never for package output or secrets."
        ),
    ),
    interrupt_on={"write_file": True, "edit_file": True},
    name="downloading_agents",
)

deploy_mcp_docs_agent = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[
        utc_now,
        list_langchain_doc_sources,
        fetch_allowed_url,
        list_agent_fleet,
        agent_readiness_report,
    ],
    backend=_build_backend("deploy_mcp_docs_agent"),
    system_prompt=_prompt(
        "a Deep Agents documentation researcher for LangChain, LangGraph, and Deep Agents",
        (
            "Before answering framework or deployment questions, identify relevant official docs, "
            "fetch allowed sources when useful, and separate confirmed facts from assumptions. "
            "For CLI, deployment, memory, or Agent Server behavior, prefer docs.langchain.com "
            "evidence over memory."
        ),
    ),
    subagents=[
        {
            "name": "docs-researcher",
            "description": "Research official docs and return compact evidence.",
            "model": RESEARCH_MODEL,
            "system_prompt": "Use official docs first. Return URLs, facts, and uncertainty.",
            "tools": [list_langchain_doc_sources, fetch_allowed_url],
        },
        {
            "name": "docs-critic",
            "description": "Check whether an answer is actually supported by official docs.",
            "model": CRITIC_MODEL,
            "system_prompt": (
                "Identify unsupported claims, missing docs links, version ambiguity, and "
                "places where the answer should say 'unknown'."
            ),
            "tools": [utc_now],
        },
    ],
    name="deploy_mcp_docs_agent",
)

deploy_coding_agent = create_deep_agent(
    model=CODING_MODEL,
    tools=[utc_now, retrieve_project_context, list_agent_fleet, agent_readiness_report],
    backend=_build_backend("deploy_coding_agent"),
    system_prompt=_prompt(
        "a guarded Deep Agents coding and deployment planner",
        (
            "Plan code changes, tests, deployment steps, and rollback checks. You may use the "
            "virtual Deep Agents filesystem for drafts, but you do not have direct shell or VPS "
            "control in this cloud graph. Require human approval before real infrastructure changes. "
            "For every implementation plan, include files to inspect, tests to run, and rollback."
        ),
    ),
    subagents=[
        {
            "name": "code-reviewer",
            "description": "Review plans and code-change proposals for risks.",
            "model": CRITIC_MODEL,
            "system_prompt": "Find missing tests, unsafe deploy assumptions, and rollback gaps.",
            "tools": [retrieve_project_context],
        },
        {
            "name": "release-manager",
            "description": "Review deployment, rollback, and post-deploy verification plans.",
            "model": CRITIC_MODEL,
            "system_prompt": (
                "Check release sequencing: discovery, preflight, dry-run, approval, deploy, "
                "post-check, rollback. Flag any missing owner or secret boundary."
            ),
            "tools": [retrieve_project_context, agent_readiness_report],
        },
    ],
    interrupt_on={"write_file": True, "edit_file": True},
    name="deploy_coding_agent",
)

async_subagent_server = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, retrieve_project_context, list_agent_fleet, agent_readiness_report],
    backend=_build_backend("async_subagent_server"),
    system_prompt=_prompt(
        "an async subagent server architect",
        (
            "Help design self-hosted Agent Protocol subagent servers and supervisors. "
            "Explain when to split a subagent into a remote service, what API boundaries matter, "
            "and how to test async delegation. Always include auth, queue/backpressure, timeout, "
            "idempotency, observability, and failure-mode notes."
        ),
    ),
    subagents=[
        {
            "name": "api-reviewer",
            "description": "Review async server API contracts and production failure modes.",
            "model": CRITIC_MODEL,
            "system_prompt": (
                "Review remote-agent designs for auth, retries, cancellation, streaming, "
                "idempotency, schema versioning, and observability gaps."
            ),
            "tools": [agent_readiness_report],
        }
    ],
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
            "ask for real passwords or API keys in chat. Separate findings into confirmed, "
            "likely, and unknown. Do not use more than one failed fetch for the same URL."
        ),
    ),
    subagents=[
        {
            "name": "researcher",
            "description": "Fetch and summarize sources for one focused research question.",
            "model": RESEARCH_MODEL,
            "system_prompt": (
                "Fetch and summarize one focused research question. Return concise evidence "
                "with source URLs, dates when visible, and caveats. Do not make final recommendations."
            ),
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
            "system_prompt": (
                "Flag weak sourcing, overclaims, missing counterevidence, stale information, "
                "and unsupported recommendations."
            ),
            "tools": [utc_now],
        },
    ],
    interrupt_on={"fetch_approved_web_url": True},
    name="deep_research",
)

llm_wiki = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[
        utc_now,
        retrieve_project_context,
        fetch_allowed_url,
        list_agent_fleet,
        agent_readiness_report,
    ],
    backend=_build_backend("llm_wiki"),
    system_prompt=_prompt(
        "a persistent LLM wiki maintainer",
        (
            "Maintain durable project knowledge under /memories/wiki/. For each useful finding, "
            "write short, factual wiki notes with source hints. For questions, read relevant wiki "
            "notes first, then fetch allowed sources only when needed. Never store secrets or "
            "large transcripts. If asked to remember something, summarize it into durable facts."
        ),
    ),
    subagents=[
        {
            "name": "wiki-auditor",
            "description": "Review wiki notes for stale, duplicated, or unsafe memory.",
            "model": CRITIC_MODEL,
            "system_prompt": (
                "Review proposed memory notes. Reject secrets, private credentials, transient "
                "chat noise, unsupported claims, and duplicates."
            ),
            "tools": [utc_now, retrieve_project_context],
        }
    ],
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
