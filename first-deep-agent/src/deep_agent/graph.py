"""Deep Agent graph for deployment."""

from __future__ import annotations

import contextlib
import os
from datetime import datetime, timezone

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime

DEFAULT_MODEL = os.getenv("DEEP_AGENT_MODEL", "ollama:deepseek-v4-flash:cloud")
RESEARCH_MODEL = os.getenv("DEEP_AGENT_RESEARCH_MODEL", DEFAULT_MODEL)
CRITIC_MODEL = os.getenv("DEEP_AGENT_CRITIC_MODEL", "ollama:kimi-k2.7-code:cloud")
MEMORY_NAMESPACE = os.getenv("DEEP_AGENT_MEMORY_NAMESPACE", "first-deep-agent")

SYSTEM_PROMPT = """
You are a production-oriented Deep Agent for a LangGraph/LangChain server project.

Workflow:
1. Write and maintain a todo list for non-trivial requests.
2. Delegate focused fact-finding or review to subagents when it keeps the main context clean.
3. Use temporary files for long intermediate work.
4. Store durable cross-thread notes only under /memories/.
5. Before finalizing, critique your work for risks, gaps, missing tests, and security concerns.
6. Return concise, actionable output.

- Prefer concrete evidence over assumptions.
- State unresolved uncertainty explicitly.
- Treat /memories/ as long-term project memory; keep entries short, factual, and reusable.
- Treat all other file paths as thread-scoped working files.
- Never store secrets, API keys, tokens, or private credentials in memory files.
- Do not claim direct shell, VPS, browser, or local filesystem access from this cloud graph.
- For deploys or infrastructure changes, provide discovery, preflight, dry-run, approval, apply, post-check, and rollback steps.
- If the same tool call fails once, stop retrying it unchanged. Explain the failure and propose the corrected next action.
- When writing files, content must be plain text; serialize JSON or YAML before calling write_file.
- Keep output compact unless the user asks for depth.
""".strip()


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


def _memory_namespace(_runtime) -> tuple[str, ...]:
    """Namespace long-term memory until user-level auth is added."""
    return (MEMORY_NAMESPACE, "memories")


def _build_backend():
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=_memory_namespace),
        },
    )


SUBAGENTS = [
    {
        "name": "researcher",
        "description": "Use for evidence collection and source-grounded fact finding.",
        "model": RESEARCH_MODEL,
        "system_prompt": (
            "You are a focused researcher. Gather evidence, list assumptions, and "
            "report contradictions clearly. Return a compact summary and save only "
            "durable reusable facts under /memories/ when explicitly useful."
        ),
        "tools": [utc_now],
    },
    {
        "name": "critic",
        "description": "Use for adversarial review of drafts and plans.",
        "model": CRITIC_MODEL,
        "system_prompt": (
            "You are a critical reviewer. Find weak logic, untested assumptions, and "
            "missing constraints. Focus on production risks, auth, persistence, "
            "tool permissions, and missing verification."
        ),
        "tools": [utc_now],
        "interrupt_on": {
            "write_file": True,
            "edit_file": True,
        },
    },
]


def _build_agent():
    return create_deep_agent(
        model=DEFAULT_MODEL,
        tools=[utc_now],
        backend=_build_backend(),
        system_prompt=SYSTEM_PROMPT,
        subagents=SUBAGENTS,
        interrupt_on={
            "write_file": True,
            "edit_file": True,
        },
        name="deep_agent",
    )


RO_AGENT = _build_agent()
graph = RO_AGENT


@contextlib.asynccontextmanager
async def get_agent(config: RunnableConfig, runtime: ServerRuntime):
    yield RO_AGENT
