from langgraph.pregel import Pregel

from deep_agent.fleet import ALL_GRAPHS


def test_fleet_graphs_compile() -> None:
    expected = {
        "react_agent",
        "retrieval_agent",
        "rag_research",
        "downloading_agents",
        "deploy_mcp_docs_agent",
        "deploy_coding_agent",
        "async_subagent_server",
        "deep_research",
        "llm_wiki",
    }
    assert set(ALL_GRAPHS) == expected
    assert all(isinstance(graph, Pregel) for graph in ALL_GRAPHS.values())
