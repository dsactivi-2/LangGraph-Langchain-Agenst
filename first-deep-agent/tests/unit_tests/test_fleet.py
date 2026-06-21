from langgraph.pregel import Pregel

from deep_agent.fleet import (
    ALL_GRAPHS,
    PACKAGEABLE_AGENT_CATALOG,
    agent_readiness_report,
    build_agent_package_manifest,
    list_agent_fleet,
)


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


def test_agent_catalog_matches_graphs() -> None:
    catalog_ids = {item["graph_id"] for item in PACKAGEABLE_AGENT_CATALOG}
    assert catalog_ids == {"agent", *ALL_GRAPHS.keys()}
    assert all(item["package_fit"] for item in PACKAGEABLE_AGENT_CATALOG)
    assert all(item["primary_model"] for item in PACKAGEABLE_AGENT_CATALOG)


def test_list_agent_fleet_returns_json_string() -> None:
    result = list_agent_fleet.invoke({})
    assert '"graph_id": "deep_research"' in result
    assert '"primary_model"' in result


def test_package_manifest_serializes_json_as_text() -> None:
    result = build_agent_package_manifest.invoke({"graph_id": "downloading_agents"})
    assert isinstance(result, str)
    assert '"graph_id": "downloading_agents"' in result
    assert '"preflight_checks"' in result


def test_readiness_report_includes_limitations() -> None:
    result = agent_readiness_report.invoke({"graph_id": "all"})
    assert "Cloud graphs do not have direct Mac/VPS shell access" in result
    assert "readiness_criteria" in result
