"""Tests for the InfluenceGraph class."""

import pytest
from src.agent import Agent
from src.influence import InfluenceGraph


@pytest.fixture
def test_agents():
    """Create test agents."""
    return [
        Agent(
            agent_id=i,
            belief=[i * 0.1, 1 - i * 0.1],
            influence_tolerance=0.5,
            opinion_strength=0.8
        )
        for i in range(5)
    ]


def test_influence_graph_initialization():
    """Test influence graph is created empty."""
    graph = InfluenceGraph()
    assert graph.get_node_count() == 0
    assert graph.get_edge_count() == 0


def test_add_agent(test_agents):
    """Test adding agents to graph."""
    graph = InfluenceGraph()
    agent = test_agents[0]

    graph.add_agent(agent)

    assert graph.get_node_count() == 1
    assert graph.get_agent_by_id(0) == agent


def test_add_edge(test_agents):
    """Test adding edges between agents."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    agent1, agent2 = test_agents[0], test_agents[1]

    graph.add_edge(agent1, agent2, weight=0.7)

    assert graph.get_edge_count() == 1


def test_add_edge_by_id(test_agents):
    """Test adding edges using agent IDs."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    graph.add_edge_by_id(0, 1, weight=0.5)

    assert graph.get_edge_count() == 1


def test_get_influence_sources(test_agents):
    """Test getting influence sources."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    # Agent 1 influences agent 2
    graph.add_edge(test_agents[1], test_agents[2], weight=0.6)

    sources = graph.get_influence_sources(test_agents[2])
    assert len(sources) == 1
    assert sources[0][0] == test_agents[1]
    assert sources[0][1] == 0.6


def test_get_influence_targets(test_agents):
    """Test getting influence targets."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    # Agent 1 influences agent 2
    graph.add_edge(test_agents[1], test_agents[2], weight=0.6)

    targets = graph.get_influence_targets(test_agents[1])
    assert len(targets) == 1
    assert targets[0][0] == test_agents[2]
    assert targets[0][1] == 0.6


def test_get_in_degree(test_agents):
    """Test in-degree calculation."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    graph.add_edge(test_agents[0], test_agents[1], weight=0.5)
    graph.add_edge(test_agents[2], test_agents[1], weight=0.3)

    assert graph.get_in_degree(test_agents[1]) == 2


def test_get_out_degree(test_agents):
    """Test out-degree calculation."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    graph.add_edge(test_agents[1], test_agents[0], weight=0.5)
    graph.add_edge(test_agents[1], test_agents[2], weight=0.3)

    assert graph.get_out_degree(test_agents[1]) == 2


def test_remove_edge(test_agents):
    """Test removing an edge."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    graph.add_edge(test_agents[0], test_agents[1], weight=0.5)
    assert graph.get_edge_count() == 1

    graph.remove_edge(test_agents[0], test_agents[1])
    assert graph.get_edge_count() == 0


def test_remove_agent(test_agents):
    """Test removing an agent."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    graph.add_edge(test_agents[0], test_agents[1], weight=0.5)
    graph.add_edge(test_agents[0], test_agents[2], weight=0.3)

    graph.remove_agent(test_agents[0])

    assert graph.get_node_count() == 4
    assert graph.get_agent_by_id(0) is None


def test_get_agents(test_agents):
    """Test getting all agents."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    agents = graph.get_agents()
    assert len(agents) == 5


def test_clear_graph(test_agents):
    """Test clearing the graph."""
    graph = InfluenceGraph()

    for agent in test_agents:
        graph.add_agent(agent)

    graph.add_edge(test_agents[0], test_agents[1], weight=0.5)

    graph.clear()

    assert graph.get_node_count() == 0
    assert graph.get_edge_count() == 0
