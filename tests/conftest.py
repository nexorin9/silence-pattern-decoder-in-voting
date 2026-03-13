"""Test fixtures for the silence-decoder project."""

import pytest
from src.agent import Agent
from src.influence import InfluenceGraph
from src.voting import VotingSystem
from src.pattern_detector import PatternDetector
from src.simulation import SimulationEngine


@pytest.fixture
def sample_agents():
    """Create sample agents for testing."""
    agents = [
        Agent(
            agent_id=1,
            belief=[0.3, 0.7],
            influence_tolerance=0.5,
            opinion_strength=0.8,
            voting_history=[]
        ),
        Agent(
            agent_id=2,
            belief=[0.6, 0.4],
            influence_tolerance=0.3,
            opinion_strength=0.6,
            voting_history=[]
        ),
        Agent(
            agent_id=3,
            belief=[0.2, 0.8],
            influence_tolerance=0.7,
            opinion_strength=0.9,
            voting_history=[]
        )
    ]
    return agents


@pytest.fixture
def sample_influence_graph(sample_agents):
    """Create sample influence graph."""
    graph = InfluenceGraph()
    for agent in sample_agents:
        graph.add_agent(agent)

    # Create some edges
    graph.add_edge(sample_agents[0], sample_agents[1], weight=0.7)
    graph.add_edge(sample_agents[1], sample_agents[2], weight=0.5)

    return graph


@pytest.fixture
def sample_candidates():
    """Create sample candidates."""
    return ["Candidate_A", "Candidate_B", "Candidate_C"]


@pytest.fixture
def sample_voting_data():
    """Create sample voting data for pattern detection."""
    return {
        "rounds": [
            {
                "abstention_rate": 0.2,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 45, "Candidate_B": 30, "Candidate_C": 15},
                "belief_variance": [0.02, 0.03]
            },
            {
                "abstention_rate": 0.25,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 42, "Candidate_B": 32, "Candidate_C": 18},
                "belief_variance": [0.025, 0.035]
            },
            {
                "abstention_rate": 0.3,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 40, "Candidate_B": 35, "Candidate_C": 20},
                "belief_variance": [0.03, 0.04]
            }
        ],
        "agents": [
            {
                "agent_id": i,
                "belief": [i * 0.1, 1 - i * 0.1],
                "abstention": i > 7
            }
            for i in range(10)
        ]
    }
