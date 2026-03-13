"""Tests for the Agent data class."""

import pytest
from src.agent import Agent


def test_agent_initialization():
    """Test agent initialization with valid parameters."""
    agent = Agent(
        agent_id=1,
        belief=[0.3, 0.7],
        influence_tolerance=0.5,
        opinion_strength=0.8,
        voting_history=[]
    )

    assert agent.agent_id == 1
    assert agent.belief == [0.3, 0.7]
    assert agent.influence_tolerance == 0.5
    assert agent.opinion_strength == 0.8
    assert agent.voting_history == []


def test_agent_belief_validation():
    """Test that invalid belief parameters raise errors."""
    with pytest.raises(ValueError):
        Agent(
            agent_id=1,
            belief=[],  # Empty belief
            influence_tolerance=0.5,
            opinion_strength=0.8
        )


def test_agent_influence_tolerance_validation():
    """Test that influence_tolerance must be between 0 and 1."""
    with pytest.raises(ValueError):
        Agent(
            agent_id=1,
            belief=[0.3, 0.7],
            influence_tolerance=-0.1,  # Invalid
            opinion_strength=0.8
        )

    with pytest.raises(ValueError):
        Agent(
            agent_id=1,
            belief=[0.3, 0.7],
            influence_tolerance=1.1,  # Invalid
            opinion_strength=0.8
        )


def test_agent_opinion_strength_validation():
    """Test that opinion_strength must be between 0 and 1."""
    with pytest.raises(ValueError):
        Agent(
            agent_id=1,
            belief=[0.3, 0.7],
            influence_tolerance=0.5,
            opinion_strength=-0.1  # Invalid
        )


def test_update_belief(sample_agents):
    """Test belief update method."""
    agent = sample_agents[0]
    original_belief = agent.belief.copy()

    new_info = [0.5, 0.5]
    agent.update_belief(new_info, influence_factor=0.3)

    # Check that belief was updated
    assert agent.belief != original_belief
    assert agent.belief[0] > original_belief[0]  # Should move toward new_info


def test_update_belief_dimension_mismatch(sample_agents):
    """Test that dimension mismatch raises error."""
    agent = sample_agents[0]
    new_info = [0.5, 0.5, 0.5]  # Wrong dimensions

    with pytest.raises(ValueError):
        agent.update_belief(new_info, influence_factor=0.3)


def test_get_belief_vector(sample_agents):
    """Test that get_belief_vector returns a copy."""
    agent = sample_agents[0]
    belief_copy = agent.get_belief_vector()

    # Modify the copy
    belief_copy[0] = 999

    # Original should be unchanged
    assert agent.belief[0] != 999


def test_record_vote(sample_agents):
    """Test vote recording."""
    agent = sample_agents[0]
    agent.record_vote(round_num=1, vote="Candidate_A", abstention=False)

    assert len(agent.voting_history) == 1
    assert agent.voting_history[0] == {
        "round": 1,
        "vote": "Candidate_A",
        "abstention": False
    }


def test_get_influence_tolerance(sample_agents):
    """Test influence tolerance getter."""
    agent = sample_agents[0]
    assert agent.get_influence_tolerance() == 0.5


def test_agent_repr(sample_agents):
    """Test string representation."""
    agent = sample_agents[0]
    repr_str = repr(agent)

    assert "Agent" in repr_str
    assert "id=1" in repr_str


def test_agent_with_history(sample_agents):
    """Test agent initialization with voting history."""
    history = [
        {"round": 0, "vote": "Candidate_B", "abstention": False}
    ]

    agent = Agent(
        agent_id=4,
        belief=[0.4, 0.6],
        influence_tolerance=0.4,
        opinion_strength=0.7,
        voting_history=history
    )

    assert agent.voting_history == history
