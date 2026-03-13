"""Tests for the VotingSystem class."""

import pytest
from src.agent import Agent
from src.voting import VotingSystem


@pytest.fixture
def sample_agents():
    """Create sample agents for testing."""
    return [
        Agent(
            agent_id=i,
            belief=[i * 0.1, 1 - i * 0.1],
            influence_tolerance=0.5,
            opinion_strength=0.8
        )
        for i in range(10)
    ]


@pytest.fixture
def candidates():
    """Create candidate list."""
    return ["Candidate_A", "Candidate_B", "Candidate_C"]


def test_voting_system_initialization(sample_agents, candidates):
    """Test voting system initialization."""
    system = VotingSystem(sample_agents, candidates)

    assert len(system.agents) == 10
    assert system.candidates == candidates


def test_approval_vote(sample_agents, candidates):
    """Test approval voting rule."""
    system = VotingSystem(sample_agents, candidates)

    result = system.run_vote(
        agents=sample_agents,
        candidates=candidates,
        rule="approval",
        belief_threshold=0.3
    )

    assert result.abstention_rate >= 0
    assert result.total_votes >= 0


def test_plurality_vote(sample_agents, candidates):
    """Test plurality voting rule."""
    system = VotingSystem(sample_agents, candidates)

    result = system.run_vote(
        agents=sample_agents,
        candidates=candidates,
        rule="plurality",
        belief_threshold=0.3
    )

    assert result.winner in candidates or result.winner is None
    assert result.abstention_rate >= 0


def test_weighted_vote(sample_agents, candidates):
    """Test weighted voting rule."""
    system = VotingSystem(sample_agents, candidates)

    result = system.run_vote(
        agents=sample_agents,
        candidates=candidates,
        rule="weighted_voting",
        belief_threshold=0.3
    )

    assert result.total_votes >= 0
    assert result.abstention_rate >= 0


def test_invalid_voting_rule(sample_agents, candidates):
    """Test that invalid voting rule raises error."""
    system = VotingSystem(sample_agents, candidates)

    with pytest.raises(ValueError):
        system.run_vote(
            agents=sample_agents,
            candidates=candidates,
            rule="invalid_rule"
        )


def test_record_vote(sample_agents, candidates):
    """Test vote recording."""
    system = VotingSystem(sample_agents, candidates)

    agent = sample_agents[0]
    system.record_vote(
        agent_id=agent.agent_id,
        votes=["Candidate_A"],
        abstention=False,
        round_num=1
    )

    assert len(agent.voting_history) == 1
    assert agent.voting_history[0]["round"] == 1
    assert agent.voting_history[0]["vote"] == ["Candidate_A"]
    assert agent.voting_history[0]["abstention"] is False


def test_get_abstention_statistics(sample_agents, candidates):
    """Test abstention statistics."""
    system = VotingSystem(sample_agents, candidates)

    for i, agent in enumerate(sample_agents):
        system.record_vote(
            agent_id=agent.agent_id,
            votes=["Candidate_A"] if i < 5 else [],
            abstention=i >= 5,
            round_num=1
        )

    stats = system.get_abstention_statistics()

    assert stats["total_voters"] == 10
    assert stats["abstentions"] == 5
    assert stats["votes_cast"] == 5
    assert stats["abstention_rate"] == 0.5


def test_reset(sample_agents, candidates):
    """Test resetting the voting system."""
    system = VotingSystem(sample_agents, candidates)

    system.record_vote(0, ["Candidate_A"], False, 1)
    assert len(system._votes) == 1

    system.reset()
    assert len(system._votes) == 0


def test_get_votes_for_candidate(sample_agents, candidates):
    """Test getting votes for a specific candidate."""
    system = VotingSystem(sample_agents, candidates)

    for i, agent in enumerate(sample_agents):
        vote = ["Candidate_A"] if i < 3 else []
        system.record_vote(agent.agent_id, vote, len(vote) == 0, 1)

    votes_for_a = system.get_votes_for_candidate("Candidate_A")
    assert len(votes_for_a) == 3


def test_get_abstention_rate(sample_agents, candidates):
    """Test getting current abstention rate."""
    system = VotingSystem(sample_agents, candidates)

    for i, agent in enumerate(sample_agents):
        system.record_vote(agent.agent_id, [], i >= 5, 1)

    rate = system.get_abstention_rate()
    assert rate == 0.5
