"""Tests for the PatternDetector class."""

import pytest
from src.pattern_detector import PatternDetector


@pytest.fixture
def consensus_voting_data():
    """Create voting data with consensus pattern."""
    return {
        "rounds": [
            {
                "abstention_rate": 0.1,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 50, "Candidate_B": 30, "Candidate_C": 10},
                "belief_variance": [0.01, 0.02]
            },
            {
                "abstention_rate": 0.12,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 48, "Candidate_B": 32, "Candidate_C": 12},
                "belief_variance": [0.015, 0.025]
            },
            {
                "abstention_rate": 0.08,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 52, "Candidate_B": 28, "Candidate_C": 10},
                "belief_variance": [0.012, 0.018]
            }
        ],
        "agents": [
            {"agent_id": i, "belief": [0.3, 0.7], "abstention": i > 92}
            for i in range(100)
        ]
    }


@pytest.fixture
def minority_pressure_voting_data():
    """Create voting data with minority pressure pattern."""
    return {
        "rounds": [
            {
                "abstention_rate": 0.15,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 45, "Candidate_B": 30, "Candidate_C": 15},
                "belief_variance": [0.03, 0.04]
            },
            {
                "abstention_rate": 0.2,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 42, "Candidate_B": 33, "Candidate_C": 18},
                "belief_variance": [0.035, 0.045]
            },
            {
                "abstention_rate": 0.25,
                "winner": "Candidate_B",
                "vote_distribution": {"Candidate_A": 40, "Candidate_B": 35, "Candidate_C": 20},
                "belief_variance": [0.04, 0.05]
            }
        ],
        "agents": [
            {"agent_id": i, "belief": [i * 0.01, 1 - i * 0.01], "abstention": i > 75}
            for i in range(100)
        ]
    }


@pytest.fixture
def oppression_voting_data():
    """Create voting data with oppression pattern."""
    return {
        "rounds": [
            {
                "abstention_rate": 0.2,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 30, "Candidate_B": 25, "Candidate_C": 20},
                "belief_variance": [0.05, 0.06]
            },
            {
                "abstention_rate": 0.3,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 28, "Candidate_B": 27, "Candidate_C": 22},
                "belief_variance": [0.06, 0.07]
            },
            {
                "abstention_rate": 0.35,
                "winner": "Candidate_A",
                "vote_distribution": {"Candidate_A": 25, "Candidate_B": 30, "Candidate_C": 25},
                "belief_variance": [0.07, 0.08]
            }
        ],
        "agents": [
            {
                "agent_id": i,
                "belief": [i * 0.01, 1 - i * 0.01],
                "is_dominant": i == 0,
                "opinion_strength": 0.8 if i == 0 else 0.3,
                "abstention": i > 65
            }
            for i in range(100)
        ]
    }


def test_detect_consensus(consensus_voting_data):
    """Test consensus detection on consensus data."""
    detector = PatternDetector()
    result = detector.detect_consensus(consensus_voting_data)

    assert result.pattern_type == "consensus"
    assert 0 <= result.score <= 1
    assert 0 <= result.confidence <= 1
    assert "num_rounds_analyzed" in result.details


def test_detect_minority_pressure(minority_pressure_voting_data):
    """Test minority pressure detection."""
    detector = PatternDetector()
    result = detector.detect_minority_pressure(minority_pressure_voting_data)

    assert result.pattern_type == "minority_pressure"
    assert 0 <= result.score <= 1
    assert 0 <= result.confidence <= 1


def test_detect_oppression(oppression_voting_data):
    """Test oppression detection."""
    detector = PatternDetector()
    result = detector.detect_oppression(oppression_voting_data)

    assert result.pattern_type == "oppression"
    assert 0 <= result.score <= 1
    assert 0 <= result.confidence <= 1


def test_detect_group_pressure(consensus_voting_data):
    """Test group pressure detection."""
    detector = PatternDetector()
    result = detector.detect_group_pressure(consensus_voting_data)

    assert result.pattern_type == "group_pressure"
    assert 0 <= result.score <= 1
    assert 0 <= result.confidence <= 1


def test_detect_consensus_change(consensus_voting_data):
    """Test consensus change detection."""
    detector = PatternDetector()
    result = detector.detect_consensus_change(consensus_voting_data)

    assert result.pattern_type == "consensus_change"
    assert 0 <= result.score <= 1


def test_analyze_all_patterns(consensus_voting_data):
    """Test all pattern detection combined."""
    detector = PatternDetector()
    results = detector.analyze_all_patterns(consensus_voting_data)

    expected_patterns = [
        "consensus",
        "minority_pressure",
        "oppression",
        "group_pressure",
        "consensus_change"
    ]

    for pattern in expected_patterns:
        assert pattern in results
        assert results[pattern].pattern_type == pattern


def test_empty_voting_data():
    """Test pattern detection with empty data."""
    detector = PatternDetector()

    result = detector.detect_consensus({"rounds": []})
    assert result.score == 0
    assert result.confidence == 0

    result = detector.detect_minority_pressure({"rounds": []})
    assert result.score == 0
    assert result.confidence == 0

    result = detector.detect_oppression({"rounds": []})
    assert result.score == 0
    assert result.confidence == 0


def test_insufficient_data_detection():
    """Test detection with insufficient rounds."""
    detector = PatternDetector()

    # Only 2 rounds (too few for minority pressure which needs 3+)
    data = {"rounds": [{"abstention_rate": 0.2}, {"abstention_rate": 0.3}]}

    result = detector.detect_minority_pressure(data)
    assert result.confidence == 0  # Should have low confidence


def test_pattern_detection_with_realistic_data():
    """Test pattern detection with realistic voting patterns."""
    detector = PatternDetector()

    # Data showing increasing abstention (potential minority pressure)
    data = {
        "rounds": [
            {"abstention_rate": 0.1, "winner": "A", "vote_distribution": {"A": 50, "B": 30}, "belief_variance": [0.02]},
            {"abstention_rate": 0.15, "winner": "A", "vote_distribution": {"A": 48, "B": 32}, "belief_variance": [0.025]},
            {"abstention_rate": 0.2, "winner": "B", "vote_distribution": {"A": 45, "B": 35}, "belief_variance": [0.03]},
        ],
        "agents": [
            {"agent_id": i, "belief": [i * 0.02, 1 - i * 0.02], "abstention": i > 80}
            for i in range(100)
        ]
    }

    result = detector.detect_minority_pressure(data)
    assert result.pattern_type == "minority_pressure"
    # Should detect increasing abstention pattern
    assert result.details.get("abstention_increase", 0) > 0
