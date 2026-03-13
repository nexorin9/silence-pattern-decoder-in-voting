"""Base data classes for voting agents."""

from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass
class Agent:
    """Represents a voting agent with beliefs and social influence characteristics."""

    agent_id: int
    belief: List[float]
    influence_tolerance: float
    opinion_strength: float
    voting_history: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and normalize attributes after initialization."""
        if not (0 <= self.influence_tolerance <= 1):
            raise ValueError("influence_tolerance must be between 0 and 1")
        if not (0 <= self.opinion_strength <= 1):
            raise ValueError("opinion_strength must be between 0 and 1")
        if len(self.belief) == 0:
            raise ValueError("belief must have at least one dimension")

    def update_belief(self, new_info: List[float], influence_factor: float) -> None:
        """
        Update agent's belief based on new information with social influence.

        Args:
            new_info: New information to incorporate into belief
            influence_factor: Strength of social influence (0 to 1)
        """
        if len(new_info) != len(self.belief):
            raise ValueError("new_info must have same dimensions as belief")

        # Update belief using weighted average of current belief and new info
        for i in range(len(self.belief)):
            # Blend current belief with new info based on influence factor
            self.belief[i] = (
                (1 - influence_factor) * self.belief[i] +
                influence_factor * new_info[i]
            )

    def get_belief_vector(self) -> List[float]:
        """Return a copy of the current belief vector."""
        return self.belief.copy()

    def get_influence_tolerance(self) -> float:
        """Return the agent's tolerance for social influence."""
        return self.influence_tolerance

    def record_vote(self, round_num: int, vote: Any, abstention: bool) -> None:
        """
        Record a vote in the agent's history.

        Args:
            round_num: The voting round number
            vote: The vote cast (can be candidate ID or None for abstention)
            abstention: Whether the agent abstained from voting
        """
        self.voting_history.append({
            "round": round_num,
            "vote": vote,
            "abstention": abstention
        })

    def __repr__(self) -> str:
        """Return string representation of the agent."""
        return (
            f"Agent(id={self.agent_id}, "
            f"belief={self.belief[:3]}..., "
            f"influence_tolerance={self.influence_tolerance:.2f}, "
            f"opinion_strength={self.opinion_strength:.2f}, "
            f"votes={len(self.voting_history)})"
        )

