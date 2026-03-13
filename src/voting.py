"""Voting system implementation with multiple voting rules."""

from typing import List, Dict, Any, Optional, Tuple, Union, cast
from dataclasses import dataclass, field
import random
from collections import Counter

from .agent import Agent
from .influence import InfluenceGraph


@dataclass
class VoteResult:
    """Result of a voting round."""

    winner: Any
    runner_up: Any
    vote_counts: Dict[Any, int]
    abstention_count: int
    total_votes: int
    abstention_rate: float
    vote_details: List[dict] = field(default_factory=list)


class VotingSystem:
    """
    Implements multiple voting rules for simulation.

    Supports approval voting, plurality voting, and weighted voting.
    Tracks abstention patterns for analysis.
    """

    def __init__(self, agents: List[Agent], candidates: List[Any]) -> None:
        """
        Initialize the voting system.

        Args:
            agents: List of agents participating in the vote
            candidates: List of candidate options
        """
        self.agents: Dict[int, Agent] = {a.agent_id: a for a in agents}
        self.candidates: List[Any] = list(candidates)
        self._votes: Dict[int, dict] = {}  # agent_id -> vote record

    def run_vote(
        self,
        agents: List[Agent],
        candidates: List[Any],
        rule: str = "approval",
        influence_graph: Optional[InfluenceGraph] = None,
        belief_threshold: float = 0.6
    ) -> VoteResult:
        """
        Run a voting round with the specified rule.

        Args:
            agents: List of participating agents
            candidates: List of candidate options
            rule: Voting rule ('approval', 'plurality', 'weighted_voting')
            influence_graph: Optional graph for influence-based voting
            belief_threshold: Threshold for agent to vote for a candidate

        Returns:
            VoteResult with voting outcomes and statistics
        """
        self.agents = {a.agent_id: a for a in agents}
        self.candidates = list(candidates)

        # Apply influence if graph provided
        if influence_graph:
            self._apply_influence(influence_graph, agents)

        # Collect votes based on rule
        if rule == "approval":
            return self._run_approval_vote(belief_threshold)
        elif rule == "plurality":
            return self._run_plurality_vote(belief_threshold)
        elif rule == "weighted_voting":
            return self._run_weighted_vote(belief_threshold)
        else:
            raise ValueError(f"Unknown voting rule: {rule}")

    def record_vote(
        self,
        agent_id: int,
        votes: List[Any],
        abstention: bool,
        round_num: int
    ) -> None:
        """
        Record a vote from an agent.

        Args:
            agent_id: ID of the voting agent
            votes: List of candidates voted for (empty if abstained)
            abstention: Whether the agent abstained
            round_num: Current voting round number
        """
        agent = self.agents.get(agent_id)
        if agent:
            agent.record_vote(round_num, votes if not abstention else None, abstention)

        self._votes[agent_id] = {
            "round": round_num,
            "votes": votes,
            "abstention": abstention
        }

    def get_abstention_statistics(self) -> Dict[str, Any]:
        """Get statistics about abstention patterns."""
        total = len(self._votes)
        abstained = sum(1 for v in self._votes.values() if v["abstention"])
        return {
            "total_voters": total,
            "abstentions": abstained,
            "votes_cast": total - abstained,
            "abstention_rate": abstained / total if total > 0 else 0.0
        }

    def _apply_influence(
        self,
        influence_graph: InfluenceGraph,
        agents: List[Agent]
    ) -> None:
        """Apply social influence to agent beliefs based on graph."""
        for agent in agents:
            sources = influence_graph.get_influence_sources(agent)
            if sources:
                # Get average belief from influencers
                avg_influence: List[float] = [0.0] * len(agent.belief)
                total_weight = 0

                for influencer, weight in sources:
                    for i, val in enumerate(influencer.belief):
                        avg_influence[i] += val * weight
                    total_weight += weight

                if total_weight > 0:
                    for i in range(len(avg_influence)):
                        avg_influence[i] /= total_weight

                    # Update agent belief with average influence
                    agent.update_belief(avg_influence, 0.3)

    def _run_approval_vote(self, threshold: float) -> VoteResult:
        """Run approval voting - agents can vote for multiple candidates."""
        vote_counts: Counter = Counter()
        abstention_count = 0
        vote_details: List[dict] = []

        for agent in self.agents.values():
            # Determine if agent votes based on belief alignment
            max_alignment = max(
                self._compute_alignment(agent.belief, candidate)
                for candidate in self.candidates
            )

            if max_alignment < threshold:
                # Abstain if no candidate meets threshold
                abstention_count += 1
                vote_details.append({
                    "agent_id": agent.agent_id,
                    "votes": [],
                    "abstention": True,
                    "reason": "no_candidate_meets_threshold"
                })
            else:
                # Vote for all candidates above threshold
                approved: List[Any] = []
                for candidate in self.candidates:
                    alignment = self._compute_alignment(agent.belief, candidate)
                    if alignment >= threshold:
                        approved.append(candidate)

                for candidate in approved:
                    vote_counts[candidate] += 1

                vote_details.append({
                    "agent_id": agent.agent_id,
                    "votes": approved,
                    "abstention": False,
                    "approved_count": len(approved)
                })

        return self._compute_result(vote_counts, abstention_count, vote_details)

    def _run_plurality_vote(self, threshold: float) -> VoteResult:
        """Run plurality voting - agents vote for their most preferred candidate."""
        vote_counts: Counter = Counter()
        abstention_count = 0
        vote_details: List[dict] = []

        for agent in self.agents.values():
            # Find the best candidate
            best_candidate = None
            best_alignment = -1.0

            for candidate in self.candidates:
                alignment = self._compute_alignment(agent.belief, candidate)
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_candidate = candidate

            if best_alignment < threshold:
                abstention_count += 1
                vote_details.append({
                    "agent_id": agent.agent_id,
                    "votes": [],
                    "abstention": True,
                    "best_alignment": best_alignment
                })
            else:
                vote_counts[best_candidate] += 1
                vote_details.append({
                    "agent_id": agent.agent_id,
                    "votes": [best_candidate],
                    "abstention": False,
                    "preferred_alignment": best_alignment
                })

        return self._compute_result(vote_counts, abstention_count, vote_details)

    def _run_weighted_vote(self, threshold: float) -> VoteResult:
        """
        Run weighted voting - votes weighted by opinion strength.

        Agents with stronger opinions have more voting power.
        """
        candidate_weights = {c: 0.0 for c in self.candidates}
        abstention_count = 0
        vote_details = []

        for agent in self.agents.values():
            # Calculate weighted scores for each candidate
            best_candidate = None
            best_score = -1

            for candidate in self.candidates:
                alignment = self._compute_alignment(agent.belief, candidate)
                score = alignment * agent.opinion_strength

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_score < threshold:
                abstention_count += 1
                vote_details.append({
                    "agent_id": agent.agent_id,
                    "votes": [],
                    "abstention": True,
                    "max_score": best_score
                })
            else:
                # Add weighted vote
                candidate_weights[best_candidate] += agent.opinion_strength
                vote_details.append({
                    "agent_id": agent.agent_id,
                    "votes": [best_candidate],
                    "abstention": False,
                    "weighted_score": agent.opinion_strength
                })

        vote_counts: Dict[Any, int] = {c: int(w * 100) for c, w in candidate_weights.items()}
        # Note: Counter is compatible with dict at runtime, but mypy needs explicit cast
        return self._compute_result(cast(Dict[Any, int], vote_counts), abstention_count, vote_details)

    def _compute_alignment(self, belief: List[float], candidate: Any) -> float:
        """
        Compute alignment between agent belief and candidate.

        This is a simplified alignment model where candidates are
        represented by numerical profiles matching belief dimensions.
        """
        # For now, use a simple distance-based alignment
        # In a real system, candidates would have numerical profiles
        if isinstance(candidate, (int, float)):
            candidate_val = float(candidate)
            belief_mean = sum(belief) / len(belief)
            # Convert distance to alignment (0-1)
            distance = abs(belief_mean - candidate_val)
            return max(0.0, 1 - distance)

        # For string candidates, use hash-based pseudo-alignment
        # This ensures consistent but arbitrary alignment
        hash_val = hash(str(candidate)) % 1000
        return (hash_val + sum(belief) * 100) % 1000 / 1000

    def _compute_result(
        self,
        vote_counts: Counter,
        abstention_count: int,
        vote_details: List[dict]
    ) -> VoteResult:
        """Compute final result from vote counts."""
        total_votes = sum(vote_counts.values())
        total_agents = len(self.agents)

        # Get winner and runner-up
        if vote_counts:
            winner = vote_counts.most_common(1)[0][0]
            runner_up = vote_counts.most_common(2)[-1][0] if len(vote_counts) >= 2 else winner
        else:
            winner = None
            runner_up = None

        return VoteResult(
            winner=winner,
            runner_up=runner_up,
            vote_counts=dict(vote_counts),
            abstention_count=abstention_count,
            total_votes=total_votes,
            abstention_rate=abstention_count / total_agents if total_agents > 0 else 0.0,
            vote_details=vote_details
        )

    def reset(self) -> None:
        """Reset the voting system for a new round."""
        self._votes.clear()

    def get_votes_for_candidate(self, candidate: Any) -> List[int]:
        """Get list of agent IDs who voted for a candidate."""
        return [
            agent_id
            for agent_id, vote_data in self._votes.items()
            if not vote_data["abstention"] and candidate in vote_data["votes"]
        ]

    def get_abstention_rate(self) -> float:
        """Get current abstention rate."""
        stats = self.get_abstention_statistics()
        return stats["abstention_rate"]
