"""Monte Carlo simulation engine for voting behavior."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import random
import numpy as np

from .agent import Agent
from .influence import InfluenceGraph
from .voting import VotingSystem


@dataclass
class SimulationRound:
    """Result of a single simulation round."""

    round_num: int
    abstention_rate: float
    winner: Any
    vote_distribution: Dict[Any, int]
    belief_mean: List[float]
    belief_variance: List[float]


@dataclass
class SimulationResult:
    """Complete simulation results."""

    num_agents: int
    num_rounds: int
    num_candidates: int
    rounds: List[SimulationRound] = field(default_factory=list)
    final_abstention_rate: float = 0.0
    avg_abstention_rate: float = 0.0
    consensus_score: float = 0.0
    minority_pressure: float = 0.0


class SimulationEngine:
    """
    Monte Carlo simulation engine for voting behavior analysis.

    Runs multiple simulation iterations with random influence patterns
    to study abstention dynamics and group behavior.
    """

    def __init__(
        self,
        num_agents: int = 100,
        num_candidates: int = 3,
        num_belief_dimensions: int = 2,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the simulation engine.

        Args:
            num_agents: Number of agents in the simulation
            num_candidates: Number of candidate options
            num_belief_dimensions: Dimensions in belief space
            seed: Random seed for reproducibility
        """
        self.num_agents: int = num_agents
        self.num_candidates: int = num_candidates
        self.num_belief_dimensions: int = num_belief_dimensions

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents: List[Agent] = []
        self.influence_graph: Optional[InfluenceGraph] = None
        # Generate random candidates (as IDs)
        self.candidates: List[Any] = list(range(num_candidates))
        self._round_num: int = 0

    def generate_random_agents(
        self,
        num_agents: Optional[int] = None,
        num_belief_dimensions: Optional[int] = None
    ) -> List[Agent]:
        """
        Generate a set of random agents with varied beliefs.

        Args:
            num_agents: Number of agents to generate (uses instance default if None)
            num_belief_dimensions: Belief dimensions (uses instance default if None)

        Returns:
            List of generated Agent instances
        """
        count = num_agents or self.num_agents
        dimensions = num_belief_dimensions or self.num_belief_dimensions

        self.agents = []
        for i in range(count):
            # Generate random belief vector (values between 0 and 1)
            belief = [random.random() for _ in range(dimensions)]

            # Generate influence tolerance (some agents are more easily influenced)
            # random.beta is not in standard library, using random.random instead
            influence_tolerance = random.random()  # type: float

            # Generate opinion strength
            opinion_strength = random.random()

            agent = Agent(
                agent_id=i,
                belief=belief,
                influence_tolerance=influence_tolerance,
                opinion_strength=opinion_strength,
                voting_history=[]
            )
            self.agents.append(agent)

        return self.agents

    def generate_random_influence_graph(
        self,
        agents: Optional[List[Agent]] = None,
        density: float = 0.3,
        weight_range: Tuple[float, float] = (0.1, 0.9)
    ) -> InfluenceGraph:
        """
        Generate a random influence graph with specified density.

        Args:
            agents: List of agents (uses instance agents if None)
            density: Edge density (0 to 1, fraction of possible edges)
            weight_range: Range of influence weights (min, max)

        Returns:
            Generated InfluenceGraph instance
        """
        agents = agents or self.agents
        if not agents:
            raise ValueError("No agents available. Call generate_random_agents first.")

        graph = InfluenceGraph()
        for agent in agents:
            graph.add_agent(agent)

        # Calculate number of edges based on density
        n = len(agents)
        max_edges = n * (n - 1)  # Directed graph, no self-loops
        num_edges = int(max_edges * density)

        # Generate random edges
        existing_edges = set()
        attempts = 0
        max_attempts = num_edges * 10

        while len(existing_edges) < num_edges and attempts < max_attempts:
            source = random.choice(agents)
            target = random.choice(agents)
            attempts += 1

            if source.agent_id != target.agent_id:
                edge_key = (source.agent_id, target.agent_id)
                if edge_key not in existing_edges:
                    weight = random.uniform(*weight_range)
                    graph.add_edge(source, target, weight)
                    existing_edges.add(edge_key)

        self.influence_graph = graph
        return graph

    def run_simulation(
        self,
        num_rounds: int,
        agents: Optional[List[Agent]] = None,
        influence_graph: Optional[InfluenceGraph] = None,
        voting_rule: str = "approval",
        influence_strength: float = 0.3,
        belief_threshold: float = 0.5,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Run a full simulation with multiple voting rounds.

        Args:
            num_rounds: Number of voting rounds to simulate
            agents: List of agents (generates if None)
            influence_graph: Pre-computed influence graph (generates if None)
            voting_rule: Voting rule to use ('approval', 'plurality', 'weighted_voting')
            influence_strength: Strength of social influence between rounds
            belief_threshold: Threshold for agent to participate in voting
            verbose: Whether to print progress

        Returns:
            SimulationResult with all round data and statistics
        """
        agents = agents or self.generate_random_agents()
        influence_graph = influence_graph or self.generate_random_influence_graph(
            agents
        )

        voting_system = VotingSystem(agents, self.candidates)

        result = SimulationResult(
            num_agents=len(agents),
            num_rounds=num_rounds,
            num_candidates=self.num_candidates
        )

        for round_num in range(num_rounds):
            round_result = self._run_single_round(
                round_num=round_num,
                agents=agents,
                influence_graph=influence_graph,
                voting_system=voting_system,
                voting_rule=voting_rule,
                influence_strength=influence_strength,
                belief_threshold=belief_threshold,
                verbose=verbose
            )
            result.rounds.append(round_result)

        # Compute summary statistics
        abstention_rates = [r.abstention_rate for r in result.rounds]
        result.final_abstention_rate = abstention_rates[-1] if abstention_rates else 0
        result.avg_abstention_rate = sum(abstention_rates) / len(abstention_rates) if abstention_rates else 0

        self._compute_consensus_score(result)

        return result

    def _run_single_round(
        self,
        round_num: int,
        agents: List[Agent],
        influence_graph: InfluenceGraph,
        voting_system: VotingSystem,
        voting_rule: str,
        influence_strength: float,
        belief_threshold: float,
        verbose: bool
    ) -> SimulationRound:
        """
        Run a single voting round with influence propagation.

        Args:
            round_num: Current round number
            agents: List of agents
            influence_graph: Influence relationship graph
            voting_system: Voting system instance
            voting_rule: Voting rule to apply
            influence_strength: Strength of between-round influence
            belief_threshold: Threshold for voting participation
            verbose: Print round information

        Returns:
            SimulationRound with round results
        """
        # Update belief using influence
        self._propagate_influence(agents, influence_graph, influence_strength)

        # Run voting
        vote_result = voting_system.run_vote(
            agents=agents,
            candidates=self.candidates,
            rule=voting_rule,
            influence_graph=influence_graph,
            belief_threshold=belief_threshold
        )

        # Record round result
        round_result = SimulationRound(
            round_num=round_num,
            abstention_rate=vote_result.abstention_rate,
            winner=vote_result.winner,
            vote_distribution=vote_result.vote_counts,
            belief_mean=self._compute_belief_mean(agents),
            belief_variance=self._compute_belief_variance(agents)
        )

        if verbose:
            print(
                f"Round {round_num}: Abstention={vote_result.abstention_rate:.2%}, "
                f"Winner={vote_result.winner}"
            )

        return round_result

    def _propagate_influence(
        self,
        agents: List[Agent],
        influence_graph: InfluenceGraph,
        strength: float
    ) -> None:
        """Propagate influence between agent beliefs."""
        for agent in agents:
            sources = influence_graph.get_influence_sources(agent)
            if sources:
                # Aggregate influence from all sources
                avg_influence = [0.0] * len(agent.belief)
                total_weight = 0

                for influencer, weight in sources:
                    for i, val in enumerate(influencer.belief):
                        avg_influence[i] += val * weight
                    total_weight += weight

                if total_weight > 0:
                    for i in range(len(avg_influence)):
                        avg_influence[i] /= total_weight

                    agent.update_belief(avg_influence, strength * agent.influence_tolerance)

    def _compute_belief_mean(self, agents: List[Agent]) -> List[float]:
        """Compute mean belief across all agents."""
        if not agents:
            return []

        dimensions = len(agents[0].belief)
        means = [0.0] * dimensions

        for agent in agents:
            for i, val in enumerate(agent.belief):
                means[i] += val

        return [m / len(agents) for m in means]

    def _compute_belief_variance(self, agents: List[Agent]) -> List[float]:
        """Compute variance of belief across all agents."""
        if not agents:
            return []

        dimensions = len(agents[0].belief)
        means = self._compute_belief_mean(agents)

        variances = [0.0] * dimensions
        for agent in agents:
            for i, val in enumerate(agent.belief):
                variances[i] += (val - means[i]) ** 2

        return [v / len(agents) for v in variances]

    def _compute_consensus_score(self, result: SimulationResult) -> None:
        """Compute consensus score based on agreement patterns."""
        if not result.rounds:
            result.consensus_score = 0
            return

        # Higher consensus if same winner multiple rounds
        winners = [r.winner for r in result.rounds]
        winner_counts = {}
        for w in winners:
            winner_counts[w] = winner_counts.get(w, 0) + 1

        # Consensus score based on winner consistency
        majority_rounds = max(winner_counts.values())
        result.consensus_score = majority_rounds / len(result.rounds)

    def run_batch_simulation(
        self,
        num_simulations: int,
        sim_params: Dict[str, Any],
        verbose: bool = False
    ) -> List[SimulationResult]:
        """
        Run multiple independent simulations.

        Args:
            num_simulations: Number of independent simulations
            sim_params: Parameters for each simulation
            verbose: Print progress

        Returns:
            List of SimulationResult instances
        """
        results = []

        for i in range(num_simulations):
            if verbose:
                print(f"Simulation {i + 1}/{num_simulations}")

            # Create fresh engine for each simulation
            engine = SimulationEngine(
                num_agents=self.num_agents,
                num_candidates=self.num_candidates,
                num_belief_dimensions=self.num_belief_dimensions,
                seed=None  # Different seed for each
            )

            result = engine.run_simulation(**sim_params)
            results.append(result)

        return results

    def get_abstention_trend(self, result: SimulationResult) -> List[float]:
        """Get abstention rate trend across rounds."""
        return [r.abstention_rate for r in result.rounds]

    def reset(self) -> None:
        """Reset the simulation engine state."""
        self.agents = []
        self.influence_graph = None
        self._round_num = 0
