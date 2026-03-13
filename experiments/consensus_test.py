"""Experiment runner for consensus testing."""

from typing import List, Dict, Any, Optional
import random
import json
from pathlib import Path

from src.agent import Agent
from src.influence import InfluenceGraph
from src.voting import VotingSystem
from src.simulation import SimulationEngine
from src.pattern_detector import PatternDetector


def create_test_agents(
    num_agents: int,
    num_candidates: int,
    consensus_strength: float = 0.7
) -> List[Agent]:
    """
    Create agents with controlled consensus strength.

    Args:
        num_agents: Number of agents to create
        num_candidates: Number of candidate options
        consensus_strength: How strongly agents align (0 to 1)

    Returns:
        List of Agent instances with controlled belief distribution
    """
    agents = []

    # Base belief for consensus (shared among most agents)
    base_belief = [random.random() for _ in range(2)]

    for i in range(num_agents):
        # With high probability, agents have similar beliefs
        if random.random() < consensus_strength:
            # Align with consensus
            belief = [
                base_belief[d] + random.gauss(0, 0.1)
                for d in range(2)
            ]
        else:
            # Deviate from consensus
            belief = [random.random() for _ in range(2)]

        # Normalize beliefs to [0, 1]
        belief = [max(0, min(1, b)) for b in belief]

        agent = Agent(
            agent_id=i,
            belief=belief,
            influence_tolerance=random.uniform(0.1, 0.5),
            opinion_strength=random.uniform(0.5, 1.0),
            voting_history=[]
        )
        agents.append(agent)

    return agents


def create_influence_graph(
    agents: List[Agent],
    consensus_strength: float
) -> InfluenceGraph:
    """
    Create influence graph that promotes consensus.

    Args:
        agents: List of agents
        consensus_strength: Strength of consensus-driven connections

    Returns:
        InfluenceGraph with consensus-promoting edges
    """
    graph = InfluenceGraph()
    for agent in agents:
        graph.add_agent(agent)

    # Create strong connections between agents with similar beliefs
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i >= j:
                continue

            # Calculate belief similarity
            similarity = sum(
                1 - abs(agent1.belief[d] - agent2.belief[d])
                for d in range(len(agent1.belief))
            ) / len(agent1.belief)

            # Higher weight if beliefs are similar (consensus effect)
            weight = 0.3 + 0.7 * similarity * consensus_strength
            graph.add_edge(agent1, agent2, weight)

    return graph


def run_consensus_experiment(
    num_agents: int = 100,
    num_candidates: int = 3,
    num_rounds: int = 50,
    consensus_strength_range: tuple = (0.5, 0.9),
    num_iterations: int = 100,
    verbose: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run consensus detection experiment.

    Tests how well the pattern detector identifies consensus
    under varying conditions.

    Args:
        num_agents: Number of agents per simulation
        num_candidates: Number of candidate options
        num_rounds: Number of voting rounds per simulation
        consensus_strength_range: Range of consensus strengths to test
        num_iterations: Number of simulations to run
        verbose: Print progress
        output_dir: Directory to save results

    Returns:
        Dictionary with experimental results
    """
    detector = PatternDetector()
    results = {
        "parameters": {
            "num_agents": num_agents,
            "num_candidates": num_candidates,
            "num_rounds": num_rounds,
            "consensus_strength_range": consensus_strength_range,
            "num_iterations": num_iterations
        },
        "iterations": []
    }

    for iteration in range(num_iterations):
        # Sample consensus strength
        consensus_strength = random.uniform(*consensus_strength_range)

        # Create agents and influence graph
        agents = create_test_agents(num_agents, num_candidates, consensus_strength)
        influence_graph = create_influence_graph(agents, consensus_strength)

        # Run simulation
        engine = SimulationEngine(
            num_agents=num_agents,
            num_candidates=num_candidates,
            num_belief_dimensions=2,
            seed=iteration
        )
        engine.candidates = [f"Candidate_{i}" for i in range(num_candidates)]

        sim_result = engine.run_simulation(
            num_rounds=num_rounds,
            agents=agents,
            influence_graph=influence_graph,
            voting_rule="approval",
            influence_strength=0.3
        )

        # Analyze with pattern detector
        voting_data = {
            "rounds": [
                {
                    "abstention_rate": r.abstention_rate,
                    "winner": r.winner,
                    "vote_distribution": r.vote_distribution,
                    "belief_variance": r.belief_variance
                }
                for r in sim_result.rounds
            ],
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "belief": a.belief,
                    "belief_deviation": 0,
                    "abstention": a.voting_history[-1].get("abstention", False) if a.voting_history else False
                }
                for a in agents
            ]
        }

        pattern_result = detector.detect_consensus(voting_data)

        # Store iteration results
        iteration_result = {
            "iteration": iteration,
            "consensus_strength": consensus_strength,
            "sim_abstention_rate": sim_result.avg_abstention_rate,
            "sim_winner_consistency": sim_result.consensus_score,
            "detected_consensus_score": pattern_result.score,
            "detected_consensus_confidence": pattern_result.confidence
        }
        results["iterations"].append(iteration_result)

        if verbose:
            print(
                f"Iteration {iteration + 1}/{num_iterations}: "
                f"Actual consensus={consensus_strength:.2f}, "
                f"Detected={pattern_result.score:.2f}"
            )

    # Calculate summary statistics
    actual_scores = [r["consensus_strength"] for r in results["iterations"]]
    detected_scores = [r["detected_consensus_score"] for r in results["iterations"]]

    # Calculate correlation
    correlation = calculate_correlation(actual_scores, detected_scores)

    results["summary"] = {
        "num_iterations": num_iterations,
        "mean_actual_consensus": sum(actual_scores) / len(actual_scores),
        "mean_detected_consensus": sum(detected_scores) / len(detected_scores),
        "correlation": correlation,
        "mse": calculate_mse(actual_scores, detected_scores)
    }

    # Save results if output directory specified
    if output_dir:
        save_results(output_dir, results, "consensus")

    return results


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation between two lists."""
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    denom_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

    if denom_x * denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def calculate_mse(actual: List[float], predicted: List[float]) -> float:
    """Calculate mean squared error."""
    n = len(actual)
    if n != len(predicted) or n == 0:
        return 0.0

    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n


def save_results(output_dir: str, results: Dict[str, Any], experiment_name: str) -> None:
    """Save experiment results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_path / f"{experiment_name}_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary_file = output_path / f"{experiment_name}_summary.txt"
    with open(summary_file, "w") as f:
        summary = results.get("summary", {})
        f.write(f"{experiment_name} Experiment Summary\n")
        f.write("=" * 40 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def main():
    """Run consensus experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run consensus detection experiment")
    parser.add_argument("--num-agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--num-candidates", type=int, default=3, help="Number of candidates")
    parser.add_argument("--num-rounds", type=int, default=50, help="Number of voting rounds")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")

    args = parser.parse_args()

    results = run_consensus_experiment(
        num_agents=args.num_agents,
        num_candidates=args.num_candidates,
        num_rounds=args.num_rounds,
        num_iterations=args.iterations,
        verbose=args.verbose,
        output_dir=args.output_dir
    )

    print(f"\nExperiment complete. Results saved to {args.output_dir}/")
    print(f"Correlation: {results['summary']['correlation']:.4f}")
    print(f"MSE: {results['summary']['mse']:.4f}")


if __name__ == "__main__":
    main()
