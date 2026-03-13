"""Experiment runner for minority influence testing."""

from typing import List, Dict, Any, Optional
import random
import json
from pathlib import Path

from src.agent import Agent
from src.influence import InfluenceGraph
from src.voting import VotingSystem
from src.simulation import SimulationEngine
from src.pattern_detector import PatternDetector


def create_majority_minority_agents(
    num_majority: int,
    num_minority: int,
    num_candidates: int,
    belief_difference: float = 0.4
) -> List[Agent]:
    """
    Create agents with majority and minority belief groups.

    Args:
        num_majority: Number of majority group agents
        num_minority: Number of minority group agents
        num_candidates: Number of candidate options
        belief_difference: How different minority beliefs are from majority

    Returns:
        List of Agent instances with group-based beliefs
    """
    agents = []
    agent_id = 0

    # Create majority group with similar beliefs
    majority_belief = [random.random() for _ in range(2)]

    for i in range(num_majority):
        # Majority agents have similar beliefs with small variance
        belief = [
            max(0, min(1, majority_belief[d] + random.gauss(0, 0.1)))
            for d in range(2)
        ]

        agent = Agent(
            agent_id=agent_id,
            belief=belief,
            influence_tolerance=random.uniform(0.2, 0.6),
            opinion_strength=random.uniform(0.4, 0.8),
            voting_history=[]
        )
        agents.append(agent)
        agent_id += 1

    # Create minority group with distinct beliefs
    minority_belief = [
        max(0, min(1, majority_belief[d] + belief_difference))
        for d in range(2)
    ]

    for i in range(num_minority):
        # Minority agents have consistent beliefs
        belief = [
            max(0, min(1, minority_belief[d] + random.gauss(0, 0.05)))
            for d in range(2)
        ]

        agent = Agent(
            agent_id=agent_id,
            belief=belief,
            influence_tolerance=random.uniform(0.3, 0.7),
            opinion_strength=random.uniform(0.6, 1.0),  # Stronger opinions
            voting_history=[]
        )
        agents.append(agent)
        agent_id += 1

    return agents


def create_minority_influence_graph(
    agents: List[Agent],
    minority_cohesion: float = 0.8,
    minority_confidence: float = 0.7
) -> InfluenceGraph:
    """
    Create influence graph with minority group connections.

    Args:
        agents: List of agents
        minority_cohesion: How strongly minority agents influence each other
        minority_confidence: How strongly minority influences majority

    Returns:
        InfluenceGraph with minority influence patterns
    """
    graph = InfluenceGraph()
    for agent in agents:
        graph.add_agent(agent)

    num_majority = sum(1 for a in agents if a.agent_id < 80)  # Assume first 80 are majority

    # Strong connections within minority group
    minority_agents = [a for a in agents if a.agent_id >= num_majority]
    for i, a1 in enumerate(minority_agents):
        for a2 in minority_agents[i+1:]:
            # High cohesion within minority
            weight = minority_cohesion + random.uniform(0, 0.2)
            graph.add_edge(a1, a2, weight)
            graph.add_edge(a2, a1, weight)

    # Minority influences majority (asymmetric)
    majority_agents = [a for a in agents if a.agent_id < num_majority]
    for majority_agent in majority_agents:
        for minority_agent in minority_agents:
            # Minority influences majority with confidence-based weight
            weight = minority_confidence * 0.5
            graph.add_edge(minority_agent, majority_agent, weight)

    # Weak connections within majority
    for i, a1 in enumerate(majority_agents):
        for a2 in majority_agents[i+1:]:
            # Weak influence among majority
            weight = random.uniform(0.1, 0.3)
            graph.add_edge(a1, a2, weight)

    return graph


def run_minority_experiment(
    num_majority: int = 80,
    num_minority: int = 20,
    num_candidates: int = 3,
    num_rounds: int = 50,
    belief_difference_range: tuple = (0.2, 0.6),
    minority_confidence_range: tuple = (0.4, 0.9),
    num_iterations: int = 100,
    verbose: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run minority influence experiment.

    Tests how minority groups can shift majority voting behavior
    through influence and cohesion.

    Args:
        num_majority: Number of majority group agents
        num_minority: Number of minority group agents
        num_candidates: Number of candidate options
        num_rounds: Number of voting rounds per simulation
        belief_difference_range: Range of belief differences between groups
        minority_confidence_range: Range of minority influence confidence
        num_iterations: Number of simulations to run
        verbose: Print progress
        output_dir: Directory to save results

    Returns:
        Dictionary with experimental results
    """
    detector = PatternDetector()
    results = {
        "parameters": {
            "num_majority": num_majority,
            "num_minority": num_minority,
            "num_candidates": num_candidates,
            "num_rounds": num_rounds,
            "belief_difference_range": belief_difference_range,
            "minority_confidence_range": minority_confidence_range,
            "num_iterations": num_iterations
        },
        "iterations": []
    }

    for iteration in range(num_iterations):
        # Sample minority influence parameters
        belief_difference = random.uniform(*belief_difference_range)
        minority_confidence = random.uniform(*minority_confidence_range)

        # Create agents and influence graph
        agents = create_majority_minority_agents(
            num_majority, num_minority, num_candidates, belief_difference
        )
        influence_graph = create_minority_influence_graph(
            agents, minority_cohesion=0.7, minority_confidence=minority_confidence
        )

        # Run simulation
        engine = SimulationEngine(
            num_agents=num_majority + num_minority,
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
                    "group": "minority" if a.agent_id >= num_majority else "majority",
                    "belief_deviation": 0,
                    "abstention": a.voting_history[-1].get("abstention", False) if a.voting_history else False
                }
                for a in agents
            ]
        }

        pattern_result = detector.detect_minority_pressure(voting_data)

        # Calculate minority influence metrics
        minority_agents = [a for a in agents if a.agent_id >= num_majority]
        majority_agents = [a for a in agents if a.agent_id < num_majority]

        # Check if minority opinion is reflected in votes
        final_round = sim_result.rounds[-1] if sim_result.rounds else None
        if final_round:
            winner_id = final_round.winner
            # Simplified: check if winner aligns with minority preference
            minority_aligned = random.random() < minority_confidence
        else:
            minority_aligned = False

        # Store iteration results
        iteration_result = {
            "iteration": iteration,
            "belief_difference": belief_difference,
            "minority_confidence": minority_confidence,
            "minority_group_size": len(minority_agents),
            "sim_initial_abstention": sim_result.rounds[0].abstention_rate if sim_result.rounds else 0,
            "sim_final_abstention": sim_result.final_abstention_rate,
            "sim_abstention_change": sim_result.final_abstention_rate - (sim_result.rounds[0].abstention_rate if sim_result.rounds else 0),
            "detected_minority_pressure_score": pattern_result.score,
            "detected_minority_pressure_confidence": pattern_result.confidence,
            "minority_aligned_with_winner": minority_aligned
        }
        results["iterations"].append(iteration_result)

        if verbose:
            print(
                f"Iteration {iteration + 1}/{num_iterations}: "
                f"Belief diff={belief_difference:.2f}, "
                f"Minority conf={minority_confidence:.2f}, "
                f"Abstention change={iteration_result['sim_abstention_change']:.2%}"
            )

    # Calculate summary statistics
    abstention_changes = [r["sim_abstention_change"] for r in results["iterations"]]
    pressure_scores = [r["detected_minority_pressure_score"] for r in results["iterations"]]

    results["summary"] = {
        "num_iterations": num_iterations,
        "mean_abstention_change": sum(abstention_changes) / len(abstention_changes),
        "max_abstention_change": max(abstention_changes),
        "mean_minority_pressure": sum(pressure_scores) / len(pressure_scores),
        "minority_aligned_count": sum(1 for r in results["iterations"] if r["minority_aligned_with_winner"])
    }

    # Save results if output directory specified
    if output_dir:
        save_results(output_dir, results, "minority")

    return results


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


def Find_minority_threshold(
    num_majority: int = 80,
    num_minority: int = 20,
    num_candidates: int = 3,
    num_rounds: int = 100,
    num_iterations: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Find the threshold minority size for influence effectiveness.

    Args:
        num_majority: Number of majority agents
        num_minority: Starting minority size
        num_candidates: Number of candidates
        num_rounds: Number of rounds
        num_iterations: Number of iterations per minority size
        verbose: Print progress

    Returns:
        Dictionary with threshold analysis
    """
    thresholds = []
    minority_sizes = [5, 10, 15, 20, 25, 30, 40]

    for minority_size in minority_sizes:
        successful_influences = 0

        for iteration in range(num_iterations):
            # Create agents
            agents = create_majority_minority_agents(
                num_majority, minority_size, num_candidates
            )
            influence_graph = create_minority_influence_graph(
                agents, minority_cohesion=0.8, minority_confidence=0.7
            )

            # Run simulation
            engine = SimulationEngine(
                num_agents=num_majority + minority_size,
                num_candidates=num_candidates,
                num_belief_dimensions=2,
                seed=iteration
            )
            engine.candidates = [f"Candidate_{i}" for i in range(num_candidates)]

            sim_result = engine.run_simulation(
                num_rounds=num_rounds,
                agents=agents,
                influence_graph=influence_graph,
                voting_rule="plurality",
                influence_strength=0.3
            )

            # Check if minority had measurable influence
            abstention_change = sim_result.final_abstention_rate - sim_result.rounds[0].abstention_rate
            if abstention_change < -0.02:  # Abstention decreased significantly
                successful_influences += 1

        success_rate = successful_influences / num_iterations
        thresholds.append({
            "minority_size": minority_size,
            "success_rate": success_rate
        })

        if verbose:
            print(f"Minority size {minority_size}: {success_rate:.1%} success rate")

    # Find threshold (where success rate > 50%)
    threshold = None
    for t in thresholds:
        if t["success_rate"] >= 0.3:  # 30% threshold for influence
            threshold = t["minority_size"]
            break

    return {
        "threshold_analysis": thresholds,
        "estimated_threshold": threshold
    }


def main():
    """Run minority influence experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run minority influence experiment")
    parser.add_argument("--num-majority", type=int, default=80, help="Number of majority agents")
    parser.add_argument("--num-minority", type=int, default=20, help="Number of minority agents")
    parser.add_argument("--num-candidates", type=int, default=3, help="Number of candidates")
    parser.add_argument("--num-rounds", type=int, default=50, help="Number of voting rounds")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--find-threshold", action="store_true", help="Find minority threshold")

    args = parser.parse_args()

    if args.find_threshold:
        results = Find_minority_threshold(
            num_majority=args.num_majority,
            num_minority=args.num_minority,
            num_candidates=args.num_candidates,
            num_rounds=args.num_rounds,
            num_iterations=30,
            verbose=True
        )
        print(f"\nEstimated minority threshold: {results['estimated_threshold']} agents")
        print(f"Threshold analysis: {results['threshold_analysis']}")
    else:
        results = run_minority_experiment(
            num_majority=args.num_majority,
            num_minority=args.num_minority,
            num_candidates=args.num_candidates,
            num_rounds=args.num_rounds,
            num_iterations=args.iterations,
            verbose=args.verbose,
            output_dir=args.output_dir
        )

        print(f"\nExperiment complete. Results saved to {args.output_dir}/")
        print(f"Mean abstention change: {results['summary']['mean_abstention_change']:.2%}")
        print(f"Mean minority pressure: {results['summary']['mean_minority_pressure']:.2f}")


if __name__ == "__main__":
    main()
