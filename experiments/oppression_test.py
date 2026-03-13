"""Experiment runner for oppression testing."""

from typing import List, Dict, Any, Optional
import random
import json
from pathlib import Path

from src.agent import Agent
from src.influence import InfluenceGraph
from src.voting import VotingSystem
from src.simulation import SimulationEngine
from src.pattern_detector import PatternDetector


def create_oppressed_agents(
    num_agents: int,
    num_candidates: int,
    dominant_agent_strength: float = 0.8,
    oppression_level: str = "high"  # "low", "medium", "high"
) -> List[Agent]:
    """
    Create agents with an oppressive dominant agent.

    Args:
        num_agents: Total number of agents (including dominant)
        num_candidates: Number of candidate options
        dominant_agent_strength: Strength of dominant agent's influence
        oppression_level: Level of oppression (low/medium/high)

    Returns:
        List of Agent instances with dominant agent
    """
    agents = []

    # Create dominant agent with very strong opinion
    dominant_belief = [random.random() for _ in range(2)]
    dominant_agent = Agent(
        agent_id=0,
        belief=dominant_belief,
        influence_tolerance=0.1,  # Very low tolerance (hard to influence)
        opinion_strength=1.0,  # Maximum opinion strength
        voting_history=[]
    )
    agents.append(dominant_agent)

    # Create oppressed agents with varying compliance levels
    oppression_factors = {
        "low": {"compliance": 0.2, "fear": 0.1},
        "medium": {"compliance": 0.5, "fear": 0.3},
        "high": {"compliance": 0.8, "fear": 0.6}
    }

    factors = oppression_factors.get(oppresion_level, oppression_factors["medium"])

    for i in range(1, num_agents):
        # Most agents have diverse beliefs (not aligned with dominant)
        belief = [random.random() for _ in range(2)]

        # Some agents comply with dominant (strong influence tolerance)
        # Others resist (low influence tolerance)
        if random.random() < factors["compliance"]:
            # Compliant agent - easily influenced by dominant
            influence_tolerance = random.uniform(0.5, 0.9)
            opinion_strength = random.uniform(0.1, 0.4)  # Weak opinions
        else:
            # Resistant agent - resists influence
            influence_tolerance = random.uniform(0.1, 0.3)
            opinion_strength = random.uniform(0.5, 0.9)  # Strong opinions

        agent = Agent(
            agent_id=i,
            belief=belief,
            influence_tolerance=influence_tolerance,
            opinion_strength=opinion_strength,
            voting_history=[]
        )
        agents.append(agent)

    return agents


def create_oppression_graph(
    agents: List[Agent],
    oppression_level: str = "high"
) -> InfluenceGraph:
    """
    Create influence graph with dominant agent压迫 structure.

    Args:
        agents: List of agents
        oppression_level: Level of oppression

    Returns:
        InfluenceGraph with oppressive structure
    """
    graph = InfluenceGraph()
    for agent in agents:
        graph.add_agent(agent)

    dominant_agent = agents[0]

    # Create asymmetric influence: everyone → dominant (deference)
    # dominant → everyone (oppression)
    oppression_weights = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8
    }
    weight = oppression_weights.get(oppresion_level, oppression_weights["high"])

    for agent in agents[1:]:
        # Dominant influences all others (oppression)
        graph.add_edge(dominant_agent, agent, weight)

        # Others have weak influence on dominant (deference)
        graph.add_edge(agent, dominant_agent, 0.1)

        # Weak influence among ordinary agents
        for other_agent in agents[1:]:
            if agent.agent_id < other_agent.agent_id:
                graph.add_edge(agent, other_agent, random.uniform(0.1, 0.3))

    return graph


def run_oppression_experiment(
    num_agents: int = 100,
    num_candidates: int = 3,
    num_rounds: int = 50,
    num_oppresion_levels: int = 3,
    num_iterations: int = 50,
    verbose: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run oppression detection experiment.

    Simulates oppressive group dynamics and measures effect on abstention.

    Args:
        num_agents: Number of agents per simulation
        num_candidates: Number of candidate options
        num_rounds: Number of voting rounds per simulation
        num_oppresion_levels: Number of different oppression levels to test
        num_iterations: Number of simulations per oppression level
        verbose: Print progress
        output_dir: Directory to save results

    Returns:
        Dictionary with experimental results
    """
    detector = PatternDetector()
    oppression_levels = ["low", "medium", "high"]
    results = {
        "parameters": {
            "num_agents": num_agents,
            "num_candidates": num_candidates,
            "num_rounds": num_rounds,
            "num_oppresion_levels": num_oppresion_levels,
            "num_iterations_per_level": num_iterations,
            "oppresion_levels": oppression_levels
        },
        "oppresion_levels": {}
    }

    all_opp_data = []  # Collect all data for cross-level analysis

    for opp_level in oppression_levels:
        if verbose:
            print(f"\nTesting oppression level: {opp_level}")

        level_results = {
            "oppresion_level": opp_level,
            "iterations": [],
            "abstention_correlations": []
        }

        for iteration in range(num_iterations):
            # Create agents
            agents = create_oppressed_agents(
                num_agents, num_candidates, dominant_agent_strength=0.9,
                oppression_level=opp_level
            )

            # Create oppression graph
            influence_graph = create_oppression_graph(agents, opp_level)

            # Run simulation
            engine = SimulationEngine(
                num_agents=num_agents,
                num_candidates=num_candidates,
                num_belief_dimensions=2,
                seed=iteration * 100  # Unique seed per level+iteration
            )
            engine.candidates = [f"Candidate_{i}" for i in range(num_candidates)]

            sim_result = engine.run_simulation(
                num_rounds=num_rounds,
                agents=agents,
                influence_graph=influence_graph,
                voting_rule="plurality",
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
                        "is_dominant": a.agent_id == 0,
                        "opinion_strength": a.opinion_strength,
                        "influence_tolerance": a.influence_tolerance,
                        "abstention": a.voting_history[-1].get("abstention", False) if a.voting_history else False
                    }
                    for a in agents
                ]
            }

            pattern_result = detector.detect_oppression(voting_data)

            # Calculate belief-abstention correlation (key oppression indicator)
            non_dominant_agents = [a for a in agents if a.agent_id != 0]
            belief_variances = [
                sum((b - 0.5) ** 2 for b in a.belief) / len(a.belief)
                for a in non_dominant_agents
            ]
            abstentions = [
                1 if a.voting_history and a.voting_history[-1].get("abstention") else 0
                for a in non_dominant_agents
            ]

            # Calculate correlation
            correlation = calculate_correlation(belief_variances, abstentions)

            # Store iteration results
            iteration_result = {
                "iteration": iteration,
                "sim_initial_abstention": sim_result.rounds[0].abstention_rate if sim_result.rounds else 0,
                "sim_final_abstention": sim_result.final_abstention_rate,
                "sim_abstention_change": sim_result.final_abstention_rate - (sim_result.rounds[0].abstention_rate if sim_result.rounds else 0),
                "sim_winners": [r.winner for r in sim_result.rounds[-5:]] if sim_result.rounds else [],
                "detected_oppression_score": pattern_result.score,
                "detected_oppression_confidence": pattern_result.confidence,
                "belief_abstention_correlation": correlation
            }
            level_results["iterations"].append(iteration_result)

            # Collect for cross-level analysis
            all_opp_data.append({
                "oppresion_level": opp_level,
                "abstention_change": iteration_result["sim_abstention_change"],
                "oppression_score": pattern_result.score,
                "correlation": correlation
            })

            if verbose and iteration % 10 == 0:
                print(
                    f"  Iteration {iteration}: "
                    f"Abstention change={iteration_result['sim_abstention_change']:.2%}, "
                    f"Oppression score={pattern_result.score:.2f}"
                )

        # Calculate level summary
        level_results["summary"] = {
            "num_iterations": num_iterations,
            "mean_abstention_change": sum(i["sim_abstention_change"] for i in level_results["iterations"]) / num_iterations,
            "mean_oppression_score": sum(i["detected_oppression_score"] for i in level_results["iterations"]) / num_iterations,
            "mean_correlation": sum(i["belief_abstention_correlation"] for i in level_results["iterations"]) / num_iterations,
            "max_abstention_change": max(i["sim_abstention_change"] for i in level_results["iterations"])
        }

        results["oppresion_levels"][opp_level] = level_results

    # Cross-level analysis
    results["cross_level_analysis"] = {
        "abstention_trend": [],  # Abstention change by oppression level
        "detection_correlation": calculate_correlation(
            [1 if d["oppresion_level"] == "high" else (0.5 if d["oppresion_level"] == "medium" else 0) for d in all_opp_data],
            [d["oppression_score"] for d in all_opp_data]
        ),
        "abstention_correlation": calculate_correlation(
            [1 if d["oppresion_level"] == "high" else (0.5 if d["oppresion_level"] == "medium" else 0) for d in all_opp_data],
            [d["correlation"] for d in all_opp_data]
        )
    }

    # Calculate trend across levels
    for opp_level in oppression_levels:
        level_data = results["oppresion_levels"][opp_level]["iterations"]
        if level_data:
            results["cross_level_analysis"]["abstention_trend"].append({
                "oppresion_level": opp_level,
                "mean_abstention_change": results["oppresion_levels"][opp_level]["summary"]["mean_abstention_change"]
            })

    # Save results if output directory specified
    if output_dir:
        save_results(output_dir, results, "oppression")

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
        f.write(f"{experiment_name} Experiment Summary\n")
        f.write("=" * 40 + "\n\n")

        cross_analysis = results.get("cross_level_analysis", {})
        f.write(f"Detection Correlation: {cross_analysis.get('detection_correlation', 0):.4f}\n")
        f.write(f"Abstention Correlation: {cross_analysis.get('abstention_correlation', 0):.4f}\n\n")

        f.write("By Oppression Level:\n")
        for level, level_data in results.get("oppresion_levels", {}).items():
            summary = level_data.get("summary", {})
            f.write(f"\n{level.title()} Oppression:\n")
            f.write(f"  Mean Abstention Change: {summary.get('mean_abstention_change', 0):.2%}\n")
            f.write(f"  Mean Oppression Score: {summary.get('mean_oppression_score', 0):.4f}\n")
            f.write(f"  Mean Correlation: {summary.get('mean_correlation', 0):.4f}\n")


def run_abstention_analysis(
    num_agents: int = 100,
    num_candidates: int = 3,
    num_rounds: int = 50,
    oppression_level: str = "high",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run detailed abstention analysis under oppression.

    Analyzes how different factors correlate with abstention.

    Args:
        num_agents: Number of agents
        num_candidates: Number of candidates
        num_rounds: Number of rounds
        oppression_level: Level of oppression
        verbose: Print progress

    Returns:
        Dictionary with detailed analysis
    """
    agents = create_oppressed_agents(
        num_agents, num_candidates, oppression_level=oppresion_level
    )
    influence_graph = create_oppression_graph(agents, oppression_level)

    engine = SimulationEngine(
        num_agents=num_agents,
        num_candidates=num_candidates,
        num_belief_dimensions=2
    )
    engine.candidates = [f"Candidate_{i}" for i in range(num_candidates)]

    sim_result = engine.run_simulation(
        num_rounds=num_rounds,
        agents=agents,
        influence_graph=influence_graph,
        voting_rule="plurality",
        influence_strength=0.3
    )

    # Analyze factors correlated with abstention
    non_dominant_agents = [a for a in agents if a.agent_id != 0]

    # Factor 1: Opinion strength vs abstention
    opinion_strengths = [a.opinion_strength for a in non_dominant_agents]
    abstentions = [
        1 if a.voting_history and a.voting_history[-1].get("abstention") else 0
        for a in non_dominant_agents
    ]
    opinion_corr = calculate_correlation(opinion_strengths, abstentions)

    # Factor 2: Influence tolerance vs abstention
    influence_tolerances = [a.influence_tolerance for a in non_dominant_agents]
    tolerance_corr = calculate_correlation(influence_tolerances, abstentions)

    # Factor 3: Belief deviation from dominant vs abstention
    dominant_belief = agents[0].belief
    belief_deviations = [
        sum(abs(a.belief[d] - dominant_belief[d]) for d in range(len(a.belief))) / len(a.belief)
        for a in non_dominant_agents
    ]
    deviation_corr = calculate_correlation(belief_deviations, abstentions)

    analysis = {
        "oppresion_level": oppresion_level,
        "num_agents": num_agents,
        "sim_final_abstention": sim_result.final_abstention_rate,
        "factors": {
            "opinion_strength_vs_abstention": {
                "correlation": opinion_corr,
                "interpretation": "Lower opinion strength → higher abstention" if opinion_corr < -0.1 else "Higher opinion strength → higher abstention" if opinion_corr > 0.1 else "No clear relationship"
            },
            "influence_tolerance_vs_abstention": {
                "correlation": tolerance_corr,
                "interpretation": "High tolerance → higher abstention" if tolerance_corr > 0.1 else "Low tolerance → higher abstention" if tolerance_corr < -0.1 else "No clear relationship"
            },
            "belief_deviation_vs_abstention": {
                "correlation": deviation_corr,
                "interpretation": "High deviation → higher abstention (oppression effect)" if deviation_corr > 0.1 else "No clear relationship"
            }
        },
        "abstention_by_round": [r.abstention_rate for r in sim_result.rounds]
    }

    if verbose:
        print(f"\nAbstention Analysis under {oppresion_level} oppression:")
        print(f"Final abstention rate: {sim_result.final_abstention_rate:.2%}")
        print("\nCorrelations with abstention:")
        for factor, data in analysis["factors"].items():
            print(f"  {factor.replace('_', ' ').title()}: {data['correlation']:.3f}")
            print(f"    → {data['interpretation']}")

    return analysis


def main():
    """Run oppression experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run oppression detection experiment")
    parser.add_argument("--num-agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--num-candidates", type=int, default=3, help="Number of candidates")
    parser.add_argument("--num-rounds", type=int, default=50, help="Number of voting rounds")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations per level")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--detailed", action="store_true", help="Run detailed abstention analysis")
    parser.add_argument("--level", default="high", choices=["low", "medium", "high"], help="Oppression level for detailed analysis")

    args = parser.parse_args()

    if args.detailed:
        results = run_abstention_analysis(
            num_agents=args.num_agents,
            num_candidates=args.num_candidates,
            num_rounds=args.num_rounds,
            oppression_level=args.level,
            verbose=True
        )
    else:
        results = run_oppression_experiment(
            num_agents=args.num_agents,
            num_candidates=args.num_candidates,
            num_rounds=args.num_rounds,
            num_iterations=args.iterations,
            verbose=args.verbose,
            output_dir=args.output_dir
        )

        print(f"\nExperiment complete. Results saved to {args.output_dir}/")

        # Print cross-level summary
        cross = results.get("cross_level_analysis", {})
        print(f"\nDetection Accuracy (correlation with true oppression level): {cross.get('detection_correlation', 0):.3f}")
        print(f"Abstention Correlation: {cross.get('abstention_correlation', 0):.3f}")

        print("\nOppression Level Analysis:")
        for level, level_data in results.get("oppresion_levels", {}).items():
            summary = level_data.get("summary", {})
            print(f"  {level.title()}: Abstention change = {summary.get('mean_abstention_change', 0):.2%}, "
                  f"Oppression score = {summary.get('mean_oppression_score', 0):.3f}")


if __name__ == "__main__":
    main()
