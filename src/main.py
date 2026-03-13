#!/usr/bin/env python3
"""Command-line interface for Silence Pattern Decoder."""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.influence import InfluenceGraph
from src.voting import VotingSystem
from src.simulation import SimulationEngine
from src.pattern_detector import PatternDetector
from src.analysis import compute_abstention_rate, load_voting_data
from src.visualizer import (
    plot_abstention_timeline,
    plot_belief_distribution,
    plot_consensus_vs_abstention,
    plot_pattern_scores
)


def run_simulation(args) -> dict:
    """Run a single simulation."""
    engine = SimulationEngine(
        num_agents=args.num_agents,
        num_candidates=args.num_candidates,
        num_belief_dimensions=args.belief_dimensions,
        seed=args.seed
    )

    # Generate agents and influence graph
    agents = engine.generate_random_agents()
    influence_graph = engine.generate_random_influence_graph(agents, density=args.influence_density)

    engine.candidates = [f"Candidate_{i}" for i in range(args.num_candidates)]

    result = engine.run_simulation(
        num_rounds=args.num_rounds,
        agents=agents,
        influence_graph=influence_graph,
        voting_rule=args.voting_rule,
        influence_strength=args.influence_strength,
        belief_threshold=args.belief_threshold,
        verbose=args.verbose
    )

    if args.verbose:
        print(f"\nSimulation complete!")
        print(f"  Final abstention rate: {result.final_abstention_rate:.2%}")
        print(f"  Average abstention rate: {result.avg_abstention_rate:.2%}")
        print(f"  Consensus score: {result.consensus_score:.2f}")

    # Save results if requested
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_dir / "simulation_result.json"
        with open(json_path, "w") as f:
            json.dump({
                "num_agents": result.num_agents,
                "num_rounds": result.num_rounds,
                "num_candidates": result.num_candidates,
                "final_abstention_rate": result.final_abstention_rate,
                "avg_abstention_rate": result.avg_abstention_rate,
                "consensus_score": result.consensus_score,
                "rounds": [
                    {
                        "round_num": r.round_num,
                        "abstention_rate": r.abstention_rate,
                        "winner": r.winner,
                        "vote_distribution": r.vote_distribution
                    }
                    for r in result.rounds
                ]
            }, f, indent=2)

        print(f"Results saved to {json_path}")

    return {"result": result}


def run_experiment(args) -> dict:
    """Run a full experiment."""
    from experiments.consensus_test import run_consensus_experiment
    from experiments.minority_test import run_minority_experiment
    from experiments.oppression_test import run_oppression_experiment

    experiment_dir = Path("experiments/results")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment_type == "consensus":
        results = run_consensus_experiment(
            num_agents=args.num_agents,
            num_candidates=args.num_candidates,
            num_rounds=args.num_rounds,
            num_iterations=args.iterations,
            verbose=args.verbose,
            output_dir=str(experiment_dir / "consensus")
        )
    elif args.experiment_type == "minority":
        results = run_minority_experiment(
            num_majority=args.num_majority,
            num_minority=args.num_minority,
            num_candidates=args.num_candidates,
            num_rounds=args.num_rounds,
            num_iterations=args.iterations,
            verbose=args.verbose,
            output_dir=str(experiment_dir / "minority")
        )
    elif args.experiment_type == "oppression":
        results = run_oppression_experiment(
            num_agents=args.num_agents,
            num_candidates=args.num_candidates,
            num_rounds=args.num_rounds,
            num_iterations=args.iterations,
            verbose=args.verbose,
            output_dir=str(experiment_dir / "oppression")
        )
    else:
        print(f"Unknown experiment type: {args.experiment_type}")
        return {"error": "Unknown experiment type"}

    print(f"\nExperiment complete! Results saved to experiments/results/{args.experiment_type}/")
    return {"result": results}


def analyze_data(args) -> dict:
    """Analyze saved voting data."""
    if not args.input:
        print("No input file specified. Use --input to specify a data file.")
        return {"error": "No input file"}

    # Load data
    try:
        data = load_voting_data(args.input)
    except FileNotFoundError:
        print(f"File not found: {args.input}")
        return {"error": "File not found"}

    # Compute statistics
    votes = data.get("votes", data.get("rounds", []))
    abstention_rate = compute_abstention_rate(votes)

    print(f"\nAnalysis results:")
    print(f"  Total votes: {len(votes)}")
    print(f"  Abstention rate: {abstention_rate:.2%}")

    # Pattern detection
    detector = PatternDetector()

    # For round-based data
    if "rounds" in data:
        voting_data = {
            "rounds": data["rounds"],
            "agents": data.get("agents", [])
        }
    else:
        # Construct voting data from flat votes
        voting_data = {
            "rounds": [{"abstention_rate": abstention_rate}],
            "agents": []
        }

    results = detector.analyze_all_patterns(voting_data)

    print(f"\nPattern Detection:")
    for pattern_name, result in results.items():
        print(f"  {pattern_name}:")
        print(f"    Score: {result.score:.3f}")
        print(f"    Confidence: {result.confidence:.3f}")

    # Generate visualization
    if args.visualize and "rounds" in data:
        abstention_rates = [r.get("abstention_rate", 0) for r in data["rounds"]]
        if abstention_rates:
            plot_path = Path(args.output) / "abstention_timeline.png" if args.output else "abstention_timeline.png"
            plot_abstention_timeline(abstention_rates, save_path=str(plot_path))
            print(f"\nVisualization saved to {plot_path}")

    return {"result": {"abstention_rate": abstention_rate, "patterns": results}}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Silence Pattern Decoder - Analyze voting behavior and abstention patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run-sim subcommand
    sim_parser = subparsers.add_parser("run-sim", help="Run a single simulation")
    sim_parser.add_argument("--num-agents", "-n", type=int, default=100, help="Number of agents")
    sim_parser.add_argument("--num-candidates", "-c", type=int, default=3, help="Number of candidates")
    sim_parser.add_argument("--num-rounds", "-r", type=int, default=50, help="Number of voting rounds")
    sim_parser.add_argument("--belief-dimensions", "-d", type=int, default=2, help="Belief space dimensions")
    sim_parser.add_argument("--influence-density", type=float, default=0.3, help="Influence graph density")
    sim_parser.add_argument("--influence-strength", type=float, default=0.3, help="Influence strength")
    sim_parser.add_argument("--belief-threshold", type=float, default=0.5, help="Belief threshold for voting")
    sim_parser.add_argument("--voting-rule", type=str, default="approval", choices=["approval", "plurality", "weighted_voting"], help="Voting rule")

    # run-experiment subcommand
    exp_parser = subparsers.add_parser("run-experiment", help="Run a full experiment")
    exp_parser.add_argument("--experiment-type", "-t", type=str, default="consensus", choices=["consensus", "minority", "oppression"], help="Type of experiment")
    exp_parser.add_argument("--num-agents", "-n", type=int, default=100, help="Number of agents")
    exp_parser.add_argument("--num-majority", type=int, default=80, help="Number of majority agents (minority experiment)")
    exp_parser.add_argument("--num-minority", type=int, default=20, help="Number of minority agents (minority experiment)")
    exp_parser.add_argument("--num-candidates", "-c", type=int, default=3, help="Number of candidates")
    exp_parser.add_argument("--num-rounds", "-r", type=int, default=50, help="Number of voting rounds")
    exp_parser.add_argument("--iterations", "-i", type=int, default=100, help="Number of simulation iterations")

    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze saved voting data")
    analyze_parser.add_argument("--input", "-i", type=str, required=True, help="Input data file")
    analyze_parser.add_argument("--visualize", action="store_true", help="Generate visualization")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run-sim":
        result = run_simulation(args)
        return 0
    elif args.command == "run-experiment":
        result = run_experiment(args)
        return 0
    elif args.command == "analyze":
        result = analyze_data(args)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
