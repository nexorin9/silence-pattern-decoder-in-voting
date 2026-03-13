#!/usr/bin/env python3
"""
Performance benchmark script for Silence Pattern Decoder.
Measures simulation performance at different scales to identify optimal agent counts.
"""

import time
import tracemalloc
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
parent_dir = str(project_root.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from silence_decoder.src.agent import Agent
from silence_decoder.src.influence import InfluenceGraph
from silence_decoder.src.simulation import SimulationEngine


def measure_simulation_performance(
    num_agents: int,
    num_belief_dimensions: int = 3,
    num_rounds: int = 50,
    influence_density: float = 0.3,
) -> dict:
    """
    Measure simulation performance for given parameters.

    Returns:
        dict with runtime_seconds, memory_mb, agents_per_second
    """
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()

    # Create simulation
    engine = SimulationEngine()
    agents = engine.generate_random_agents(num_agents, num_belief_dimensions)
    influence_graph = engine.generate_random_influence_graph(agents, influence_density)

    # Run simulation
    result = engine.run_simulation(
        num_rounds=num_rounds,
        verbose=False,
    )

    # End timing
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_seconds = end_time - start_time
    memory_mb = peak / (1024 * 1024)

    return {
        "num_agents": num_agents,
        "num_belief_dimensions": num_belief_dimensions,
        "num_rounds": num_rounds,
        "runtime_seconds": runtime_seconds,
        "memory_mb": memory_mb,
        "agents_per_second": num_agents * num_rounds / runtime_seconds if runtime_seconds > 0 else 0,
    }


def run_scalability_test(
    agent_counts: list = None,
    belief_dimensions: list = None,
    num_rounds: int = 50,
    influence_density: float = 0.3,
    output_dir: str = None,
) -> list:
    """
    Run scalability tests across different agent counts.

    Args:
        agent_counts: List of agent counts to test
        belief_dimensions: List of belief dimensions to test
        num_rounds: Number of simulation rounds
        influence_density: Density of influence graph
        output_dir: Directory to save results

    Returns:
        List of performance results
    """
    if agent_counts is None:
        agent_counts = [50, 100, 200, 500, 1000]
    if belief_dimensions is None:
        belief_dimensions = [3]

    results = []
    output_path = Path(output_dir) if output_dir else Path(__file__).parent.parent / "data" / "benchmarks"

    print("=" * 60)
    print("Performance Benchmark: Scalability Test")
    print("=" * 60)
    print(f"Agent counts: {agent_counts}")
    print(f"Belief dimensions: {belief_dimensions}")
    print(f"Rounds per simulation: {num_rounds}")
    print(f"Influence density: {influence_density}")
    print("=" * 60)
    print()

    for dims in belief_dimensions:
        print(f"\n--- Testing with {dims} belief dimensions ---")
        for num_agents in agent_counts:
            print(f"Testing {num_agents} agents...", end=" ")
            try:
                result = measure_simulation_performance(
                    num_agents=num_agents,
                    num_belief_dimensions=dims,
                    num_rounds=num_rounds,
                    influence_density=influence_density,
                )
                results.append(result)

                print(f"done")
                print(f"  Runtime: {result['runtime_seconds']:.2f}s")
                print(f"  Memory: {result['memory_mb']:.2f} MB")
                print(f"  Throughput: {result['agents_per_second']:.0f} agent-rounds/s")

            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "num_agents": num_agents,
                    "num_belief_dimensions": dims,
                    "num_rounds": num_rounds,
                    "runtime_seconds": -1,
                    "memory_mb": -1,
                    "agents_per_second": -1,
                    "error": str(e),
                })

    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    # Filter valid results for summary
    valid_results = [r for r in results if r["runtime_seconds"] > 0]
    if valid_results:
        for r in valid_results:
            print(f"{r['num_agents']:5d} agents: {r['runtime_seconds']:6.2f}s, "
                  f"{r['memory_mb']:6.2f}MB, {r['agents_per_second']:8.0f} agent-rounds/s")

    # Save results
    if output_path and valid_results:
        output_path.mkdir(parents=True, exist_ok=True)
        import csv
        output_file = output_path / "scalability_benchmark.csv"
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=valid_results[0].keys())
            writer.writeheader()
            writer.writerows(valid_results)
        print(f"\nResults saved to: {output_file}")

    return results


def plot_scalability(results: list, output_dir: str = None):
    """
    Create scalability visualization plots.

    Args:
        results: List of performance results from run_scalability_test
        output_dir: Directory to save plots
    """
    valid_results = [r for r in results if r["runtime_seconds"] > 0]
    if not valid_results:
        print("No valid results to plot")
        return

    output_path = Path(output_dir) if output_dir else Path(__file__).parent.parent / "docs"
    output_path.mkdir(parents=True, exist_ok=True)

    # Sort by agent count
    valid_results.sort(key=lambda x: x["num_agents"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = [r["num_agents"] for r in valid_results]
    runtimes = [r["runtime_seconds"] for r in valid_results]
    memories = [r["memory_mb"] for r in valid_results]
    throughput = [r["agents_per_second"] for r in valid_results]

    # Plot 1: Runtime vs Agents
    axes[0, 0].plot(agents, runtimes, "o-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Number of Agents")
    axes[0, 0].set_ylabel("Runtime (seconds)")
    axes[0, 0].set_title("Simulation Runtime vs. Agent Count")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Memory vs Agents
    axes[0, 1].plot(agents, memories, "o-", linewidth=2, markersize=8, color="orange")
    axes[0, 1].set_xlabel("Number of Agents")
    axes[0, 1].set_ylabel("Peak Memory (MB)")
    axes[0, 1].set_title("Memory Usage vs. Agent Count")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Throughput vs Agents
    axes[1, 0].plot(agents, throughput, "o-", linewidth=2, markersize=8, color="green")
    axes[1, 0].set_xlabel("Number of Agents")
    axes[1, 0].set_ylabel("Throughput (agent-rounds/s)")
    axes[1, 0].set_title("Throughput vs. Agent Count")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Operations per second per agent
    per_agent_throughput = [r / a for r, a in zip(throughput, agents)]
    axes[1, 1].plot(agents, per_agent_throughput, "o-", linewidth=2, markersize=8, color="red")
    axes[1, 1].set_xlabel("Number of Agents")
    axes[1, 1].set_ylabel("Operations per Agent (agent-rounds/s)")
    axes[1, 1].set_title("Per-Agent Throughput vs. Agent Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = output_path / "scalability_benchmark.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_file}")
    plt.close()


def find_optimal_agent_count(
    results: list,
    max_runtime: float = 60.0,
) -> dict:
    """
    Find optimal agent count for interactive use (completing in <60s).

    Args:
        results: List of performance results
        max_runtime: Maximum acceptable runtime in seconds

    Returns:
        dict with optimal_count and recommendations
    """
    valid_results = [r for r in results if r["runtime_seconds"] > 0 and r["runtime_seconds"] <= max_runtime]

    if not valid_results:
        return {
            "optimal_count": None,
            "recommendation": f"No configurations completed within {max_runtime}s",
            "all_results": results,
        }

    # Find the largest count within time limit
    optimal = max(valid_results, key=lambda x: x["num_agents"])
    memory_limit = 4096  # 4GB

    return {
        "optimal_count": optimal["num_agents"],
        "recommended_rounds": 50,
        "runtime_at_optimal": optimal["runtime_seconds"],
        "memory_at_optimal": optimal["memory_mb"],
        "throughput_at_optimal": optimal["agents_per_second"],
        "recommendation": (
            f" optimum for interactive use: {optimal['num_agents']} agents "
            f"({optimal['runtime_seconds']:.1f}s runtime, {optimal['memory_mb']:.1f}MB memory)"
        ),
        "all_results": results,
        "can_scale_to_1000": any(r["num_agents"] == 1000 and r["runtime_seconds"] <= max_runtime for r in valid_results),
    }


def run_benchmark_suite(output_dir: str = None):
    """
    Run complete benchmark suite.

    Args:
        output_dir: Directory to save all benchmark results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "benchmarks"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" SILENCE PATTERN DECODER - PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)

    # Run primary scalability test
    agent_counts = [50, 100, 200, 500, 1000]
    belief_dims = [2, 3, 5]
    rounds = 50

    print(f"\n[1/3] Running scalability test...")
    results = run_scalability_test(
        agent_counts=agent_counts,
        belief_dimensions=belief_dims,
        num_rounds=rounds,
        output_dir=output_dir,
    )

    # Create visualization
    print(f"\n[2/3] Creating visualization...")
    plot_scalability(results, output_dir=output_dir)

    # Find optimal configuration
    print(f"\n[3/3] Analyzing optimal configuration...")
    optimal = find_optimal_agent_count(results, max_runtime=60.0)

    print("\n" + "=" * 70)
    print(" OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Optimal for interactive use: {optimal.get('optimal_count', 'N/A')}")
    print(f"Recommendation: {optimal.get('recommendation', 'N/A')}")

    # Save optimal configuration report
    report_file = output_dir / "benchmark_report.txt"
    with open(report_file, "w") as f:
        f.write("Performance Benchmark Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Optimal Agent Count: {optimal.get('optimal_count', 'N/A')}\n")
        f.write(f"Recommendation: {optimal.get('recommendation', 'N/A')}\n\n")

        f.write("All Results:\n")
        for r in results:
            f.write(f"  {r['num_agents']} agents: {r['runtime_seconds']:.2f}s, "
                   f"{r['memory_mb']:.2f}MB\n")

    print(f"\nReport saved to: {report_file}")

    return results, optimal


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance benchmark for Silence Pattern Decoder")
    parser.add_argument("--agents", type=int, nargs="+", default=None,
                       help="Agent counts to test (default: 50, 100, 200, 500, 1000)")
    parser.add_argument("--rounds", type=int, default=50,
                       help="Number of simulation rounds (default: 50)")
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                       help="Belief dimensions to test (default: 2, 3, 5)")
    parser.add_argument("--density", type=float, default=0.3,
                       help="Influence graph density (default: 0.3)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--plot-only", action="store_true",
                       help="Only create plots from existing results")
    parser.add_argument("--max-runtime", type=float, default=60.0,
                       help="Maximum acceptable runtime for 'optimal' classification")

    args = parser.parse_args()

    if args.plot_only:
        # Load existing results and plot
        output_path = Path(args.output) if args.output else Path(__file__).parent.parent / "data" / "benchmarks"
        import csv
        results_file = output_path / "scalability_benchmark.csv"
        if results_file.exists():
            with open(results_file) as f:
                reader = csv.DictReader(f)
                results = list(reader)
                # Convert string values to numeric
                for r in results:
                    r["num_agents"] = int(r["num_agents"])
                    r["num_belief_dimensions"] = int(r["num_belief_dimensions"])
                    r["runtime_seconds"] = float(r["runtime_seconds"])
                    r["memory_mb"] = float(r["memory_mb"])
                    r["agents_per_second"] = float(r["agents_per_second"])
            plot_scalability(results, output_dir=output_path)
        else:
            print(f"Results file not found: {results_file}")
            print("Run benchmark first without --plot-only")
    else:
        results, optimal = run_benchmark_suite(output_dir=args.output)
        print(f"\nComplete. Optimal: {optimal.get('recommendation', 'N/A')}")
