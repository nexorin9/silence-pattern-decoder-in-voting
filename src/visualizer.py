"""Visualization module for simulation results."""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx


def plot_abstention_timeline(
    abstention_rates: List[float],
    title: str = "Abstention Rate Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a time series plot of abstention rates.

    Args:
        abstention_rates: List of abstention rates per round
        title: Plot title
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    rounds = list(range(len(abstention_rates)))

    # Plot the data
    ax.plot(rounds, abstention_rates, linewidth=2, color="#2E86AB")

    # Add scatter points
    ax.scatter(rounds, abstention_rates, s=20, color="#A23B72", alpha=0.7)

    # Styling
    ax.set_xlabel("Voting Round", fontsize=12)
    ax.set_ylabel("Abstention Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(lambda x, p: f"{x:.1%}")

    # Add horizontal line at mean
    mean_rate = np.mean(abstention_rates) if abstention_rates else 0
    ax.axhline(y=mean_rate, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_rate:.2%}")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_belief_distribution(
    agents: List[Any],
    round_num: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Optional[plt.Figure]:
    """
    Create a histogram of agent belief distributions.

    Args:
        agents: List of Agent instances
        round_num: Optional round number for title
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if round_num is not None:
        fig.suptitle(f"Belief Distribution - Round {round_num}", fontsize=14, fontweight="bold")

    # Extract belief values
    beliefs = [agent.belief for agent in agents]
    num_dimensions = len(beliefs[0]) if beliefs else 0

    # Plot each dimension
    colors = ["#F18F01", "#C73E1D", "#3B7AA3", "#6A9F4C"]
    bins = np.linspace(0, 1, 20)

    for dim in range(min(num_dimensions, 2)):
        dim_beliefs = [b[dim] for b in beliefs]
        ax = axes[dim]

        # Histogram
        ax.hist(dim_beliefs, bins=bins, alpha=0.7, color=colors[dim % len(colors)])
        ax.axvline(np.mean(dim_beliefs), color="black", linestyle="--", linewidth=1.5)

        ax.set_xlabel("Belief Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Belief Dimension {dim + 1}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    if num_dimensions > 2:
        # Add combined distribution
        ax = axes[2] if len(axes) > 2 else axes[1]
        mean_beliefs = [np.mean(b) for b in beliefs]
        ax.hist(mean_beliefs, bins=bins, alpha=0.7, color="#555555")
        ax.axvline(np.mean(mean_beliefs), color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Mean Belief", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Combined Belief Distribution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_influence_network(
    influence_graph: Any,
    agents: List[Any],
    show_labels: bool = True,
    node_size: int = 300,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[plt.Figure]:
    """
    Create a network graph visualization of influence relationships.

    Args:
        influence_graph: InfluenceGraph instance
        agents: List of Agent instances
        show_labels: Whether to show agent IDs
        node_size: Size of nodes
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    # Get positions using spring layout
    pos = influence_graph._get_positions() if hasattr(influence_graph, "_get_positions") else None

    if pos is None:
        # Use circular layout as fallback
        n = len(agents)
        pos = {
            agent.agent_id: (
                np.cos(2 * np.pi * i / n),
                np.sin(2 * np.pi * i / n)
            )
            for i, agent in enumerate(agents)
        }

    fig, ax = plt.subplots(figsize=figsize)

    # Separate dominant agent for special highlighting
    dominant_agents = [a for a in agents if getattr(a, "influence_tolerance", 1) < 0.3]
    normal_agents = [a for a in agents if a not in dominant_agents]

    # Draw normal agents
    normal_ids = [a.agent_id for a in normal_agents]
    normal_pos = {k: v for k, v in pos.items() if k in normal_ids}
    nx_draw_network_nodes_with_pos(ax, normal_pos, node_size=node_size, node_color="#6B9AC4", edgecolors="black")

    # Draw dominant agents
    if dominant_agents:
        dominant_ids = [a.agent_id for a in dominant_agents]
        dominant_pos = {k: v for k, v in pos.items() if k in dominant_ids}
        nx_draw_network_nodes_with_pos(ax, dominant_pos, node_size=node_size * 1.5, node_color="#E63946", edgecolors="black")

    # Draw edges
    edges = influence_graph.graph.edges() if hasattr(influence_graph, "graph") else []
    edge_weights = [0.5 for _ in edges]

    for i, (u, v) in enumerate(edges):
        if hasattr(influence_graph, "graph"):
            edge_weights[i] = influence_graph.graph[u][v].get("weight", 0.5)

    # Draw edges with width based on weight
    for i, (u, v) in enumerate(edges):
        weight = edge_weights[i]
        ax.annotate(
            "",
            xy=pos[v], xycoords="data",
            xytext=pos[u], textcoords="data",
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                linewidth=weight * 3,
                alpha=0.6,
                color="#555555"
            )
        )

    # Draw labels
    if show_labels:
        labels = {agent.agent_id: f"A{agent.agent_id}" for agent in agents}
        for agent_id, (x, y) in pos.items():
            ax.text(x, y + 0.05, labels.get(agent_id, str(agent_id)),
                   fontsize=9, ha="center", va="center", fontweight="bold")

    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_consensus_vs_abstention(
    consensus_scores: List[float],
    abstention_rates: List[float],
    title: str = "Consensus vs Abstention",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[plt.Figure]:
    """
    Create a scatter plot of consensus scores vs abstention rates.

    Args:
        consensus_scores: List of consensus scores
        abstention_rates: List of abstention rates
        title: Plot title
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter points
    ax.scatter(abstention_rates, consensus_scores, s=100, alpha=0.6, color="#2E86AB")

    # Add trend line
    if len(abstention_rates) >= 3:
        z = np.polyfit(abstention_rates, consensus_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(abstention_rates), max(abstention_rates), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")

    ax.set_xlabel("Abstention Rate", fontsize=12)
    ax.set_ylabel("Consensus Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Format axes as percentages
    ax.xaxis.set_major_formatter(lambda x, p: f"{x:.1%}")
    ax.yaxis.set_major_formatter(lambda x, p: f"{x:.1%}")

    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_belief_changes(
    initial_beliefs: List[List[float]],
    final_beliefs: List[List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Optional[plt.Figure]:
    """
    Create a visualization of belief changes between rounds.

    Args:
        initial_beliefs: List of initial belief vectors
        final_beliefs: List of final belief vectors
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    if not initial_beliefs or not final_beliefs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Belief Evolution", fontsize=14, fontweight="bold")

    # Calculate belief changes
    belief_changes = []
    for initial, final in zip(initial_beliefs, final_beliefs):
        change = sum((f - i) ** 2 for i, f in zip(initial, final)) / len(initial)
        belief_changes.append(change)

    # Plot 1: Change distribution
    axes[0].hist(belief_changes, bins=20, alpha=0.7, color="#F18F01")
    axes[0].axvline(np.mean(belief_changes), color="black", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Belief Change (MSE)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Distribution of Belief Changes", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Change vs initial belief
    mean_initial = [np.mean(b) for b in initial_beliefs]
    axes[1].scatter(mean_initial, belief_changes, alpha=0.6, color="#C73E1D")
    axes[1].set_xlabel("Initial Mean Belief", fontsize=11)
    axes[1].set_ylabel("Belief Change", fontsize=11)
    axes[1].set_title("Belief Change vs Initial Belief", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_multi_round_comparison(
    round_data_list: List[Dict[str, Any]],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Optional[plt.Figure]:
    """
    Create a comparison plot across multiple simulation rounds.

    Args:
        round_data_list: List of round data dictionaries
        metrics: List of metrics to plot
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    if metrics is None:
        metrics = ["abstention_rate", "belief_variance"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    rounds = list(range(len(round_data_list)))

    for i, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in round_data_list]

        axes[i].plot(rounds, values, linewidth=2, marker="o", markersize=3)
        axes[i].set_xlabel("Round", fontsize=11)
        axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        axes[i].set_title(f"{metric.replace('_', ' ').title()} Over Time", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_statistics_overview(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> Optional[plt.Figure]:
    """
    Create a comprehensive statistics overview plot.

    Args:
        results: Analysis results dictionary
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    rounds = results.get("rounds", [])
    if not rounds:
        return None

    abstention_rates = [r.get("abstention_rate", 0) for r in rounds]
    belief_variances = [r.get("belief_variance", 0) for r in rounds if r.get("belief_variance") is not None]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Simulation Statistics Overview", fontsize=14, fontweight="bold")

    # Top-left: Abstention timeline
    rounds_range = list(range(len(abstention_rates)))
    axes[0, 0].plot(rounds_range, abstention_rates, linewidth=2, color="#2E86AB")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Abstention Rate")
    axes[0, 0].set_title("Abstention Rate Over Time")
    axes[0, 0].yaxis.set_major_formatter(lambda x, p: f"{x:.1%}")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: belief variance timeline
    if belief_variances:
        axes[0, 1].plot(list(range(len(belief_variances))), belief_variances, linewidth=2, color="#F18F01")
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Belief Variance")
        axes[0, 1].set_title("Belief Variance Over Time")
        axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Abstention distribution
    if abstention_rates:
        axes[1, 0].hist(abstention_rates, bins=15, alpha=0.7, color="#C73E1D")
        axes[1, 0].axvline(np.mean(abstention_rates), color="black", linestyle="--", linewidth=1.5)
        axes[1, 0].set_xlabel("Abstention Rate")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Abstention Rate Distribution")
        axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Statistics summary
    if belief_variances:
        axes[1, 1].bar(["Abstention", "Belief Variance"], [np.mean(abstention_rates), np.mean(belief_variances)], color=["#2E86AB", "#F18F01"])
        axes[1, 1].set_ylabel("Mean Value")
        axes[1, 1].set_title("Mean Statistics")
        axes[1, 1].yaxis.set_major_formatter(lambda x, p: f"{x:.2f}" if x > 1 else f"{x:.1%}" if x < 1 else f"{x:.2f}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def plot_pattern_scores(
    pattern_results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[plt.Figure]:
    """
    Create a plot comparing pattern detection scores.

    Args:
        pattern_results: Dictionary mapping pattern types to results
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    if not pattern_results:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    pattern_names = list(pattern_results.keys())
    scores = [pattern_results[p].get("score", 0) for p in pattern_names]
    confidences = [pattern_results[p].get("confidence", 0) for p in pattern_names]

    x = np.arange(len(pattern_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, scores, width, label="Score", color="#2E86AB", alpha=0.7)
    bars2 = ax.bar(x + width/2, confidences, width, label="Confidence", color="#F18F01", alpha=0.7)

    ax.set_xlabel("Pattern Type", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Pattern Detection Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig


def nx_draw_network_nodes_with_pos(
    ax: plt.Axes,
    pos: Dict[Any, Tuple[float, float]],
    node_size: int = 300,
    node_color: str = "lightblue",
    edgecolors: str = "black"
) -> None:
    """Helper function to draw network nodes without networkx dependency."""
    import matplotlib.patches as mpatches

    for (x, y) in pos.values():
        circle = mpatches.Circle((x, y), radius=0.03, facecolor=node_color, edgecolor=edgecolors)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)


def create_summary_grid(
    results_list: List[Dict[str, Any]],
    labels: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> Optional[plt.Figure]:
    """
    Create a grid plot comparing multiple simulation results.

    Args:
        results_list: List of simulation results dictionaries
        labels: Labels for each result
        save_path: If provided, save to file instead of showing
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object
    """
    if labels is None:
        labels = [f"Result {i+1}" for i in range(len(results_list))]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Simulation Comparison", fontsize=14, fontweight="bold")

    color_palette = ["#2E86AB", "#C73E1D", "#F18F01", "#6B9AC4", "#555555"]

    for i, (result, label) in enumerate(zip(results_list, labels)):
        color = color_palette[i % len(color_palette)]
        rounds = result.get("rounds", [])

        if not rounds:
            continue

        abstention_rates = [r.get("abstention_rate", 0) for r in rounds]
        rounds_range = list(range(len(abstention_rates)))

        ax = axes[i // 2, i % 2]
        ax.plot(rounds_range, abstention_rates, linewidth=2, color=color, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel("Abstention Rate")
        ax.set_title(f"{label} - Abstention Trend")
        ax.yaxis.set_major_formatter(lambda x, p: f"{x:.1%}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    return fig
