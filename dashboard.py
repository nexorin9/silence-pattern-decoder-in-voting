#!/usr/bin/env python3
"""
Streamlit dashboard for Silence Pattern Decoder.

Simple web dashboard to visualize running simulations and results.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.influence import InfluenceGraph
from src.voting import VotingSystem
from src.simulation import SimulationEngine
from src.pattern_detector import PatternDetector


def initialize_session_state():
    """Initialize session state variables."""
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None
    if "abstention_history" not in st.session_state:
        st.session_state.abstention_history = []
    if "agents" not in st.session_state:
        st.session_state.agents = []
    if "influence_graph" not in st.session_state:
        st.session_state.influence_graph = None


def generate_simulation_agents(num_agents: int, num_candidates: int) -> List[Agent]:
    """Generate random agents for simulation."""
    agents = []
    for i in range(num_agents):
        belief = [np.random.random(), np.random.random()]
        influence_tolerance = np.random.beta(2, 5)
        opinion_strength = np.random.random()

        agent = Agent(
            agent_id=i,
            belief=belief,
            influence_tolerance=influence_tolerance,
            opinion_strength=opinion_strength,
            voting_history=[]
        )
        agents.append(agent)
    return agents


def generate_influence_graph(agents: List[Agent], density: float) -> InfluenceGraph:
    """Generate random influence graph."""
    graph = InfluenceGraph()
    for agent in agents:
        graph.add_agent(agent)

    n = len(agents)
    max_edges = n * (n - 1)
    num_edges = int(max_edges * density)

    existing_edges = set()
    attempts = 0

    while len(existing_edges) < num_edges and attempts < num_edges * 10:
        source = np.random.choice(agents)
        target = np.random.choice(agents)
        attempts += 1

        if source.agent_id != target.agent_id:
            edge_key = (source.agent_id, target.agent_id)
            if edge_key not in existing_edges:
                weight = np.random.uniform(0.1, 0.9)
                graph.add_edge(source, target, weight)
                existing_edges.add(edge_key)

    return graph


def run_simulation(
    num_agents: int,
    num_candidates: int,
    num_rounds: int,
    influence_density: float,
    voting_rule: str
) -> Dict[str, Any]:
    """Run simulation and return results."""
    engine = SimulationEngine(
        num_agents=num_agents,
        num_candidates=num_candidates,
        num_belief_dimensions=2,
        seed=None
    )

    agents = engine.generate_random_agents(num_agents, 2)
    influence_graph = engine.generate_random_influence_graph(agents, density=influence_density)

    engine.candidates = [f"Candidate_{i}" for i in range(num_candidates)]

    result = engine.run_simulation(
        num_rounds=num_rounds,
        agents=agents,
        influence_graph=influence_graph,
        voting_rule=voting_rule,
        influence_strength=0.3,
        belief_threshold=0.5,
        verbose=False
    )

    return {
        "result": result,
        "agents": agents,
        "influence_graph": influence_graph
    }


def plot_abstention_timeline(abstention_rates: List[float]) -> plt.Figure:
    """Create abstention timeline plot."""
    fig, ax = plt.subplots(figsize=(10, 4))

    rounds = list(range(len(abstention_rates)))
    ax.plot(rounds, abstention_rates, linewidth=2, color="#2E86AB")
    ax.scatter(rounds, abstention_rates, s=20, color="#A23B72", alpha=0.7)

    ax.set_xlabel("Voting Round", fontsize=12)
    ax.set_ylabel("Abstention Rate", fontsize=12)
    ax.set_title("Abstention Rate Over Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(lambda x, p: f"{x:.1%}")

    plt.tight_layout()
    return fig


def display_pattern_analysis(result: Dict[str, Any]) -> None:
    """Display pattern detection results."""
    st.subheader("Pattern Detection")

    if not result.get("rounds"):
        st.warning("No simulation data available for pattern analysis")
        return

    voting_data = {
        "rounds": [
            {
                "abstention_rate": r.abstention_rate,
                "winner": r.winner,
                "vote_distribution": r.vote_distribution,
                "belief_variance": r.belief_variance
            }
            for r in result["rounds"]
        ]
    }

    detector = PatternDetector()
    patterns = detector.analyze_all_patterns(voting_data)

    col1, col2, col3 = st.columns(3)

    for i, (pattern_name, pattern_result) in enumerate(patterns.items()):
        if i == 0:
            target_col = col1
        elif i == 1:
            target_col = col2
        else:
            target_col = col3

        with target_col:
            st.metric(
                label=pattern_name.replace("_", " ").title(),
                value=f"{pattern_result.score:.2f}",
                delta=f"Conf: {pattern_result.confidence:.2f}"
            )


def main():
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Silence Pattern Decoder Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Silence Pattern Decoder 📊")
    st.markdown("### Social Science Voting Simulation Dashboard")

    initialize_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Controls")

        num_agents = st.slider(
            "Number of Agents",
            min_value=20,
            max_value=500,
            value=100,
            step=10
        )

        num_candidates = st.slider(
            "Number of Candidates",
            min_value=2,
            max_value=10,
            value=3,
            step=1
        )

        num_rounds = st.slider(
            "Number of Rounds",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )

        influence_density = st.slider(
            "Influence Graph Density",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )

        voting_rule = st.selectbox(
            "Voting Rule",
            options=["approval", "plurality", "weighted_voting"],
            format_func=lambda x: x.replace("_", " ").title()
        )

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                st.session_state.simulation_running = True

                result = run_simulation(
                    num_agents=num_agents,
                    num_candidates=num_candidates,
                    num_rounds=num_rounds,
                    influence_density=influence_density,
                    voting_rule=voting_rule
                )

                st.session_state.simulation_result = result
                st.session_state.abstention_history = [
                    r.abstention_rate for r in result["result"].rounds
                ]

                st.session_state.simulation_running = False
                st.rerun()

    # Main content area
    if st.session_state.simulation_result:
        result = st.session_state.simulation_result
        simulation_result = result["result"]

        st.subheader("Simulation Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Final Abstention Rate",
                f"{simulation_result.final_abstention_rate:.1%}"
            )

        with col2:
            st.metric(
                "Avg Abstention Rate",
                f"{simulation_result.avg_abstention_rate:.1%}"
            )

        with col3:
            st.metric(
                "Consensus Score",
                f"{simulation_result.consensus_score:.2f}"
            )

        with col4:
            st.metric(
                "Last Round Winner",
                str(simulation_result.rounds[-1].winner) if simulation_result.rounds else "N/A"
            )

        # Abstention timeline
        st.subheader("Abstention Timeline")
        fig = plot_abstention_timeline(st.session_state.abstention_history)
        st.pyplot(fig)

        # Pattern analysis
        display_pattern_analysis(result)

        # Round-by-round details
        st.subheader("Round-by-Round Details")
        details_data = []
        for i, round_result in enumerate(simulation_result.rounds):
            details_data.append({
                "Round": i,
                "Abstention Rate": f"{round_result.abstention_rate:.1%}",
                "Winner": round_result.winner,
                "Belief Variance": f"{round_result.belief_variance[0]:.3f}" if round_result.belief_variance else "N/A"
            })

        st.dataframe(details_data, use_container_width=True)

        # Result summary
        with st.expander("View Complete Results"):
            st.json({
                "num_agents": simulation_result.num_agents,
                "num_rounds": simulation_result.num_rounds,
                "num_candidates": simulation_result.num_candidates,
                "avg_abstention_rate": simulation_result.avg_abstention_rate,
                "final_abstention_rate": simulation_result.final_abstention_rate,
                "consensus_score": simulation_result.consensus_score
            })

    else:
        st.info(
            "Configure simulation parameters in the sidebar and click "
            "**Run Simulation** to begin."
        )

        # Show sample instructions
        st.markdown("### How to Use")
        st.markdown("""
        1. Adjust simulation parameters in the sidebar
        2. Click "Run Simulation" to execute
        3. View results including:
           - Abstention rate timeline
           - Pattern detection analysis
           - Round-by-round details
        """)

        st.markdown("### Dashboard Features")
        st.markdown("""
        - Real-time simulation execution
        - Pattern detection for abstention analysis
        - Visual timeline of abstention rates
        - Configurable voting rules
        """)


if __name__ == "__main__":
    main()
