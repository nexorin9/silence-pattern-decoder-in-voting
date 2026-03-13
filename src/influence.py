"""Influence model graph for representingsocial influence relationships."""

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .agent import Agent


@dataclass
class InfluenceEdge:
    """Represents an edge in the influence graph."""

    source: int
    target: int
    weight: float

    def __post_init__(self) -> None:
        if not (0 <= self.weight <= 1):
            raise ValueError("weight must be between 0 and 1")


class InfluenceGraph:
    """
    Represents social influence relationships between voting agents.

    Uses a directed graph where edges represent influence relationships.
    An edge from agent A to agent B means A can influence B's beliefs.
    """

    def __init__(self) -> None:
        """Initialize an empty influence graph."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._agent_map: Dict[int, Agent] = {}

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the graph.

        Args:
            agent: The Agent instance to add
        """
        self.graph.add_node(agent.agent_id, agent=agent)
        self._agent_map[agent.agent_id] = agent

    def add_edge(self, agent1: Agent, agent2: Agent, weight: float = 1.0) -> None:
        """
        Add an influence edge between two agents.

        Args:
            agent1: Source agent (influencer)
            agent2: Target agent (influencee)
            weight: Strength of influence (0 to 1)
        """
        if agent1.agent_id not in self.graph.nodes:
            self.add_agent(agent1)
        if agent2.agent_id not in self.graph.nodes:
            self.add_agent(agent2)

        self.graph.add_edge(
            agent1.agent_id,
            agent2.agent_id,
            weight=weight
        )

    def add_edge_by_id(self, agent1_id: int, agent2_id: int, weight: float = 1.0) -> None:
        """
        Add an influence edge using agent IDs.

        Args:
            agent1_id: Source agent ID (influencer)
            agent2_id: Target agent ID (influencee)
            weight: Strength of influence (0 to 1)
        """
        if agent1_id not in self.graph.nodes or agent2_id not in self.graph.nodes:
            raise ValueError("Both agents must be added to the graph first")

        self.graph.add_edge(agent1_id, agent2_id, weight=weight)

    def get_influence_sources(self, agent: Agent) -> List[Tuple[Agent, float]]:
        """
        Get agents that influence the given agent.

        Args:
            agent: The target agent

        Returns:
            List of (influencer_agent, weight) tuples
        """
        sources: List[Tuple[Agent, float]] = []
        for source_id in self.graph.predecessors(agent.agent_id):
            influence_weight = self.graph[source_id][agent.agent_id]["weight"]
            source_agent = self._agent_map.get(source_id)
            if source_agent:
                sources.append((source_agent, influence_weight))
        return sources

    def get_influence_targets(self, agent: Agent) -> List[Tuple[Agent, float]]:
        """
        Get agents that the given agent influences.

        Args:
            agent: The source agent

        Returns:
            List of (influenced_agent, weight) tuples
        """
        targets: List[Tuple[Agent, float]] = []
        for target_id in self.graph.successors(agent.agent_id):
            influence_weight = self.graph[agent.agent_id][target_id]["weight"]
            target_agent = self._agent_map.get(target_id)
            if target_agent:
                targets.append((target_agent, influence_weight))
        return targets

    def get_in_degree(self, agent: Agent) -> int:
        """Get the number of agents influencing this agent."""
        return self.graph.in_degree(agent.agent_id)

    def get_out_degree(self, agent: Agent) -> int:
        """Get the number of agents this agent influences."""
        return self.graph.out_degree(agent.agent_id)

    def get_total_influence(self, agent: Agent, inbound: bool = True) -> float:
        """
        Get total influence strength for an agent.

        Args:
            agent: The agent
            inbound: If True, sum inbound influence; else sum outbound

        Returns:
            Sum of influence weights
        """
        if inbound:
            return sum(
                self.graph[src][agent.agent_id]["weight"]
                for src in self.graph.predecessors(agent.agent_id)
            )
        else:
            return sum(
                self.graph[agent.agent_id][tgt]["weight"]
                for tgt in self.graph.successors(agent.agent_id)
            )

    def remove_edge(self, agent1: Agent, agent2: Agent) -> None:
        """Remove an influence edge between two agents."""
        self.graph.remove_edge(agent1.agent_id, agent2.agent_id)

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent and all its influence edges."""
        self.graph.remove_node(agent.agent_id)
        if agent.agent_id in self._agent_map:
            del self._agent_map[agent.agent_id]

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the influence graph.

        Returns:
            NumPy array representing the adjacency matrix
        """
        return nx.to_numpy_array(self.graph)

    def get_neighbors(self, agent: Agent, include_weight: bool = False) -> List[Any]:
        """
        Get all neighbors (both sources and targets) of an agent.

        Args:
            agent: The agent
            include_weight: Whether to include edge weights

        Returns:
            List of neighbor agents or (neighbor, weight) tuples
        """
        neighbors = set()
        for src in self.graph.predecessors(agent.agent_id):
            neighbors.add(src)
        for tgt in self.graph.successors(agent.agent_id):
            neighbors.add(tgt)

        if include_weight:
            result: List[Any] = []
            for neighbor_id in neighbors:
                weight = self.graph[agent.agent_id][neighbor_id].get("weight", 1.0)
                result.append((self._agent_map[neighbor_id], weight))
            return result
        return [self._agent_map[n] for n in neighbors]

    def copy(self) -> "InfluenceGraph":
        """Create a deep copy of the influence graph."""
        new_graph = InfluenceGraph()
        new_graph.graph = self.graph.copy()
        new_graph._agent_map = self._agent_map.copy()
        return new_graph

    def draw_graph(
        self,
        show_labels: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the influence graph.

        Args:
            show_labels: Whether to show agent IDs on nodes
            figsize: Figure size as (width, height)
            save_path: If provided, save to file instead of showing
        """
        plt.figure(figsize=figsize)

        # Get positions using.spring layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=500,
            node_color="lightblue",
            edgecolors="black"
        )

        # Draw edges with width based on weight
        edges = self.graph.edges()
        weights = [self.graph[u][v]["weight"] for u, v in edges]

        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edges,
            width=[w * 3 for w in weights],
            alpha=0.6,
            edge_color="gray",
            arrows=True,
            arrowsize=20
        )

        # Draw labels
        if show_labels:
            labels = {
                node_id: f"A{node_id}"
                for node_id in self.graph.nodes()
            }
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels=labels,
                font_size=10,
                font_weight="bold"
            )

        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def get_node_count(self) -> int:
        """Get the number of agents in the graph."""
        return self.graph.number_of_nodes()

    def get_edge_count(self) -> int:
        """Get the number of influence edges."""
        return self.graph.number_of_edges()

    def get_agents(self) -> List[Agent]:
        """Get all agents in the graph."""
        return list(self._agent_map.values())

    def get_agent_by_id(self, agent_id: int) -> Optional[Agent]:
        """Get an agent by its ID."""
        return self._agent_map.get(agent_id)

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.graph.clear()
        self._agent_map.clear()
