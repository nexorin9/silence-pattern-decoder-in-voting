"""Synthetic voting data generator."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import csv
import json
from pathlib import Path


@dataclass
class GroundTruth:
    """Ground truth configuration for synthetic data."""

    pattern_type: str
    params: Dict[str, Any]


class SyntheticVotingDataGenerator:
    """
    Generates synthetic voting datasets with known ground truth patterns.

    Supports consensus, minority influence, oppression, and random patterns.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.seed = seed
        self._noise_level = 0.05

    def set_noise_level(self, noise_level: float) -> None:
        """
        Set the noise level for data generation.

        Args:
            noise_level: Noise level (0 to 1)
        """
        self._noise_level = max(0, min(1, noise_level))

    def generate_synthetic_voting_data(
        self,
        num_agents: int,
        num_rounds: int,
        ground_truth: GroundTruth,
        num_candidates: int = 3,
        include_belief_history: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic voting data.

        Args:
            num_agents: Number of voting agents
            num_rounds: Number of voting rounds
            ground_truth: GroundTruth configuration with pattern type and parameters
            num_candidates: Number of candidate options
            include_belief_history: Whether to include belief state per agent

        Returns:
            List of vote records
        """
        data = []
        candidates = [f"Candidate_{i}" for i in range(num_candidates)]

        # Initialize agents
        agents = self._initialize_agents(num_agents, ground_truth)

        # Generate data for each round
        for round_num in range(num_rounds):
            # Apply pattern-specific logic
            pattern_weights = self._get_pattern_weights(round_num, ground_truth)

            for agent in agents:
                # Get agent's true preference
                true_preference = self._get_true_preference(agent, candidates)

                # Apply ground truth pattern influence
                agent_id = agent.get("agent_id", getattr(agent, "agent_id", None))
                if random.random() < pattern_weights.get(agent_id, 1.0):
                    #_vote according to pattern
                    vote = self._get_pattern_vote(agent, true_preference, candidates, ground_truth)
                else:
                    # Vote based on true preference (with some noise)
                    vote = true_preference if random.random() > self._noise_level else None

                # Determine abstention
                abstention = vote is None

                # Record vote
                agent_id = agent.get("agent_id", getattr(agent, "agent_id", None))
                vote_record = {
                    "round": round_num,
                    "agent_id": agent_id,
                    "original_belief": agent["belief"].copy(),
                    "vote": vote,
                    "abstention": abstention,
                    "true_preference": true_preference,
                    "pattern_weight": pattern_weights.get(agent_id, 1.0)
                }

                if include_belief_history:
                    vote_record["belief_history"] = agent["belief"].copy()

                data.append(vote_record)

        return data

    def _initialize_agents(
        self,
        num_agents: int,
        ground_truth: GroundTruth
    ) -> List[Dict[str, Any]]:
        """Initialize agents with belief states."""
        agents = []
        belief_dimensions = 2

        for i in range(num_agents):
            # Initialize random belief
            belief = [random.random() for _ in range(belief_dimensions)]

            agent = {
                "agent_id": i,
                "belief": belief,
                "influence_tolerance": random.uniform(0.1, 0.9),
                "opinion_strength": random.uniform(0.1, 1.0)
            }
            agents.append(agent)

        return agents

    def _get_pattern_weights(
        self,
        round_num: int,
        ground_truth: GroundTruth
    ) -> Dict[int, float]:
        """Get pattern influence weights for each agent."""
        pattern_type = ground_truth.pattern_type
        params = ground_truth.params
        num_agents = params.get("num_agents", 100)

        weights = {}

        if pattern_type == "consensus":
            # Most agents align with consensus
            consensus_strength = params.get("consensus_strength", 0.8)
            for i in range(num_agents):
                weights[i] = 1.0 if random.random() < consensus_strength else 0.0

        elif pattern_type == "minority_influence":
            # Minority group has higher weight
            minority_size = params.get("minority_size", 20)
            minority_influence = params.get("minority_influence", 0.7)
            for i in range(num_agents):
                if i < minority_size:
                    weights[i] = minority_influence
                else:
                    weights[i] = random.uniform(0, 0.2)

        elif pattern_type == "oppression":
            # Dominant agent has high influence
            oppressive_agents = params.get("oppressive_agents", [0])
            oppressive_weight = params.get("oppressive_weight", 0.9)
            compliance_rate = params.get("compliance_rate", 0.8)

            for i in range(num_agents):
                if i in oppressive_agents:
                    weights[i] = 1.0
                elif random.random() < compliance_rate:
                    weights[i] = oppressive_weight
                else:
                    weights[i] = random.uniform(0, 0.3)

        else:  # random pattern
            for i in range(num_agents):
                weights[i] = random.uniform(0, 1)

        return weights

    def _get_true_preference(
        self,
        agent: Dict[str, Any],
        candidates: List[str]
    ) -> Optional[str]:
        """Get agent's true preference based on belief alignment."""
        if not candidates:
            return None

        # Simple alignment: random candidate for now
        # In a real system, this would be based on belief-candidate alignment
        return random.choice(candidates)

    def _get_pattern_vote(
        self,
        agent: Dict[str, Any],
        true_preference: Optional[str],
        candidates: List[str],
        ground_truth: GroundTruth
    ) -> Optional[str]:
        """Get agent's vote based on ground truth pattern."""
        pattern_type = ground_truth.pattern_type
        params = ground_truth.params

        if pattern_type == "consensus":
            # Consensus: most agents vote for same candidate
            consensus_candidate = candidates[0]
            if random.random() < params.get("consensus_strength", 0.8):
                return consensus_candidate

        elif pattern_type == "minority_influence":
            # Minority influence: minority agents vote for their preferred candidate
            if agent["agent_id"] < params.get("minority_size", 20):
                return candidates[0] if random.random() < params.get("minority_alignment", 0.9) else None

        elif pattern_type == "oppression":
            # Oppression: agents vote differently based on oppression level
            oppressive_agents = params.get("oppressive_agents", [0])
            if agent["agent_id"] in oppressive_agents:
                # Dominant agents vote for their preferred candidate
                return candidates[0]
            elif random.random() < params.get("compliance_rate", 0.8):
                # Compliant agents abstain or vote against true preference
                return None if random.random() < params.get("abstention_rate", 0.3) else candidates[1]

        # Default to true preference with noise
        if random.random() < self._noise_level:
            return None  # Abstain due to noise
        return true_preference

    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filepath: str,
        include_belief_history: bool = False
    ) -> None:
        """
        Export data to CSV format.

        Args:
            data: Voting data to export
            filepath: Output file path
            include_belief_history: Whether to include belief history columns
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine columns
        base_columns = [
            "round", "agent_id", "vote", "abstention",
            "true_preference", "pattern_weight"
        ]

        if include_belief_history:
            base_columns.append("original_belief")

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=base_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)

    def export_to_json(
        self,
        data: List[Dict[str, Any]],
        ground_truth: GroundTruth,
        filepath: str
    ) -> None:
        """
        Export data to JSON format with metadata.

        Args:
            data: Voting data to export
            ground_truth: Ground truth configuration
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "metadata": {
                "num_agents": len(set(d["agent_id"] for d in data)),
                "num_rounds": len(set(d["round"] for d in data)),
                "num_candidates": 3,
                "ground_truth": {
                    "pattern_type": ground_truth.pattern_type,
                    "params": ground_truth.params
                },
                "noise_level": self._noise_level
            },
            "data": data
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def generate_multiple_datasets(
        self,
        ground_truth_configs: List[GroundTruth],
        num_agents: int,
        num_rounds: int,
        output_dir: str,
        num_datasets_per_config: int = 5
    ) -> Dict[str, int]:
        """
        Generate multiple datasets for different configurations.

        Args:
            ground_truth_configs: List of ground truth configurations
            num_agents: Number of agents
            num_rounds: Number of rounds
            output_dir: Output directory
            num_datasets_per_config: Number of datasets per configuration

        Returns:
            Dictionary mapping file paths to dataset index
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        for config in ground_truth_configs:
            pattern_dir = output_path / config.pattern_type
            pattern_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_datasets_per_config):
                # Create fresh generator for each dataset
                generator = SyntheticVotingDataGenerator(seed=self.seed + i * 100)
                generator.set_noise_level(self._noise_level)

                data = generator.generate_synthetic_voting_data(
                    num_agents=num_agents,
                    num_rounds=num_rounds,
                    ground_truth=config
                )

                # Export
                csv_path = pattern_dir / f"dataset_{i:03d}.csv"
                json_path = pattern_dir / f"dataset_{i:03d}.json"

                generator.export_to_csv(data, str(csv_path))
                generator.export_to_json(data, config, str(json_path))

                results[str(csv_path)] = i
                results[str(json_path)] = i

        return results


def generate_validation_dataset(
    num_agents: int = 100,
    num_rounds: int = 50,
    ground_truth_type: str = "consensus",
    output_dir: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], GroundTruth]:
    """
    Generate a single validation dataset.

    Args:
        num_agents: Number of agents
        num_rounds: Number of rounds
        ground_truth_type: Type of ground truth pattern
        output_dir: Optional output directory

    Returns:
        Tuple of (data, ground_truth)
    """
    ground_truth = GroundTruth(
        pattern_type=ground_truth_type,
        params={
            "num_agents": num_agents,
            f"{ground_truth_type}_strength": 0.7
        }
    )

    generator = SyntheticVotingDataGenerator(seed=42)
    generator.set_noise_level(0.1)

    data = generator.generate_synthetic_voting_data(
        num_agents=num_agents,
        num_rounds=num_rounds,
        ground_truth=ground_truth
    )

    if output_dir:
        generator.export_to_csv(data, str(Path(output_dir) / f"{ground_truth_type}_validation.csv"))
        generator.export_to_json(data, ground_truth, str(Path(output_dir) / f"{ground_truth_type}_validation.json"))

    return data, ground_truth


def get_ground_truth_definitions() -> Dict[str, Dict[str, Any]]:
    """Get definitions for different ground truth patterns."""
    return {
        "consensus": {
            "description": "Implicit consensus among voters",
            "params": {
                "consensus_strength": {"type": "float", "range": [0.5, 0.95]},
                "num_agents": {"type": "int", "range": [50, 200]},
                "num_rounds": {"type": "int", "range": [20, 100]}
            }
        },
        "minority_influence": {
            "description": "Minority group influencing majority voting",
            "params": {
                "minority_size": {"type": "int", "range": [10, 50]},
                "minority_influence": {"type": "float", "range": [0.5, 0.95]},
                "minority_alignment": {"type": "float", "range": [0.6, 1.0]},
                "num_agents": {"type": "int", "range": [100, 200]},
                "num_rounds": {"type": "int", "range": [30, 100]}
            }
        },
        "oppression": {
            "description": "Oppressive group dynamics causing suppressed voting",
            "params": {
                "oppressive_agents": {"type": "list", "default": [0]},
                "oppressive_weight": {"type": "float", "range": [0.7, 1.0]},
                "compliance_rate": {"type": "float", "range": [0.5, 0.95]},
                "abstention_rate": {"type": "float", "range": [0.1, 0.6]},
                "num_agents": {"type": "int", "range": [50, 150]},
                "num_rounds": {"type": "int", "range": [20, 80]}
            }
        },
        "random": {
            "description": "Random voting with no pattern",
            "params": {
                "num_agents": {"type": "int", "range": [50, 200]},
                "num_rounds": {"type": "int", "range": [20, 100]},
                "noise_level": {"type": "float", "range": [0.0, 0.3]}
            }
        }
    }


def main():
    """Generate synthetic datasets from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic voting data")
    parser.add_argument("--num-agents", "-n", type=int, default=100, help="Number of agents")
    parser.add_argument("--num-rounds", "-r", type=int, default=50, help="Number of rounds")
    parser.add_argument("--pattern", "-p", type=str, default="consensus",
                       choices=["consensus", "minority_influence", "oppression", "random"],
                       help="Ground truth pattern")
    parser.add_argument("--output", "-o", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level")

    args = parser.parse_args()

    ground_truth = GroundTruth(
        pattern_type=args.pattern,
        params={
            "num_agents": args.num_agents,
            f"{args.pattern}_strength": 0.7
        }
    )

    generator = SyntheticVotingDataGenerator(seed=args.seed)
    generator.set_noise_level(args.noise)

    print(f"Generating synthetic data with {args.num_agents} agents, {args.num_rounds} rounds...")
    print(f"Pattern: {args.pattern}")
    print(f"Noise level: {args.noise}")

    data = generator.generate_synthetic_voting_data(
        num_agents=args.num_agents,
        num_rounds=args.num_rounds,
        ground_truth=ground_truth
    )

    generator.export_to_csv(data, str(Path(args.output) / f"{args.pattern}_data.csv"))
    generator.export_to_json(data, ground_truth, str(Path(args.output) / f"{args.pattern}_data.json"))

    print(f"\nData saved to {args.output}/")
    print(f"Total votes: {len(data)}")
    print(f"Unique agents: {len(set(d['agent_id'] for d in data))}")
    print(f"Unique rounds: {len(set(d['round'] for d in data))}")

    abstentions = sum(1 for d in data if d["abstention"])
    print(f"Abstention rate: {abstentions / len(data):.2%}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
