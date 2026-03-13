"""Generate synthetic benchmark datasets for validation."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import (
    SyntheticVotingDataGenerator,
    GroundTruth
)


def generate_dataset(output_dir: str, pattern_type: str, name: str, num_agents: int = 100, num_rounds: int = 50):
    """Generate a synthetic dataset with the given pattern type."""
    generator = SyntheticVotingDataGenerator(seed=42)

    # Setup ground truth based on pattern type
    if pattern_type == "consensus":
        ground_truth = GroundTruth(
            pattern_type="consensus",
            params={
                "num_agents": num_agents,
                "consensus_strength": 0.8
            }
        )
    elif pattern_type == "minority_influence":
        ground_truth = GroundTruth(
            pattern_type="minority_influence",
            params={
                "num_agents": num_agents,
                "minority_size": 20,
                "minority_influence": 0.7,
                "minority_alignment": 0.9
            }
        )
    elif pattern_type == "oppression":
        ground_truth = GroundTruth(
            pattern_type="oppression",
            params={
                "num_agents": num_agents,
                "oppressive_agents": [0],
                "oppressive_weight": 0.9,
                "compliance_rate": 0.8,
                "abstention_rate": 0.3
            }
        )
    else:  # random
        ground_truth = GroundTruth(
            pattern_type="random",
            params={"num_agents": num_agents}
        )

    # Generate data
    generator.set_noise_level(0.1)
    data = generator.generate_synthetic_voting_data(
        num_agents=num_agents,
        num_rounds=num_rounds,
        ground_truth=ground_truth
    )

    # Export to CSV
    csv_path = os.path.join(output_dir, f"synthetic_{name}.csv")
    generator.export_to_csv(data, csv_path)

    # Export to JSON with metadata
    json_path = os.path.join(output_dir, f"synthetic_{name}.json")
    generator.export_to_json(data, ground_truth, json_path)

    print(f"Generated {csv_path}")
    print(f"  - Agents: {num_agents}")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Pattern: {pattern_type}")
    print(f"  - Noise: 0.1")

    return csv_path


def main():
    """Generate all benchmark datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "external")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Benchmark Datasets")
    print("=" * 60)
    print()

    # Generate all datasets
    patterns = [
        ("consensus", "consensus", "Consensus pattern with 80% strength"),
        ("minority_influence", "minority", "Minority influence pattern"),
        ("oppression", "oppression", "Oppression pattern"),
        ("random", "random", "Random baseline"),
    ]

    for pattern_type, name, description in patterns:
        print(f"Generating {name} dataset ({description})...")
        generate_dataset(output_dir, pattern_type, name)
        print()

    print("=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
