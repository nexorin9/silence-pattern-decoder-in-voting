"""Analysis utilities for data processing."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import csv
from pathlib import Path
import math


@dataclass
class VotingDataset:
    """A voting dataset with metadata."""

    voters: List[Dict[str, Any]]
    rounds: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def compute_abstention_rate(votes_list: List[Dict]) -> float:
    """
    Compute overall abstention percentage.

    Args:
        votes_list: List of vote records, each containing 'abstention' field

    Returns:
        Abstention rate as fraction (0 to 1)
    """
    if not votes_list:
        return 0.0

    abstentions = sum(1 for v in votes_list if v.get("abstention", False))
    return abstentions / len(votes_list)


def compute_group_abstention(
    votes_list: List[Dict],
    group_ids: List[Any]
) -> Dict[Any, float]:
    """
    Compute abstention rate per group.

    Args:
        votes_list: List of vote records with 'group_id' field
        group_ids: List of group identifiers to compute for

    Returns:
        Dictionary mapping group_id to abstention rate
    """
    if not votes_list:
        return {gid: 0.0 for gid in group_ids}

    # Group votes by group_id
    group_votes: Dict[Any, List[Dict]] = {gid: [] for gid in group_ids}
    for vote in votes_list:
        gid = vote.get("group_id")
        if gid in group_votes:
            group_votes[gid].append(vote)

    # Compute rate per group
    return {
        gid: compute_abstention_rate(votes)
        for gid, votes in group_votes.items()
    }


def compute_belief_voting_correlation(
    votes_list: List[Dict],
    agents: List[Dict]
) -> float:
    """
    Compute correlation between belief and voting behavior.

    Args:
        votes_list: List of vote records
        agents: List of agent records with 'belief' and 'agent_id'

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if not votes_list or not agents:
        return 0.0

    # Build agent lookup
    agent_lookup = {a.get("agent_id"): a for a in agents}

    # Get belief variance and voting for each agent
    belief_variances = []
    voting_binary = []

    for vote in votes_list:
        agent_id = vote.get("agent_id")
        agent = agent_lookup.get(agent_id)

        if agent:
            belief = agent.get("belief", [])
            if belief:
                # Compute belief variance
                mean_belief = sum(belief) / len(belief)
                variance = sum((b - mean_belief) ** 2 for b in belief) / len(belief)
                belief_variances.append(variance)

                # Binary voting (1 = abstain, 0 = vote)
                voting_binary.append(1 if vote.get("abstention", False) else 0)

    if len(belief_variances) < 2:
        return 0.0

    return _pearson_correlation(belief_variances, voting_binary)


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x * denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def load_voting_data(filepath: str) -> Dict[str, Any]:
    """
    Load voting data from file.

    Supports JSON and CSV formats.

    Args:
        filepath: Path to data file

    Returns:
        Dictionary with voting data
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith(".json"):
        return _load_json_data(filepath)
    elif filepath.endswith(".csv"):
        return _load_csv_data(filepath)
    else:
        # Try JSON first
        try:
            return _load_json_data(filepath)
        except:
            return _load_csv_data(filepath)


def _load_json_data(filepath: str) -> Dict[str, Any]:
    """Load voting data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def _load_csv_data(filepath: str) -> Dict[str, Any]:
    """Load voting data from CSV file."""
    data = {"votes": [], "metadata": {}}

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            converted_row = {}
            for key, value in row.items():
                try:
                    converted_row[key] = float(value)
                except (ValueError, TypeError):
                    converted_row[key] = value
            data["votes"].append(converted_row)

    return data


def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save analysis results to file.

    Automatically chooses format based on extension.

    Args:
        results: Results dictionary to save
        filename: Output filename
    """
    path = Path(filename)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    if filename.endswith(".json"):
        _save_json(results, filename)
    elif filename.endswith(".csv"):
        _save_csv(results, filename)
    elif filename.endswith(".md"):
        _save_markdown(results, filename)
    else:
        # Default to JSON
        _save_json(results, filename + ".json")


def _save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _save_csv(data: Dict[str, Any], filepath: str) -> None:
    """Save data to CSV file."""
    if not data:
        return

    # Handle nested data
    flat_data = _flatten_dict(data)
    if not flat_data:
        return

    fieldnames = list(flat_data[0].keys()) if flat_data else []

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> List[Dict]:
    """Flatten nested dictionary into list of flat dictionaries."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(_flatten_dict(item, f"{new_key}_{i}", sep))
                else:
                    items.append({f"{new_key}_{i}": item})
        else:
            items.append({new_key: v})
    return items


def _save_markdown(data: Dict[str, Any], filepath: str) -> None:
    """Save data to Markdown file."""
    with open(filepath, "w") as f:
        f.write("# Analysis Results\n\n")

        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                f.write(f"## {key}\n\n{value}\n\n")
            elif isinstance(value, list):
                f.write(f"## {key}\n\n")
                if value and isinstance(value[0], dict):
                    # Table format
                    headers = list(value[0].keys())
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("|" + " | ".join(["---"] * len(headers)) + "|\n")
                    for row in value:
                        row_values = [str(row.get(h, "")) for h in headers]
                        f.write("| " + " | ".join(row_values) + " |\n")
                else:
                    for item in value:
                        f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                f.write(f"## {key}\n\n")
                for subkey, subvalue in value.items():
                    f.write(f"- **{subkey}**: {subvalue}\n")
                f.write("\n")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(path, "r") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def aggregate_statistics(data_list: List[Dict], key: str) -> Dict[str, Any]:
    """
    Aggregate statistics for a specific key across multiple data points.

    Args:
        data_list: List of data dictionaries
        key: Key to extract and aggregate

    Returns:
        Dictionary with mean, std, min, max
    """
    values = [
        d.get(key) for d in data_list
        if d.get(key) is not None
    ]

    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)

    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "count": len(values)
    }


def compute_round_stats(round_data: List[Dict]) -> Dict[str, Any]:
    """
    Compute statistics for a list of voting rounds.

    Args:
        round_data: List of round dictionaries

    Returns:
        Dictionary with round statistics
    """
    if not round_data:
        return {}

    abstention_rates = [r.get("abstention_rate", 0) for r in round_data]
    belief_variances = [r.get("belief_variance", 0) for r in round_data if r.get("belief_variance") is not None]

    return {
        "num_rounds": len(round_data),
        "abstention": {
            "mean": sum(abstention_rates) / len(abstention_rates) if abstention_rates else 0,
            "std": math.sqrt(sum((r - sum(abstention_rates)/len(abstention_rates))**2 for r in abstention_rates) / len(abstention_rates)) if abstention_rates else 0,
            "min": min(abstention_rates) if abstention_rates else 0,
            "max": max(abstention_rates) if abstention_rates else 0
        },
        "belief_variance": {
            "mean": sum(belief_variances) / len(belief_variances) if belief_variances else 0,
            "std": math.sqrt(sum((v - sum(belief_variances)/len(belief_variances))**2 for v in belief_variances) / len(belief_variances)) if belief_variances else 0
        }
    }


def export_for_visualization(
    results: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """
    Export data for visualization.

    Creates separate files for different visualizations.

    Args:
        results: Analysis results
        output_dir: Directory to save visualization data

    Returns:
        Dictionary mapping plot type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_paths = {}

    # Export abstention timeline
    if "rounds" in results:
        abstention_data = [
            {"round": i, "abstention_rate": r.get("abstention_rate", 0)}
            for i, r in enumerate(results["rounds"])
        ]
        save_results(abstention_data, str(output_path / "abstention_timeline.csv"))
        file_paths["abstention_timeline"] = str(output_path / "abstention_timeline.csv")

    # Export belief distribution (if available)
    if "agents" in results:
        belief_data = [
            {"agent_id": a.get("agent_id"), "belief_mean": sum(a.get("belief", [])) / len(a.get("belief", [0]))}
            for a in results["agents"]
        ]
        save_results(belief_data, str(output_path / "belief_distribution.csv"))
        file_paths["belief_distribution"] = str(output_path / "belief_distribution.csv")

    # Export correlation data
    if "correlations" in results:
        save_results(results["correlations"], str(output_path / "correlations.json"))
        file_paths["correlations"] = str(output_path / "correlations.json")

    return file_paths
