"""Experiment results parser and report generator."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    pattern_type: str
    score: float
    confidence: float
    details: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create ExperimentResult from dictionary."""
        return cls(
            pattern_type=data.get("pattern_type", "unknown"),
            score=data.get("score", 0.0),
            confidence=data.get("confidence", 0.0),
            details=data.get("details", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "score": self.score,
            "confidence": self.confidence,
            "details": self.details
        }


class ResultsParser:
    """Parser for experiment results."""

    def __init__(self, results_dir: str) -> None:
        """
        Initialize the results parser.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self._cached_results: Dict[str, Any] = {}

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from a JSON file."""
        filepath = self.results_dir / filename
        if filepath not in self._cached_results:
            if filepath.exists():
                with open(filepath, "r") as f:
                    self._cached_results[filepath] = json.load(f)
            else:
                self._cached_results[filepath] = {}
        return self._cached_results[filepath]

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all results from the results directory."""
        all_results = {}

        for filepath in self.results_dir.glob("*.json"):
            filename = filepath.stem
            all_results[filename] = self.load_results(filepath.name)

        return all_results

    def aggregate_results(self, results_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate results from multiple experiment runs.

        Args:
            results_dir: Directory containing experiments (uses instance default if None)

        Returns:
            Aggregated results dictionary
        """
        results_dir_path = Path(results_dir or self.results_dir)

        aggregated = {
            "metadata": {
                "results_dir": str(results_dir_path),
                "num_experiments": 0,
                "num_iterations": 0
            },
            "experiments": {},
            "summary": {}
        }

        for experiment_dir in results_dir_path.iterdir():
            if experiment_dir.is_dir():
                experiment_name = experiment_dir.name
                experiment_results = self._aggregate_experiment(experiment_dir)
                aggregated["experiments"][experiment_name] = experiment_results
                aggregated["metadata"]["num_experiments"] += 1
                if "iterations" in experiment_results:
                    aggregated["metadata"]["num_iterations"] += len(experiment_results["iterations"])

        # Compute summary across all experiments
        aggregated["summary"] = self._compute_summary(aggregated["experiments"])

        return aggregated

    def _aggregate_experiment(self, experiment_dir: Path) -> Dict[str, Any]:
        """Aggregate results for a single experiment type."""
        experiment_data = {
            "iterations": [],
            "summary": {},
            "files": []
        }

        for json_file in experiment_dir.glob("*.json"):
            experiment_data["files"].append(json_file.name)
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                if "iterations" in data:
                    experiment_data["iterations"].extend(data["iterations"])

                if "summary" in data:
                    for key, value in data["summary"].items():
                        if key not in experiment_data["summary"]:
                            experiment_data["summary"][key] = []
                        if isinstance(value, list):
                            experiment_data["summary"][key].extend(value)
                        else:
                            experiment_data["summary"][key].append(value)
            except json.JSONDecodeError:
                continue

        # Compute final summary for this experiment
        if experiment_data["iterations"]:
            experiment_data["summary"] = self._compute_experiment_summary(
                experiment_data["iterations"]
            )

        return experiment_data

    def _compute_experiment_summary(self, iterations: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics for experiment iterations."""
        if not iterations:
            return {}

        # Get all numeric keys from iterations
        all_keys = set()
        for iteration in iterations:
            all_keys.update(iteration.keys())

        summary = {}
        for key in all_keys:
            values = []
            for iteration in iterations:
                if key in iteration and isinstance(iteration[key], (int, float)):
                    values.append(iteration[key])

            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = variance ** 0.5
                summary[key] = {
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return summary

    def _compute_summary(self, experiments: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute overall summary across all experiments."""
        overall_summary = {}

        for experiment_name, experiment_data in experiments.items():
            summary = experiment_data.get("summary", {})

            # Get the mean score if available
            if "detected_consensus_score" in summary:
                mean_score = summary["detected_consensus_score"].get("mean", 0)
                overall_summary[f"{experiment_name}_consensus_score"] = mean_score
            elif "detected_minority_pressure_score" in summary:
                mean_score = summary["detected_minority_pressure_score"].get("mean", 0)
                overall_summary[f"{experiment_name}_pressure_score"] = mean_score
            elif "detected_oppression_score" in summary:
                mean_score = summary["detected_oppression_score"].get("mean", 0)
                overall_summary[f"{experiment_name}_oppression_score"] = mean_score

            # Get abstention change if available
            if "sim_abstention_change" in summary:
                mean_change = summary["sim_abstention_change"].get("mean", 0)
                overall_summary[f"{experiment_name}_abstention_change"] = mean_change

        return overall_summary


class ReportGenerator:
    """Generator for experiment reports."""

    def __init__(self, results_dir: str) -> None:
        """
        Initialize the report generator.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.parser = ResultsParser(str(results_dir))

    def generate_summary_report(self, aggregated_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate reports in multiple formats.

        Args:
            aggregated_results: Aggregated results from parser

        Returns:
            Dictionary mapping format to file path
        """
        report_paths = {}

        # Generate JSON report
        json_path = self.results_dir / "summary_report.json"
        with open(json_path, "w") as f:
            json.dump(aggregated_results, f, indent=2)
        report_paths["json"] = str(json_path)

        # Generate Markdown report
        md_path = self.results_dir / "summary_report.md"
        md_content = self._generate_markdown_report(aggregated_results)
        with open(md_path, "w") as f:
            f.write(md_content)
        report_paths["markdown"] = str(md_path)

        # Generate text report
        txt_path = self.results_dir / "summary_report.txt"
        txt_content = self._generate_text_report(aggregated_results)
        with open(txt_path, "w") as f:
            f.write(txt_content)
        report_paths["text"] = str(txt_path)

        return report_paths

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown formatted report."""
        lines = []
        lines.append("# Experiment Summary Report\n")

        # Metadata
        metadata = results.get("metadata", {})
        lines.append("## Metadata\n")
        lines.append(f"- **Results Directory**: {metadata.get('results_dir', 'N/A')}")
        lines.append(f"- **Number of Experiments**: {metadata.get('num_experiments', 0)}")
        lines.append(f"- **Total Iterations**: {metadata.get('num_iterations', 0)}")
        lines.append("")

        # Summary
        summary = results.get("summary", {})
        if summary:
            lines.append("## Summary\n")
            lines.append("| Experiment | Key Metric | Value |")
            lines.append("|------------|------------|-------|")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    lines.append(f"| {key} | Value | {value:.4f} |")
            lines.append("")

        # Experiments
        experiments = results.get("experiments", {})
        if experiments:
            lines.append("## Experiment Details\n")

            for experiment_name, experiment_data in experiments.items():
                lines.append(f"### {experiment_name.title()}\n")

                summary = experiment_data.get("summary", {})
                if summary:
                    lines.append("#### Summary Statistics\n")
                    for key, stats in summary.items():
                        if isinstance(stats, dict) and "mean" in stats:
                            lines.append(f"- **{key}**:")
                            lines.append(f"  - Mean: {stats['mean']:.4f}")
                            lines.append(f"  - Std: {stats['std']:.4f}")
                            lines.append(f"  - Min: {stats['min']:.4f}")
                            lines.append(f"  - Max: {stats['max']:.4f}")
                            lines.append(f"  - Count: {stats['count']}")
                    lines.append("")

                if experiment_data.get("iterations"):
                    lines.append(f"#### Sample Results (First 5 Iterations)\n")
                    lines.append("| Iteration | Metric | Value |")
                    lines.append("|-----------|--------|-------|")
                    for i, iteration in enumerate(experiment_data["iterations"][:5]):
                        lines.append(f"| {i+1} | {list(iteration.keys())[0]} | {list(iteration.values())[0]:.4f} |")
                    lines.append("")

        return "\n".join(lines)

    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text formatted report."""
        lines = []
        lines.append("=" * 60)
        lines.append("EXPERIMENT SUMMARY REPORT")
        lines.append("=" * 60)
        lines.append("")

        metadata = results.get("metadata", {})
        lines.append(f"Results Directory: {metadata.get('results_dir', 'N/A')}")
        lines.append(f"Number of Experiments: {metadata.get('num_experiments', 0)}")
        lines.append(f"Total Iterations: {metadata.get('num_iterations', 0)}")
        lines.append("")

        summary = results.get("summary", {})
        if summary:
            lines.append("-" * 40)
            lines.append("OVERALL SUMMARY")
            lines.append("-" * 40)
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{key}: {value:.4f}")
            lines.append("")

        experiments = results.get("experiments", {})
        for experiment_name, experiment_data in experiments.items():
            lines.append("-" * 40)
            lines.append(f"{experiment_name.upper()}")
            lines.append("-" * 40)

            summary = experiment_data.get("summary", {})
            for key, stats in summary.items():
                if isinstance(stats, dict) and "mean" in stats:
                    lines.append(f"{key}:")
                    lines.append(f"  Mean: {stats['mean']:.4f} (+/- {stats['std']:.4f})")
                    lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    lines.append(f"  Count: {stats['count']}")
            lines.append("")

        return "\n".join(lines)

    def generateDetailedReport(self, results_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a detailed report from raw results.

        Args:
            results_dir: Directory containing experiment results

        Returns:
            Dictionary containing detailed analysis
        """
        aggregated = self.parser.aggregate_results(results_dir)
        report = {
            "overview": {
                "total_experiments": aggregated["metadata"]["num_experiments"],
                "total_iterations": aggregated["metadata"]["num_iterations"]
            },
            "findings": [],
            "recommendations": []
        }

        # Analyze each experiment
        for experiment_name, experiment_data in aggregated["experiments"].items():
            summary = experiment_data.get("summary", {})

            # Detect patterns
            if "mean_detected_consensus_score" in summary:
                mean_consensus = summary["mean_detected_consensus_score"]["mean"]
                if mean_consensus > 0.7:
                    report["findings"].append({
                        "experiment": experiment_name,
                        "finding": "High consensus detected (score > 0.7)",
                        "confidence": summary["mean_detected_consensus_score"]["mean"]
                    })
                elif mean_consensus < 0.3:
                    report["findings"].append({
                        "experiment": experiment_name,
                        "finding": "Low consensus detected (score < 0.3)",
                        "confidence": summary["mean_detected_consensus_score"]["mean"]
                    })

            # Generate recommendations
            if "mean_sim_abstention_change" in summary:
                abstention_change = summary["mean_sim_abstention_change"]["mean"]
                if abstention_change > 0.1:
                    report["recommendations"].append({
                        "experiment": experiment_name,
                        "recommendation": "Investigate increasing abstention rates",
                        "rationale": f"Abstention increased by {abstention_change:.1%}"
                    })

        return report


def generate_report(input_results: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Generate reports from experiment results.

    Args:
        input_results: Path to results directory or JSON file
        output_dir: Output directory for reports

    Returns:
        Dictionary mapping format to output file path
    """
    input_path = Path(input_results)

    if output_dir is None:
        output_dir = input_results if input_path.is_dir() else str(input_path.parent)
    else:
        output_dir = str(output_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = ReportGenerator(input_results)

    # Aggregate results
    parser = ResultsParser(input_results)
    aggregated = parser.aggregate_results(input_results if input_path.is_dir() else None)

    # Generate reports
    report_paths = generator.generate_summary_report(aggregated)

    # Add detailed findings
    detailed = generator.generateDetailedReport(input_results if input_path.is_dir() else None)

    detailed_path = output_path / "detailed_findings.json"
    with open(detailed_path, "w") as f:
        json.dump(detailed, f, indent=2)
    report_paths["detailed_findings"] = str(detailed_path)

    return report_paths


def main():
    """Main entry point for report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment reports")
    parser.add_argument("--input", "-i", required=True, help="Input results directory or file")
    parser.add_argument("--output", "-o", help="Output directory (default: input directory)")
    parser.add_argument("--format", "-f", choices=["json", "markdown", "text"], default="all",
                       help="Output format (default: all)")

    args = parser.parse_args()

    print(f"Generating reports from {args.input}...")

    try:
        report_paths = generate_report(args.input, args.output)

        print("\nReports generated:")
        for format_name, path in report_paths.items():
            print(f"  {format_name}: {path}")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
