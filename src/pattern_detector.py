"""Pattern detector for abstention analysis."""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
from collections import Counter


@dataclass
class PatternDetectionResult:
    """Result of pattern detection."""

    pattern_type: str
    score: float
    confidence: float
    details: Dict[str, Any]


class PatternDetector:
    """
    Analyzes abstention patterns to infer group dynamics.

    Detects implicit consensus, minority influence, and oppression patterns
    from voting data.
    """

    def __init__(self) -> None:
        """Initialize the pattern detector."""
        self._history: List[Dict[str, Any]] = []

    def detect_consensus(
        self,
        voting_data: Dict[str, Any]
    ) -> PatternDetectionResult:
        """
        Detect implicit consensus patterns in voting data.

        Consensus is indicated by:
        - Low abstention rate across the group
        - Consistent winners across multiple rounds
        - Low belief variance among voters

        Args:
            voting_data: Dictionary containing voting round data

        Returns:
            PatternDetectionResult with consensus score and confidence
        """
        rounds = voting_data.get("rounds", [])
        if not rounds:
            return PatternDetectionResult(
                pattern_type="consensus",
                score=0.0,
                confidence=0.0,
                details={}
            )

        # Calculate abstention rate consistency
        abstention_rates = [r.get("abstention_rate", 0) for r in rounds]
        avg_abstention = sum(abstention_rates) / len(abstention_rates)

        # Low abstention suggests consensus
        abstention_score = 1 - avg_abstention

        # Calculate winner consistency
        winners = [r.get("winner") for r in rounds if r.get("winner") is not None]
        if winners:
            winner_counts = Counter(winners)
            majority_winner = max(winner_counts.values())
            winner_consistency = majority_winner / len(winners)
        else:
            winner_consistency = 0

        # Calculate belief consistency (if available)
        belief_variances = [
            r.get("belief_variance", 0)
            for r in rounds
            if r.get("belief_variance") is not None
        ]
        if belief_variances:
            avg_variance = sum(belief_variances) / len(belief_variances)
            # Lower variance = higher consensus
            belief_score = max(0, 1 - avg_variance * 2)
        else:
            belief_score = 0.5  # Default uncertainty

        # Combine scores
        score = (
            0.4 * abstention_score +
            0.4 * winner_consistency +
            0.2 * belief_score
        )

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(
            num_rounds=len(rounds),
            abstention_variability=self._calculate_variance(abstention_rates),
            winner_consistency=winner_consistency
        )

        return PatternDetectionResult(
            pattern_type="consensus",
            score=score,
            confidence=confidence,
            details={
                "avg_abstention_rate": avg_abstention,
                "winner_consistency": winner_consistency,
                "belief_variance": belief_variances[-1] if belief_variances else None,
                "num_rounds_analyzed": len(rounds)
            }
        )

    def detect_minority_pressure(
        self,
        voting_data: Dict[str, Any]
    ) -> PatternDetectionResult:
        """
        Detect minority influence patterns.

        Minority influence is indicated by:
        - Increasing abstention rate as minority becomes more active
        - Shifts in winning candidates matching minority preferences
        - Polarization in voting patterns

        Args:
            voting_data: Dictionary containing voting round data

        Returns:
            PatternDetectionResult with pressure indicator and confidence
        """
        rounds = voting_data.get("rounds", [])
        if len(rounds) < 3:
            return PatternDetectionResult(
                pattern_type="minority_pressure",
                score=0.0,
                confidence=0.0,
                details={"reason": "insufficient_data"}
            )

        # Analyze abstention trend
        abstention_rates = [r.get("abstention_rate", 0) for r in rounds]

        # Detect increasing abstention trend
        if len(abstention_rates) >= 3:
            first_half = abstention_rates[:len(abstention_rates)//2]
            second_half = abstention_rates[len(abstention_rates)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            abstention_increase = second_avg - first_avg
        else:
            abstention_increase = 0

        # Analyze vote distribution changes
        winner_changes = self._count_winner_changes(rounds)

        # Detect polarization (increasing variance in outcomes)
        vote_counts = [
            r.get("vote_distribution", {})
            for r in rounds
            if r.get("vote_distribution")
        ]

        if vote_counts:
            vote_variances = [
                self._calculate_distribution_variance(vc)
                for vc in vote_counts
            ]
            variance_trend = sum(vote_variances) / len(vote_variances)
        else:
            variance_trend = 0

        # Calculate minority pressure score
        # Higher score if: abstention increases, vote variance increases
        score = min(1.0, 0.5 * abstention_increase + 0.5 * variance_trend)

        # Confidence based on data patterns
        confidence = self._calculate_confidence(
            num_rounds=len(rounds),
            abstention_variability=self._calculate_variance(abstention_rates),
            winner_consistency=1 - min(1, winner_changes / max(len(rounds) - 1, 1))
        )

        return PatternDetectionResult(
            pattern_type="minority_pressure",
            score=score,
            confidence=confidence,
            details={
                "abstention_increase": abstention_increase,
                "winner_changes": winner_changes,
                "vote_variance_trend": variance_trend,
                "num_rounds_analyzed": len(rounds)
            }
        )

    def detect_oppression(
        self,
        voting_data: Dict[str, Any]
    ) -> PatternDetectionResult:
        """
        Detect oppression patterns in voting data.

        Oppression is indicated by:
        - High abstention rate that doesn't match belief distribution
        -Dominant candidate that nobody truly prefers
        - Abstention correlates with belief deviation

        Args:
            voting_data: Dictionary containing voting round data

        Returns:
            PatternDetectionResult with oppression score and confidence
        """
        rounds = voting_data.get("rounds", [])
        agents_data = voting_data.get("agents", [])
        if not rounds:
            return PatternDetectionResult(
                pattern_type="oppression",
                score=0.0,
                confidence=0.0,
                details={}
            )

        abstention_rates = [r.get("abstention_rate", 0) for r in rounds]
        avg_abstention = sum(abstention_rates) / len(abstention_rates)

        # Check for belief-abstention correlation (if data available)
        if agents_data:
            belief_abstention_corr = self._calculate_belief_abstention_correlation(
                agents_data, abstention_rates
            )
        else:
            belief_abstention_corr = 0

        # Check for dominant but unpopular winner
        winner_scores = [
            self._calculate_winner_unpopularity(r)
            for r in rounds
        ]
        avg_unpopularity = sum(winner_scores) / len(winner_scores) if winner_scores else 0

        # Calculate oppression score
        # High score if: high abstention + belief-abstention correlation + unpopular winner
        score = (
            0.3 * avg_abstention +
            0.4 * abs(belief_abstention_corr) +
            0.3 * avg_unpopularity
        )

        # Confidence based on data quality and signal strength
        confidence = self._calculate_confidence(
            num_rounds=len(rounds),
            abstention_variability=self._calculate_variance(abstention_rates),
            winner_consistency=1 - avg_unpopularity
        )

        return PatternDetectionResult(
            pattern_type="oppression",
            score=score,
            confidence=confidence,
            details={
                "avg_abstention_rate": avg_abstention,
                "belief_abstention_correlation": belief_abstention_corr,
                "avg_winner_unpopularity": avg_unpopularity,
                "num_rounds_analyzed": len(rounds)
            }
        )

    def detect_group_pressure(
        self,
        voting_data: Dict[str, Any]
    ) -> PatternDetectionResult:
        """
        Detect general group pressure patterns.

        Group pressure combines elements of consensus and oppression:
        - Strong social influence on voting behavior
        - Abstention as form of rebellion or submission
        - Synchrony in voting patterns

        Args:
            voting_data: Dictionary containing voting round data

        Returns:
            PatternDetectionResult with pressure score and confidence
        """
        rounds = voting_data.get("rounds", [])
        if len(rounds) < 2:
            return PatternDetectionResult(
                pattern_type="group_pressure",
                score=0.0,
                confidence=0.0,
                details={}
            )

        # Calculate voting synchrony
        abstention_rates = [r.get("abstention_rate", 0) for r in rounds]

        # Calculate round-to-round similarity
        synchrony = self._calculate_synchrony(abstention_rates)

        # Calculate response to outliers
        outlier_response = self._detect_outlier_response(abstention_rates)

        # Combine for group pressure score
        score = (
            0.5 * synchrony +
            0.5 * (1 - outlier_response)  # Low outlier response = pressure
        )

        confidence = self._calculate_confidence(
            num_rounds=len(rounds),
            abstention_variability=self._calculate_variance(abstention_rates),
            winner_consistency=synchrony
        )

        return PatternDetectionResult(
            pattern_type="group_pressure",
            score=score,
            confidence=confidence,
            details={
                "synchrony_score": synchrony,
                "outlier_response": outlier_response,
                "abstention_std": math.sqrt(self._calculate_variance(abstention_rates)),
                "num_rounds_analyzed": len(rounds)
            }
        )

    def detect_consensus_change(
        self,
        voting_data: Dict[str, Any]
    ) -> PatternDetectionResult:
        """
        Detect changes in consensus over time.

        Useful for identifying when group dynamics shift.

        Args:
            voting_data: Dictionary containing voting round data

        Returns:
            PatternDetectionResult with change indicator
        """
        rounds = voting_data.get("rounds", [])
        if len(rounds) < 4:
            return PatternDetectionResult(
                pattern_type="consensus_change",
                score=0.0,
                confidence=0.0,
                details={"reason": "insufficient_data"}
            )

        # Split into quadrants
        n = len(rounds)
        q1 = rounds[:n//4]
        q2 = rounds[n//4:2*n//4]
        q3 = rounds[2*n//4:3*n//4]
        q4 = rounds[3*n//4:]

        quartiles = [q1, q2, q3, q4]

        # Calculate consensus for each quartile
        consensus_scores = []
        for quartile in quartiles:
            if not quartile:
                continue
            abstention_rates = [r.get("abstention_rate", 0) for r in quartile]
            avg_abstention = sum(abstention_rates) / len(abstention_rates)
            consensus_scores.append(1 - avg_abstention)

        # Detect significant change
        if len(consensus_scores) >= 2:
            first_avg = sum(consensus_scores[:len(consensus_scores)//2]) / 2
            last_avg = sum(consensus_scores[len(consensus_scores)//2:]) / 2
            change = abs(last_avg - first_avg)
        else:
            change = 0

        # Determine direction
        if len(consensus_scores) >= 2:
            direction = "increasing" if consensus_scores[-1] > consensus_scores[0] else "decreasing"
        else:
            direction = "unknown"

        confidence = self._calculate_confidence(
            num_rounds=len(rounds),
            abstention_variability=self._calculate_variance(
                [r.get("abstention_rate", 0) for r in rounds]
            ),
            winner_consistency=1 - change
        )

        return PatternDetectionResult(
            pattern_type="consensus_change",
            score=change,
            confidence=confidence,
            details={
                "consensus_trend": direction,
                "consensus_change": change,
                "quartile_consensus": consensus_scores,
                "num_rounds_analyzed": len(rounds)
            }
        )

    def analyze_all_patterns(
        self,
        voting_data: Dict[str, Any]
    ) -> Dict[str, PatternDetectionResult]:
        """
        Run all pattern detection methods.

        Args:
            voting_data: Dictionary containing voting round data

        Returns:
            Dictionary mapping pattern types to results
        """
        return {
            "consensus": self.detect_consensus(voting_data),
            "minority_pressure": self.detect_minority_pressure(voting_data),
            "oppression": self.detect_oppression(voting_data),
            "group_pressure": self.detect_group_pressure(voting_data),
            "consensus_change": self.detect_consensus_change(voting_data)
        }

    def _calculate_confidence(
        self,
        num_rounds: int,
        abstention_variability: float,
        winner_consistency: float
    ) -> float:
        """
        Calculate detection confidence based on data quality.

        Args:
            num_rounds: Number of rounds analyzed
            abstention_variability: Variance in abstention rates
            winner_consistency: Consistency of winners

        Returns:
            Confidence score (0 to 1)
        """
        # More rounds = higher confidence (up to a point)
        rounds_factor = min(1.0, num_rounds / 10)

        # Lower variability = higher confidence
        variability_factor = 1 - min(1.0, abstention_variability * 2)

        # Lower consistency (for minority/oppression detection) = higher confidence
        consistency_factor = 1 - winner_consistency

        confidence = (
            0.4 * rounds_factor +
            0.3 * variability_factor +
            0.3 * consistency_factor
        )

        return max(0.0, min(1.0, confidence))

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _calculate_synchrony(self, values: List[float]) -> float:
        """Calculate how synchronized the values are."""
        if len(values) < 2:
            return 1.0

        # Calculate autocorrelation
        mean = sum(values) / len(values)
        variance = self._calculate_variance(values)

        if variance == 0:
            return 1.0

        # Simple autocorrelation at lag 1
        autocorr = 0
        for i in range(len(values) - 1):
            autocorr += (values[i] - mean) * (values[i + 1] - mean)
        autocorr /= (len(values) - 1) * variance

        # Convert to 0-1 range
        return (autocorr + 1) / 2

    def _detect_outlier_response(self, values: List[float]) -> float:
        """Detect how responses to outliers are handled."""
        if len(values) < 3:
            return 0.5

        mean = sum(values) / len(values)
        std = math.sqrt(self._calculate_variance(values) + 1e-10)

        # Count values outside 2 standard deviations
        outliers = sum(1 for v in values if abs(v - mean) > 2 * std)

        # If outliers are isolated (not clusters), it suggests individual response
        if outliers == 0:
            return 0.5  # No outliers - group is uniform

        # Check if outliers cluster together
        outlier_positions = [i for i, v in enumerate(values) if abs(v - mean) > 2 * std]
        if len(outlier_positions) >= 2:
            # Check if consecutive
            consecutive = sum(
                1 for i in range(len(outlier_positions) - 1)
                if outlier_positions[i + 1] - outlier_positions[i] == 1
            )
            if consecutive > 0:
                return 0.3  # Clustered outliers - systemic response
            return 0.6  # Scattered outliers - individual responses
        return 0.8  # Single outlier - individual response

    def _count_winner_changes(self, rounds: List[Dict]) -> int:
        """Count how many times the winning candidate changed."""
        winners = [r.get("winner") for r in rounds if r.get("winner") is not None]
        changes = 0
        for i in range(1, len(winners)):
            if winners[i] != winners[i - 1]:
                changes += 1
        return changes

    def _calculate_distribution_variance(self, distribution: Dict) -> float:
        """Calculate variance of a vote distribution."""
        if not distribution:
            return 0.0

        values = list(distribution.values())
        return self._calculate_variance(values)

    def _calculate_belief_abstention_correlation(
        self,
        agents_data: List[Dict],
        abstention_rates: List[float]
    ) -> float:
        """Calculate correlation between belief and abstention."""
        if not agents_data:
            return 0.0

        # Extract belief deviation and abstention for each agent
        belief_devs = [a.get("belief_deviation", 0) for a in agents_data]
        abstentions = [a.get("abstention", 0) for a in agents_data]

        if not belief_devs or not abstentions:
            return 0.0

        # Calculate Pearson correlation
        n = min(len(belief_devs), len(abstentions))
        if n < 2:
            return 0.0

        mean_belief = sum(belief_devs[:n]) / n
        mean_abstention = sum(abstentions[:n]) / n

        numerator = sum(
            (belief_devs[i] - mean_belief) * (abstentions[i] - mean_abstention)
            for i in range(n)
        )

        denom_belief = math.sqrt(sum((x - mean_belief) ** 2 for x in belief_devs[:n]))
        denom_abstention = math.sqrt(sum((x - mean_abstention) ** 2 for x in abstentions[:n]))

        if denom_belief * denom_abstention == 0:
            return 0.0

        return numerator / (denom_belief * denom_abstention)

    def _calculate_winner_unpopularity(self, round_data: Dict) -> float:
        """Calculate how unpopular the winning candidate is."""
        vote_dist = round_data.get("vote_distribution", {})
        if not vote_dist:
            return 0.0

        total_votes = sum(vote_dist.values())
        if total_votes == 0:
            return 0.0

        # Calculate what fraction of non-voters would be needed to change winner
        sorted_votes = sorted(vote_dist.values(), reverse=True)
        if len(sorted_votes) < 2:
            return 0.0

        winner_votes = sorted_votes[0]
        runnerup_votes = sorted_votes[1]

        # Unpopularity if winner has narrow margin
        total_possible = total_votes + round_data.get("abstention_count", 0)
        if total_possible == 0:
            return 0.0

        unpopularity = 1 - (winner_votes / total_possible)
        return unpopularity
