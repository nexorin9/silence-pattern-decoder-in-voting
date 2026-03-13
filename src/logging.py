"""
Structured logging system for experiment tracking and reproducibility.
"""

import logging
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass
class LogEvent:
    """A single log event with metadata."""
    timestamp: str
    run_id: str
    level: str
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "level": self.level,
            "event_type": self.event_type,
            "data": self.data,
            "message": self.message
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ExperimentLogger:
    """
    Logger for tracking experiments with structured data.

    Provides:
    - Run ID tracking for experiment reproducibility
    - Structured logging with event types
    - Event export to JSON/CSV for analysis
    - Level-based filtering
    """

    def __init__(
        self,
        name: str = "silence_decoder",
        run_id: Optional[str] = None,
        log_dir: Optional[Path] = None,
        level: LogLevel = LogLevel.INFO
    ):
        """
        Initialize the experiment logger.

        Args:
            name: Logger name
            run_id: Unique run identifier (generated if not provided)
            log_dir: Directory for log files
            level: Minimum log level
        """
        self.name = name
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.log_dir = log_dir or Path("logs")
        self.min_level = level

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize event log
        self._events: List[LogEvent] = []

        # Setup Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self._logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)

        # File handler for structured logs
        self._setup_file_handler()

    def _setup_file_handler(self):
        """Setup file handler for structured JSON logs."""
        log_file = self.log_dir / f"experiment_{self.run_id}.json"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self._logger.addHandler(file_handler)

    def _log_event(self, level: LogLevel, event_type: str, data: Dict[str, Any] = None, message: str = ""):
        """Internal method to create and store log events."""
        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            level=level.name,
            event_type=event_type,
            data=data or {},
            message=message
        )

        # Store in memory
        self._events.append(event)

        # Log to Python logger
        self._logger.log(level.value, json.dumps(event.to_dict()))

    def log_event(self, event_type: str, data: Dict[str, Any] = None, message: str = ""):
        """Log an INFO level event."""
        if self.min_level.value <= LogLevel.INFO.value:
            self._log_event(LogLevel.INFO, event_type, data, message)

    def log_debug(self, event_type: str, data: Dict[str, Any] = None, message: str = ""):
        """Log a DEBUG level event."""
        if self.min_level.value <= LogLevel.DEBUG.value:
            self._log_event(LogLevel.DEBUG, event_type, data, message)

    def log_warning(self, event_type: str, data: Dict[str, Any] = None, message: str = ""):
        """Log a WARNING level event."""
        if self.min_level.value <= LogLevel.WARNING.value:
            self._log_event(LogLevel.WARNING, event_type, data, message)

    def log_error(self, event_type: str, data: Dict[str, Any] = None, message: str = ""):
        """Log an ERROR level event."""
        if self.min_level.value <= LogLevel.ERROR.value:
            self._log_event(LogLevel.ERROR, event_type, data, message)

    def log_simulation_start(self, config: Dict[str, Any]):
        """Log simulation start event."""
        self.log_event(
            "simulation_start",
            {
                "num_agents": config.get("num_agents", 100),
                "num_rounds": config.get("num_rounds", 50),
                "num_candidates": config.get("num_candidates", 3),
                "seed": config.get("seed"),
                "voting_rule": config.get("voting_rule", "approval")
            },
            "Simulation started"
        )

    def log_simulation_round(self, round_num: int, abstention_rate: float, winner: str):
        """Log simulation round event."""
        self.log_event(
            "simulation_round",
            {
                "round": round_num,
                "abstention_rate": abstention_rate,
                "winner": winner
            },
            f"Round {round_num} completed"
        )

    def log_simulation_end(self, result_summary: Dict[str, Any]):
        """Log simulation end event."""
        self.log_event(
            "simulation_end",
            result_summary,
            "Simulation completed"
        )

    def log_metric(self, name: str, value: Any, metadata: Dict[str, Any] = None):
        """Log a metric value."""
        self.log_event(
            "metric",
            {
                "name": name,
                "value": value,
                **(metadata or {})
            },
            f"Metric '{name}' = {value}"
        )

    def log_error_event(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log an error event with full context."""
        self._logger.error(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "level": "ERROR",
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }))

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export log events to JSON file.

        Args:
            filepath: Output file path (auto-generated if not provided)

        Returns:
            File path
        """
        if filepath is None:
            filepath = str(self.log_dir / f"events_{self.run_id}.json")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        events_data = [event.to_dict() for event in self._events]

        with open(path, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "generated_at": datetime.now().isoformat(),
                "total_events": len(events_data),
                "events": events_data
            }, f, indent=2)

        return filepath

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """
        Export log events to CSV file.

        Args:
            filepath: Output file path (auto-generated if not provided)

        Returns:
            File path
        """
        if filepath is None:
            filepath = str(self.log_dir / f"events_{self.run_id}.csv")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        import csv

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "run_id", "level", "event_type",
                "message", "data"
            ])
            writer.writeheader()

            for event in self._events:
                row = event.to_dict()
                row["data"] = json.dumps(row["data"])  # Serialize nested data
                writer.writerow(row)

        return filepath

    def get_events_by_type(self, event_type: str) -> List[LogEvent]:
        """Get events filtered by type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_events_by_level(self, level: LogLevel) -> List[LogEvent]:
        """Get events filtered by level."""
        return [e for e in self._events if e.level == level.name]

    def get_summary(self) -> Dict[str, Any]:
        """Get event log summary."""
        if not self._events:
            return {"total_events": 0}

        levels = {}
        event_types = {}

        for event in self._events:
            levels[event.level] = levels.get(event.level, 0) + 1
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        return {
            "run_id": self.run_id,
            "total_events": len(self._events),
            "levels": levels,
            "event_types": event_types,
            "first_event": self._events[0].timestamp if self._events else None,
            "last_event": self._events[-1].timestamp if self._events else None
        }


# Global logger instance
_global_logger: Optional[ExperimentLogger] = None


def get_logger(
    name: str = "silence_decoder",
    run_id: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: LogLevel = LogLevel.INFO
) -> ExperimentLogger:
    """
    Get or create global logger instance.

    Args:
        name: Logger name
        run_id: Unique run identifier
        log_dir: Log directory path
        level: Minimum log level

    Returns:
        ExperimentLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(
            name=name,
            run_id=run_id,
            log_dir=Path(log_dir) if log_dir else None,
            level=level
        )
    return _global_logger


def create_context_logger(run_id: str, **kwargs) -> ExperimentLogger:
    """
    Create a new logger for a specific run context.

    Args:
        run_id: Unique identifier for the run
        **kwargs: Additional logger parameters

    Returns:
        New ExperimentLogger instance
    """
    return ExperimentLogger(run_id=run_id, **kwargs)
