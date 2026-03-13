"""
Parallel executor for running multiple simulations concurrently.
Uses multiprocessing to distribute simulation work across CPU cores.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class SimulationResult:
    """Result of a single simulation."""
    success: bool
    result: Any = None
    error: str = None
    params: Dict[str, Any] = field(default_factory=dict)


class ParallelExecutor:
    """
    Executor for running multiple simulations in parallel.

    Uses multiprocessing to distribute simulation work across CPU cores
    with progress tracking and error handling.
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_tqdm: bool = True,
        chunk_size: int = 1
    ):
        """
        Initialize the parallel executor.

        Args:
            num_workers: Number of worker processes (uses cpu_count() if None)
            use_tqdm: Whether to show progress bar
            chunk_size: Size of chunks for multiprocessing
        """
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.use_tqdm = use_tqdm
        self.chunk_size = chunk_size

        if self.use_tqdm and not HAS_TQDM:
            print("Warning: tqdm not installed, progress bar disabled")
            self.use_tqdm = False

    def run_batch(
        self,
        simulations: List[Dict[str, Any]],
        runner_fn: Callable[[Dict[str, Any]], Any],
        verbose: bool = False
    ) -> List[SimulationResult]:
        """
        Run multiple simulations in parallel.

        Args:
            simulations: List of simulation parameter dictionaries
            runner_fn: Function that runs a single simulation
            verbose: Print detailed information

        Returns:
            List of SimulationResult instances
        """
        total = len(simulations)
        results = []

        if verbose:
            print(f"Running {total} simulations with {self.num_workers} workers...")

        if total == 0:
            return results

        # Use multiprocessing Pool
        with Pool(processes=self.num_workers) as pool:
            # Create partial function with runner
            runner_with_params = partial(self._run_single, runner_fn=runner_fn)

            if self.use_tqdm and HAS_TQDM:
                # With tqdm progress bar
                for result in tqdm(
                    pool.imap(runner_with_params, simulations, chunksize=self.chunk_size),
                    total=total,
                    desc="Simulations"
                ):
                    results.append(result)
            else:
                # Without progress bar
                for result in pool.imap(runner_with_params, simulations, chunksize=self.chunk_size):
                    results.append(result)

        return results

    def _run_single(
        self,
        params: Dict[str, Any],
        runner_fn: Callable[[Dict[str, Any]], Any]
    ) -> SimulationResult:
        """
        Run a single simulation with error handling.

        Args:
            params: Simulation parameters
            runner_fn: Function that runs the simulation

        Returns:
            SimulationResult with success/error status
        """
        try:
            result = runner_fn(params)
            return SimulationResult(
                success=True,
                result=result,
                params=params
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if hasattr(e, '__traceback__'):
                error_msg += "\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__))
            return SimulationResult(
                success=False,
                error=error_msg,
                params=params
            )

    def run_with_callback(
        self,
        simulations: List[Dict[str, Any]],
        runner_fn: Callable[[Dict[str, Any]], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verbose: bool = False
    ) -> List[SimulationResult]:
        """
        Run simulations with progress callback.

        Args:
            simulations: List of simulation parameter dictionaries
            runner_fn: Function that runs a single simulation
            progress_callback: Function called with (completed, total)
            verbose: Print detailed information

        Returns:
            List of SimulationResult instances
        """
        total = len(simulations)
        results = []

        if verbose:
            print(f"Running {total} simulations with {self.num_workers} workers...")

        if total == 0:
            return results

        with Pool(processes=self.num_workers) as pool:
            runner_with_params = partial(self._run_single, runner_fn=runner_fn)

            # Use imap with chunksize for better progress tracking
            if self.use_tqdm and HAS_TQDM:
                iterator = tqdm(
                    pool.imap(runner_with_params, simulations, chunksize=self.chunk_size),
                    total=total,
                    desc="Simulations"
                )
            else:
                iterator = pool.imap(runner_with_params, simulations, chunksize=self.chunk_size)

            for idx, result in enumerate(iterator):
                results.append(result)
                if progress_callback:
                    progress_callback(idx + 1, total)

        return results


def run_simulation_batch(
    num_simulations: int,
    sim_params: Dict[str, Any],
    engine_fn: Callable[[Dict[str, Any]], Any],
    num_workers: Optional[int] = None,
    verbose: bool = False
) -> List[Any]:
    """
    Convenience function to run multiple simulations.

    Args:
        num_simulations: Number of simulations to run
        sim_params: Base parameters for each simulation
        engine_fn: Function that creates and runs a simulation
        num_workers: Number of worker processes
        verbose: Print progress

    Returns:
        List of simulation results
    """
    # Generate parameter sets for each simulation
    simulations = []
    for i in range(num_simulations):
        params = sim_params.copy()
        params['seed'] = sim_params.get('seed', 0) + i * 1000
        simulations.append(params)

    executor = ParallelExecutor(num_workers=num_workers, verbose=verbose)
    results = executor.run_batch(simulations, engine_fn, verbose=verbose)

    # Extract successful results
    return [r.result for r in results if r.success]


# Default global executor instance
_default_executor: Optional[ParallelExecutor] = None


def get_default_executor() -> ParallelExecutor:
    """Get or create the default parallel executor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = ParallelExecutor()
    return _default_executor
