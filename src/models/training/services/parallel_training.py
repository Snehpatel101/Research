"""
Parallel model training for faster experimentation.

Uses joblib for multiprocessing to train multiple models concurrently.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from joblib import Parallel, delayed

from .model_training import ModelTrainingRequest, ModelTrainingResult, ModelTrainingService

logger = logging.getLogger(__name__)


@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel training."""

    n_jobs: int = -1  # -1 = all CPUs
    verbose: int = 10  # Joblib verbosity
    backend: str = "loky"  # "loky", "multiprocessing", "threading"
    prefer: str | None = None  # "processes" or "threads"


class ParallelTrainingService:
    """
    Service for training multiple models in parallel.

    Uses joblib for multiprocessing. Each model is trained in a separate
    process to maximize CPU utilization.

    Example:
        service = ParallelTrainingService(n_jobs=4)
        results = service.train_models_parallel(training_requests)
    """

    def __init__(
        self,
        n_jobs: int = -1,
        verbose: int = 10,
        backend: str = "loky",
    ):
        """
        Args:
            n_jobs: Number of parallel jobs (-1 = all CPUs)
            verbose: Verbosity level for joblib progress
            backend: Joblib backend ("loky", "multiprocessing", "threading")
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend

    def train_models_parallel(
        self,
        training_requests: list[ModelTrainingRequest],
    ) -> list[ModelTrainingResult]:
        """
        Train multiple models in parallel.

        Args:
            training_requests: List of ModelTrainingRequest objects

        Returns:
            List of ModelTrainingResult objects (same order as requests)
        """
        if not training_requests:
            return []

        n_models = len(training_requests)
        logger.info(
            f"Training {n_models} models in parallel "
            f"(n_jobs={self.n_jobs}, backend={self.backend})"
        )

        start_time = time.time()

        # Train in parallel using joblib
        results = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            backend=self.backend,
        )(delayed(self._train_single)(request) for request in training_requests)

        total_time = time.time() - start_time
        logger.info(
            f"Parallel training complete: {n_models} models in {total_time:.1f}s "
            f"({total_time/n_models:.1f}s avg per model)"
        )

        return results

    @staticmethod
    def _train_single(request: ModelTrainingRequest) -> ModelTrainingResult:
        """
        Train a single model (called by joblib worker).

        This is a static method to ensure it's picklable for multiprocessing.
        """
        # Create a fresh service instance in the worker process
        service = ModelTrainingService()
        return service.train_model(request)

    def train_with_callback(
        self,
        training_requests: list[ModelTrainingRequest],
        on_complete: Callable[[ModelTrainingResult], None] | None = None,
    ) -> list[ModelTrainingResult]:
        """
        Train models in parallel with optional callback on completion.

        Args:
            training_requests: List of training requests
            on_complete: Optional callback called for each completed model

        Returns:
            List of ModelTrainingResult objects
        """
        results = self.train_models_parallel(training_requests)

        if on_complete:
            for result in results:
                on_complete(result)

        return results


def train_models_parallel(
    requests: list[ModelTrainingRequest],
    n_jobs: int = -1,
) -> list[ModelTrainingResult]:
    """
    Convenience function for parallel training.

    Args:
        requests: List of ModelTrainingRequest objects
        n_jobs: Number of parallel jobs

    Returns:
        List of ModelTrainingResult objects
    """
    service = ParallelTrainingService(n_jobs=n_jobs)
    return service.train_models_parallel(requests)
