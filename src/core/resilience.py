"""
Resilience utilities for ML Factory.

Provides timeout protection, circuit breakers, and retry logic
for fault-tolerant pipeline execution.

Key Features:
- Timeout protection: Prevent operations from running indefinitely
- Circuit breakers: Isolate failures, continue with healthy components
- Retry logic: Handle transient failures gracefully with exponential backoff

Example:
    from src.core.resilience import timeout, CircuitBreaker, CircuitOpenError, retry

    # Timeout protection
    @timeout(60)  # 1 minute timeout
    def slow_operation():
        ...

    # Circuit breaker for model isolation
    breaker = CircuitBreaker(name="xgboost_training")
    try:
        with breaker:
            train_xgboost()
    except CircuitOpenError:
        logger.warning("XGBoost circuit open, skipping")

    # Retry with exponential backoff
    @retry(max_retries=3, retryable_exceptions=(RuntimeError, IOError))
    def flaky_operation():
        ...
"""

from __future__ import annotations

import functools
import logging
import random
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, TypeVar

from src.core.exceptions import MLFactoryError

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ResilienceError(MLFactoryError):
    """
    Base class for resilience-related errors.

    All resilience errors (timeout, circuit breaker, retry exhaustion)
    inherit from this class, making it easy to catch any resilience
    mechanism failure.
    """

    pass


class ResilienceTimeoutError(ResilienceError):
    """
    Raised when an operation times out.

    Attributes:
        timeout_seconds: The timeout duration that was exceeded
    """

    def __init__(self, message: str, timeout_seconds: float | None = None) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class CircuitOpenError(ResilienceError):
    """
    Raised when circuit breaker is open and rejecting calls.

    Attributes:
        circuit_name: Name of the circuit that is open
    """

    def __init__(self, circuit_name: str) -> None:
        super().__init__(f"Circuit '{circuit_name}' is open - calls are being rejected")
        self.circuit_name = circuit_name


class RetryExhaustedError(ResilienceError):
    """
    Raised when all retry attempts have been exhausted.

    Attributes:
        attempts: Number of attempts made
        last_exception: The last exception that caused the retry
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


# Backward compatibility alias
TimeoutError = ResilienceTimeoutError


# =============================================================================
# TIMEOUT PROTECTION
# =============================================================================


def timeout(
    seconds: float, use_signals: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add timeout protection to a function.

    Prevents operations from running indefinitely by enforcing a maximum
    execution time. Useful for Optuna trials, model training, and other
    potentially long-running operations.

    Args:
        seconds: Maximum execution time in seconds
        use_signals: Use signal-based timeout (only works in main thread on Unix).
                    If False, uses thread-based timeout (works everywhere but
                    cannot interrupt blocking operations).

    Returns:
        Decorated function that raises TimeoutError if execution exceeds limit

    Example:
        @timeout(60)  # 1 minute timeout
        def train_model(X, y):
            # Training code here
            return model

        @timeout(300, use_signals=False)  # 5 minutes, thread-based
        def optuna_trial(trial):
            # Trial code here
            return score

    Note:
        - Signal-based timeouts only work on Unix and in the main thread
        - Thread-based timeouts cannot interrupt blocking C extensions
        - Consider using multiprocessing for guaranteed timeout behavior
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if use_signals:
                return _timeout_with_signal(func, seconds, *args, **kwargs)
            else:
                return _timeout_with_thread(func, seconds, *args, **kwargs)

        return wrapper

    return decorator


def _timeout_with_signal(  # noqa: UP047
    func: Callable[..., T], seconds: float, *args: Any, **kwargs: Any
) -> T:
    """
    Signal-based timeout implementation.

    Uses SIGALRM to interrupt execution. Only works on Unix systems
    and only in the main thread.

    Args:
        func: Function to execute
        seconds: Timeout in seconds
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func(*args, **kwargs)

    Raises:
        TimeoutError: If execution exceeds timeout
    """

    def handler(signum: int, frame: Any) -> None:
        raise ResilienceTimeoutError(
            f"Function '{func.__name__}' timed out after {seconds}s",
            timeout_seconds=seconds,
        )

    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))

    try:
        return func(*args, **kwargs)
    finally:
        # Restore original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _timeout_with_thread(  # noqa: UP047
    func: Callable[..., T], seconds: float, *args: Any, **kwargs: Any
) -> T:
    """
    Thread-based timeout implementation.

    Uses ThreadPoolExecutor to run function in separate thread with timeout.
    Works on all platforms but cannot interrupt blocking operations.

    Args:
        func: Function to execute
        seconds: Timeout in seconds
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func(*args, **kwargs)

    Raises:
        TimeoutError: If execution exceeds timeout
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=seconds)
        except FuturesTimeoutError as e:
            raise ResilienceTimeoutError(
                f"Function '{func.__name__}' timed out after {seconds}s",
                timeout_seconds=seconds,
            ) from e


def run_with_timeout(  # noqa: UP047
    func: Callable[..., T],
    timeout_seconds: float,
    *args: Any,
    use_signals: bool = False,
    **kwargs: Any,
) -> T:
    """
    Run a function with timeout protection (non-decorator version).

    Useful when you need to apply timeout to functions you don't control
    or when the timeout duration is dynamic.

    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time
        *args: Positional arguments to pass to func
        use_signals: Use signal-based timeout (default False for compatibility)
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func(*args, **kwargs)

    Raises:
        TimeoutError: If execution exceeds timeout

    Example:
        result = run_with_timeout(
            external_lib.slow_function,
            timeout_seconds=30,
            data=my_data,
        )
    """
    if use_signals:
        return _timeout_with_signal(func, timeout_seconds, *args, **kwargs)
    else:
        return _timeout_with_thread(func, timeout_seconds, *args, **kwargs)


# =============================================================================
# RETRY LOGIC
# =============================================================================


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (0 means no retries)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add randomness to prevent thundering herd problem
        retryable_exceptions: Tuple of exception types that trigger retry
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retry with exponential backoff.

    Automatically retries a function when specified exceptions occur,
    with increasing delays between attempts. Useful for handling
    transient failures like GPU OOM, network issues, or temporary
    resource unavailability.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        jitter: Add randomness to delays to prevent thundering herd (default: True)
        retryable_exceptions: Tuple of exception types that trigger retry

    Returns:
        Decorated function that implements retry logic

    Raises:
        RetryExhaustedError: When all retry attempts fail (wraps last exception)

    Example:
        @retry(max_retries=3, retryable_exceptions=(RuntimeError, IOError))
        def flaky_operation():
            # Code that might fail transiently
            ...

        @retry(max_retries=2, base_delay=5.0, retryable_exceptions=(RuntimeError,))
        def gpu_operation():
            # Code that might OOM
            ...

    Note:
        - Delay formula: min(base_delay * (exponential_base ** attempt), max_delay)
        - With jitter, actual delay is multiplied by random value in [0.5, 1.5)
        - First attempt is not counted as a retry (happens immediately)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # If this was our last attempt, raise RetryExhaustedError
                    if attempt == max_retries:
                        raise RetryExhaustedError(
                            f"Function '{func.__name__}' failed after "
                            f"{max_retries + 1} attempts",
                            attempts=max_retries + 1,
                            last_exception=e,
                        ) from e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= 0.5 + random.random()

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/"
                        f"{max_retries + 1}), retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


def retry_with_config(config: RetryConfig) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Create retry decorator from RetryConfig.

    Convenience function to apply retry decorator using a config object
    instead of individual parameters.

    Args:
        config: RetryConfig instance with retry parameters

    Returns:
        Retry decorator configured with given parameters

    Example:
        config = RetryConfig(max_retries=5, base_delay=2.0)

        @retry_with_config(config)
        def my_function():
            ...
    """
    return retry(
        max_retries=config.max_retries,
        base_delay=config.base_delay,
        exponential_base=config.exponential_base,
        max_delay=config.max_delay,
        jitter=config.jitter,
        retryable_exceptions=config.retryable_exceptions,
    )


# Predefined retry configs for common scenarios
GPU_OOM_RETRY = RetryConfig(
    max_retries=2,
    base_delay=5.0,
    max_delay=30.0,
    jitter=True,
    retryable_exceptions=(RuntimeError,),  # CUDA OOM raises RuntimeError
)

NETWORK_RETRY = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)

TRANSIENT_RETRY = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True,
    retryable_exceptions=(RuntimeError, IOError, OSError),
)


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation - calls allowed
    OPEN = "open"  # Failing - calls rejected
    HALF_OPEN = "half_open"  # Testing recovery - limited calls allowed


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        success_threshold: Successes needed to close from half-open
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
        }


class CircuitBreaker:
    """
    Circuit breaker for isolating failures in ML pipelines.

    Implements the circuit breaker pattern to prevent cascade failures
    when a component (e.g., a specific model) repeatedly fails. The circuit
    has three states:

    - CLOSED: Normal operation, calls pass through
    - OPEN: Component is failing, calls are rejected immediately
    - HALF_OPEN: Testing if component has recovered

    Usage:
        # Create breaker for a specific model
        breaker = CircuitBreaker(name="lstm_training")

        # Use as context manager
        try:
            with breaker:
                train_lstm()
        except CircuitOpenError:
            logger.warning("LSTM circuit open, using fallback")
            use_fallback_model()

        # Or use call() method
        try:
            result = breaker.call(train_lstm, X, y)
        except CircuitOpenError:
            result = fallback_result

    The circuit opens after `failure_threshold` consecutive failures,
    waits `recovery_timeout` seconds, then allows a test call (half-open).
    If the test succeeds `success_threshold` times, circuit closes.

    Attributes:
        name: Identifier for this circuit (for logging)
        config: CircuitBreakerConfig instance
        state: Current circuit state
        stats: CircuitBreakerStats for monitoring
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (used in logs/errors)
            config: Configuration, or None for defaults
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._lock = Lock()
        self._stats = CircuitBreakerStats()

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    def __enter__(self) -> CircuitBreaker:
        """Enter context - check if calls are allowed."""
        self._check_state()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context - record success or failure."""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure()
        return False  # Don't suppress exceptions

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception raised by func
        """
        with self:
            return func(*args, **kwargs)

    def _check_state(self) -> None:
        """Check if circuit allows calls, transition states if needed."""
        with self._lock:
            self._stats.total_calls += 1

            if self.state == CircuitState.CLOSED:
                return  # Calls allowed

            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    # Transition to half-open to test recovery
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    # Still open, reject call
                    self._stats.rejected_calls += 1
                    raise CircuitOpenError(self.name)

            # HALF_OPEN: Allow call through for testing

    def _record_success(self) -> None:
        """Record successful call."""
        with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    # Recovered, close circuit
                    self._reset()

            # Any success resets failure count
            self._failure_count = 0

    def _record_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            self._stats.failed_calls += 1
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            self._stats.last_failure_time = self._last_failure_time

            if self.state == CircuitState.HALF_OPEN:
                # Failed during recovery test, reopen
                self._transition_to(CircuitState.OPEN)
                self._success_count = 0
            elif self._failure_count >= self.config.failure_threshold:
                # Too many failures, open circuit
                self._transition_to(CircuitState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to test recovery."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self._stats.state_changes += 1
        logger.info(f"Circuit '{self.name}' state: {old_state.value} -> {new_state.value}")

    def _reset(self) -> None:
        """Reset circuit to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Circuit '{self.name}' has recovered")

    def force_open(self) -> None:
        """Force circuit to open state (for testing/emergency)."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._last_failure_time = datetime.now()

    def force_close(self) -> None:
        """Force circuit to closed state (for recovery/testing)."""
        with self._lock:
            self._reset()

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self.state == CircuitState.OPEN


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to circuit breakers for different components
    (models, stages, external services). Useful for monitoring and management.

    Example:
        registry = CircuitBreakerRegistry()

        # Get or create breaker for each model
        for model_name in models:
            breaker = registry.get_or_create(model_name)
            try:
                with breaker:
                    train_model(model_name)
            except CircuitOpenError:
                logger.warning(f"Skipping {model_name}, circuit open")

        # Check status of all circuits
        for name, breaker in registry.all_breakers():
            if breaker.is_open():
                logger.warning(f"Circuit {name} is open")
    """

    def __init__(self, default_config: CircuitBreakerConfig | None = None) -> None:
        """
        Initialize registry.

        Args:
            default_config: Default config for new breakers
        """
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()
        self._lock = Lock()

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """
        Get existing breaker or create new one.

        Args:
            name: Breaker name
            config: Config for new breaker (uses default if None)

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config or self._default_config,
                )
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get breaker by name, or None if not found."""
        return self._breakers.get(name)

    def all_breakers(self) -> list[tuple[str, CircuitBreaker]]:
        """Get all registered breakers as (name, breaker) pairs."""
        return list(self._breakers.items())

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
        return [name for name, breaker in self._breakers.items() if breaker.is_open()]

    def reset_all(self) -> None:
        """Force close all circuits."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_close()

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all breakers."""
        return {name: breaker.stats.to_dict() for name, breaker in self._breakers.items()}


# Global registry for model circuit breakers
_model_breaker_registry: CircuitBreakerRegistry | None = None


def get_model_breaker_registry() -> CircuitBreakerRegistry:
    """Get or create the global model circuit breaker registry."""
    global _model_breaker_registry
    if _model_breaker_registry is None:
        _model_breaker_registry = CircuitBreakerRegistry(
            default_config=CircuitBreakerConfig(
                failure_threshold=3,  # Open after 3 failures
                recovery_timeout=300.0,  # Wait 5 minutes before retry
                success_threshold=1,  # Close after 1 success
            )
        )
    return _model_breaker_registry


__all__ = [
    # Exceptions
    "ResilienceError",
    "ResilienceTimeoutError",
    "TimeoutError",  # Backward compatibility alias
    "CircuitOpenError",
    "RetryExhaustedError",
    # Timeout
    "timeout",
    "run_with_timeout",
    # Retry
    "RetryConfig",
    "retry",
    "retry_with_config",
    "GPU_OOM_RETRY",
    "NETWORK_RETRY",
    "TRANSIENT_RETRY",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_model_breaker_registry",
]
