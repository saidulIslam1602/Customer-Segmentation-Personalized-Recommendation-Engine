"""
Error Handling and Recovery Module

Provides comprehensive error handling, recovery strategies, and monitoring
for the enterprise BI platform.
"""

import functools
import traceback
import sys
from typing import Callable, Any, Optional, Type, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BusinessException(Exception):
    """Base exception for business logic errors"""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DataValidationError(BusinessException):
    """Raised when data validation fails"""
    pass


class ModelTrainingError(BusinessException):
    """Raised when model training fails"""
    pass


class CRMIntegrationError(BusinessException):
    """Raised when CRM integration fails"""
    pass


class ConfigurationError(BusinessException):
    """Raised when configuration is invalid"""
    pass


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    retry_count: int = 0,
    retry_delay: float = 1.0,
    suppress_errors: bool = False,
    **kwargs
) -> Tuple[bool, Any]:
    """
    Safely execute a function with error handling and retry logic.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        default_return: Default value to return on error
        retry_count: Number of retries on failure
        retry_delay: Delay between retries in seconds
        suppress_errors: If True, don't raise exceptions
        **kwargs: Keyword arguments for func
    
    Returns:
        Tuple of (success: bool, result: Any)
    """
    import time
    
    attempt = 0
    last_exception = None
    
    while attempt <= retry_count:
        try:
            result = func(*args, **kwargs)
            return True, result
        
        except Exception as e:
            last_exception = e
            attempt += 1
            
            if attempt <= retry_count:
                logger.warning(
                    f"Attempt {attempt}/{retry_count + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"All {retry_count + 1} attempts failed for {func.__name__}: {e}",
                    exc_info=True
                )
    
    if suppress_errors:
        return False, default_return
    else:
        raise last_exception


def error_handler(
    default_return: Any = None,
    error_types: Tuple[Type[Exception], ...] = (Exception,),
    log_error: bool = True,
    re_raise: bool = False
):
    """
    Decorator for error handling with custom error types and logging.
    
    Args:
        default_return: Value to return when exception occurs
        error_types: Tuple of exception types to catch
        log_error: Whether to log the error
        re_raise: Whether to re-raise the exception after handling
    
    Usage:
        @error_handler(default_return=[], error_types=(ValueError, TypeError))
        def process_data(data):
            return [x * 2 for x in data]
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        exc_info=True,
                        extra={
                            'function': func.__name__,
                            'args': args,
                            'kwargs': kwargs
                        }
                    )
                
                if re_raise:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    
    Usage:
        @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_data_from_api():
            return requests.get('https://api.example.com/data')
    """
    import time
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}",
                            exc_info=True
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def graceful_shutdown(func: Callable) -> Callable:
    """
    Decorator to handle graceful shutdown on KeyboardInterrupt.
    
    Usage:
        @graceful_shutdown
        def main():
            run_application()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal. Cleaning up...")
            # Perform cleanup here if needed
            sys.exit(0)
        except Exception as e:
            logger.critical(f"Critical error in {func.__name__}: {e}", exc_info=True)
            sys.exit(1)
    
    return wrapper


class ErrorMonitor:
    """
    Monitor and track errors across the application.
    Useful for debugging and identifying patterns in failures.
    """
    
    def __init__(self, max_errors: int = 100):
        """
        Initialize error monitor.
        
        Args:
            max_errors: Maximum number of errors to keep in memory
        """
        self.errors = []
        self.max_errors = max_errors
        self.error_counts = {}
    
    def record_error(
        self,
        error: Exception,
        context: str,
        severity: str = "ERROR",
        metadata: Optional[dict] = None
    ):
        """
        Record an error for monitoring.
        
        Args:
            error: The exception that occurred
            context: Context where error occurred
            severity: Error severity (ERROR, WARNING, CRITICAL)
            metadata: Additional metadata about the error
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'severity': severity,
            'traceback': traceback.format_exc(),
            'metadata': metadata or {}
        }
        
        self.errors.append(error_info)
        
        # Keep only most recent errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
        
        # Update error counts
        error_key = f"{context}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error
        logger.error(
            f"[{severity}] Error in {context}: {error}",
            extra=error_info
        )
    
    def get_error_summary(self) -> dict:
        """Get summary of errors"""
        return {
            'total_errors': len(self.errors),
            'error_counts': self.error_counts,
            'recent_errors': self.errors[-10:] if self.errors else []
        }
    
    def clear_errors(self):
        """Clear all recorded errors"""
        self.errors = []
        self.error_counts = {}


# Global error monitor instance
error_monitor = ErrorMonitor()


def monitored_execution(context: str, severity: str = "ERROR"):
    """
    Decorator to monitor function execution and record errors.
    
    Args:
        context: Context identifier for the function
        severity: Default severity level for errors
    
    Usage:
        @monitored_execution(context="data_processing")
        def process_customer_data(data):
            return transformed_data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_monitor.record_error(
                    error=e,
                    context=f"{context}.{func.__name__}",
                    severity=severity,
                    metadata={
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                raise
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.
    Stops calling a failing service after a threshold and retries after a timeout.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds before attempting to close circuit
            name: Name for this circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception if circuit is open or function fails
        """
        import time
        
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker {self.name}: Attempting to close (HALF_OPEN)")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - close circuit
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name}: Closed successfully")
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(
                    f"Circuit breaker {self.name}: OPENED after {self.failure_count} failures"
                )
            
            raise


def circuit_breaker_protected(name: str, failure_threshold: int = 5, timeout: float = 60.0):
    """
    Decorator to protect function with circuit breaker pattern.
    
    Args:
        name: Name for the circuit breaker
        failure_threshold: Number of failures before opening
        timeout: Timeout before retry in seconds
    
    Usage:
        @circuit_breaker_protected(name="external_api", failure_threshold=3)
        def call_external_api():
            return requests.get('https://api.example.com')
    """
    breaker = CircuitBreaker(failure_threshold=failure_threshold, timeout=timeout, name=name)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
