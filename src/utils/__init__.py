"""
Utility modules for Enterprise Business Intelligence Platform

Provides configuration management, logging, validation, and error handling.
"""

from .config import config, Config
from .logger import setup_logging, get_logger, log_function_call, log_exception
from .data_validator import DataValidator, validate_data_files, ValidationResult
from .error_handler import (
    error_handler,
    retry_on_failure,
    graceful_shutdown,
    monitored_execution,
    safe_execute,
    BusinessException,
    DataValidationError,
    ModelTrainingError,
    CRMIntegrationError,
    ConfigurationError,
)

__all__ = [
    # Config
    "config",
    "Config",
    # Logging
    "setup_logging",
    "get_logger",
    "log_function_call",
    "log_exception",
    # Validation
    "DataValidator",
    "validate_data_files",
    "ValidationResult",
    # Error Handling
    "error_handler",
    "retry_on_failure",
    "graceful_shutdown",
    "monitored_execution",
    "safe_execute",
    "BusinessException",
    "DataValidationError",
    "ModelTrainingError",
    "CRMIntegrationError",
    "ConfigurationError",
]
