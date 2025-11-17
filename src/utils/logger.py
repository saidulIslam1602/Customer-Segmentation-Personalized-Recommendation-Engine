"""
Centralized Logging Configuration

Provides structured logging with proper formatting, file rotation,
and configurable log levels. Replaces print statements throughout the codebase.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color support for console output.
    Colors are only applied when outputting to a terminal.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors if outputting to terminal"""
        if sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "enterprise_bi",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        app_name: Application name for log files
        console_output: Enable console output
        file_output: Enable file output
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Define log format
    detailed_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_format = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_format)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        # Main log file
        log_file = log_path / f"{app_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_format)
        logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_path / f"{app_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_format)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter:
    """
    Adapter to gradually replace print statements with logging.
    Can be used as a drop-in replacement for print while transitioning code.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize logger adapter"""
        self.logger = logger or get_logger(__name__)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False):
        """Log critical message"""
        self.logger.critical(message, exc_info=exc_info)
    
    def print(self, *args, **kwargs):
        """Print-like interface that logs to INFO level"""
        message = ' '.join(str(arg) for arg in args)
        self.logger.info(message)


# Module-level logger instance
_default_logger = None


def init_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Initialize the default logger for the application.
    Should be called once at application startup.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
    """
    global _default_logger
    _default_logger = setup_logging(log_level=log_level, log_dir=log_dir)
    return _default_logger


def log_function_call(func):
    """
    Decorator to log function calls with arguments and execution time.
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed_time:.2f}s")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed_time:.2f}s: {e}", exc_info=True)
            raise
    
    return wrapper


def log_exception(logger: Optional[logging.Logger] = None):
    """
    Decorator to log exceptions from functions.
    
    Usage:
        @log_exception()
        def my_function():
            raise ValueError("Something went wrong")
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


# Convenience functions
def log_info(message: str, logger_name: Optional[str] = None):
    """Log info message"""
    logger = get_logger(logger_name or __name__)
    logger.info(message)


def log_error(message: str, logger_name: Optional[str] = None, exc_info: bool = True):
    """Log error message"""
    logger = get_logger(logger_name or __name__)
    logger.error(message, exc_info=exc_info)


def log_warning(message: str, logger_name: Optional[str] = None):
    """Log warning message"""
    logger = get_logger(logger_name or __name__)
    logger.warning(message)


def log_debug(message: str, logger_name: Optional[str] = None):
    """Log debug message"""
    logger = get_logger(logger_name or __name__)
    logger.debug(message)
