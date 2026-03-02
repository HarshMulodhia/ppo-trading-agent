"""
Decorators Module

Useful decorators for common tasks.
"""

import functools
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


def timing(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between retries

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s"
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def cache(func: Callable) -> Callable:
    """
    Simple function caching decorator.

    Args:
        func: Function to cache

    Returns:
        Wrapped function
    """
    cache_dict = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))

        if key not in cache_dict:
            cache_dict[key] = func(*args, **kwargs)

        return cache_dict[key]

    wrapper.cache_clear = lambda: cache_dict.clear()
    return wrapper


def validate_types(**type_checks):
    """
    Decorator to validate argument types.

    Args:
        **type_checks: Argument name to type mapping

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check kwargs
            for arg_name, expected_type in type_checks.items():
                if arg_name in kwargs:
                    if not isinstance(kwargs[arg_name], expected_type):
                        raise TypeError(
                            f"{arg_name} must be {expected_type}, "
                            f"got {type(kwargs[arg_name])}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_exceptions(func: Callable) -> Callable:
    """
    Decorator to log exceptions.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise

    return wrapper


def deprecated(message: str = ""):
    """
    Decorator to mark function as deprecated.

    Args:
        message: Deprecation message

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning = f"{func.__name__} is deprecated"
            if message:
                warning += f": {message}"
            logger.warning(warning)
            return func(*args, **kwargs)

        return wrapper

    return decorator
