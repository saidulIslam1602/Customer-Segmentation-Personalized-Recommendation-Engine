"""
Enterprise Business Intelligence Platform

A comprehensive customer analytics and CRM integration platform that combines
advanced machine learning with enterprise-grade features for retail business
intelligence.

Version: 2.0.0
Author: Enterprise Data Science Team
License: MIT

Modules:
    enterprise: Enterprise features including CRM integration, security, and monitoring
    models: Core machine learning models for customer analytics
    data: Data processing and loading utilities
    visualization: Dashboard and reporting components
    utils: Utility functions and helpers

Usage:
    from src.enterprise import EnterprisePlatformManager
    
    platform = EnterprisePlatformManager()
    results = await platform.run_enterprise_analysis()
"""

__version__ = "2.0.0"
__author__ = "Enterprise Data Science Team"
__license__ = "MIT"

# Import main components for easy access
from .enterprise.enterprise_platform_manager import EnterprisePlatformManager

__all__ = [
    "EnterprisePlatformManager",
]