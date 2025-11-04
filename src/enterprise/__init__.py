"""
Enterprise Business Intelligence Platform Extensions
Enhances existing functionality with CRM integration, security, and enterprise features
"""

__version__ = "2.0.0"
__author__ = "Enterprise Data Science Team"

from .enterprise_platform_manager import EnterprisePlatformManager
from .crm_integration_layer import CRMIntegrationLayer
from .api_gateway import APIGateway
from .security_manager import SecurityManager
from .performance_monitor import EnterprisePerformanceMonitor

__all__ = [
    "EnterprisePlatformManager",
    "CRMIntegrationLayer",
    "APIGateway",
    "SecurityManager",
    "EnterprisePerformanceMonitor",
]
