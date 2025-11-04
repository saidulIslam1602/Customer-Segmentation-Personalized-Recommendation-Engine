"""
Setup configuration for Enterprise Business Intelligence Platform

This setup script configures the installation of the Enterprise Business Intelligence
Platform, including all dependencies and optional components for development and
visualization.

Author: Enterprise Data Science Team
Version: 2.0.0
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read long description from README
readme_path = "README.md"
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Enterprise Business Intelligence Platform with advanced customer analytics and CRM integration"

# Read requirements from requirements file
requirements_path = "requirements/requirements.txt"
requirements = []
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from VERSION file
version = "2.0.0"
version_path = "VERSION"
if os.path.exists(version_path):
    with open(version_path, "r", encoding="utf-8") as fh:
        version = fh.read().strip()

setup(
    name="enterprise-business-intelligence",
    version=version,
    author="Enterprise Data Science Team",
    author_email="data-science@enterprise.com",
    description="Enterprise Business Intelligence Platform with Advanced Customer Analytics and CRM Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enterprise/business-intelligence-platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="business-intelligence, customer-analytics, crm-integration, machine-learning, enterprise",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=1.0",
            "pre-commit>=3.0"
        ],
        "enterprise": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "redis>=5.0.0",
            "celery>=5.3.0",
            "cryptography>=41.0.0"
        ],
        "viz": [
            "streamlit>=1.28",
            "plotly>=5.15",
            "dash>=2.14",
            "grafana-api>=1.0"
        ],
        "all": [
            "pytest>=7.0", "pytest-asyncio>=0.21.0", "pytest-cov>=4.0",
            "black>=22.0", "flake8>=4.0", "mypy>=1.0", "pre-commit>=3.0",
            "fastapi>=0.104.0", "uvicorn[standard]>=0.24.0",
            "redis>=5.0.0", "celery>=5.3.0", "cryptography>=41.0.0",
            "streamlit>=1.28", "plotly>=5.15", "dash>=2.14"
        ]
    },
    entry_points={
        "console_scripts": [
            "enterprise-bi=src.enterprise.enterprise_platform_manager:main",
            "enterprise-api=src.enterprise.api_gateway:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/enterprise/business-intelligence-platform/issues",
        "Source": "https://github.com/enterprise/business-intelligence-platform",
        "Documentation": "https://docs.enterprise.com/business-intelligence",
    },
)
