"""
Configuration Management Module

Handles loading and validation of environment variables and configuration settings.
Provides centralized configuration management with type safety and validation.
"""

import os
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration"""

    url: str
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class RedisConfig:
    """Redis configuration"""

    url: str
    password: Optional[str] = None
    db: int = 0
    ttl_seconds: int = 3600


@dataclass
class APIConfig:
    """API configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    max_requests: int = 1000


@dataclass
class SecurityConfig:
    """Security configuration"""

    jwt_secret_key: str
    encryption_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24


@dataclass
class CRMConfig:
    """CRM integration configuration"""

    sync_enabled: bool = True
    sync_interval_minutes: int = 30
    batch_size: int = 1000

    # Dynamics 365
    dynamics365_base_url: Optional[str] = None
    dynamics365_client_id: Optional[str] = None
    dynamics365_client_secret: Optional[str] = None
    dynamics365_tenant_id: Optional[str] = None

    # Salesforce
    salesforce_instance_url: Optional[str] = None
    salesforce_client_id: Optional[str] = None
    salesforce_client_secret: Optional[str] = None
    salesforce_username: Optional[str] = None
    salesforce_password: Optional[str] = None

    # HubSpot
    hubspot_api_key: Optional[str] = None
    hubspot_base_url: str = "https://api.hubapi.com"


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""

    enabled: bool = True

    # Email alerts
    email_enabled: bool = True
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_from: Optional[str] = None
    email_to: Optional[str] = None
    email_username: Optional[str] = None
    email_password: Optional[str] = None

    # Slack alerts
    slack_enabled: bool = False
    slack_webhook_url: Optional[str] = None

    # Thresholds
    alert_response_time_ms: int = 5000
    alert_error_rate_percent: int = 5
    alert_memory_usage_percent: int = 80
    alert_cpu_usage_percent: int = 90
    alert_disk_usage_percent: int = 85


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""

    enabled: bool = True
    per_minute: int = 60
    per_hour: int = 1000


class Config:
    """
    Central configuration manager.

    Loads configuration from environment variables with proper validation
    and type conversion. Provides easy access to all configuration settings.
    """

    def __init__(self):
        """Initialize configuration from environment variables"""
        # Load .env file if it exists
        self._load_env_file()

        # Application settings
        self.environment = self._get_env("ENVIRONMENT", "development")
        self.debug = self._get_bool("DEBUG", False)
        self.log_level = self._get_env("LOG_LEVEL", "INFO")

        # Directories
        self.data_dir = self._get_env("DATA_DIR", "data")
        self.results_dir = self._get_env("RESULTS_DIR", "reports")
        self.models_dir = self._get_env("MODELS_DIR", "models")

        # Database
        self.database = DatabaseConfig(
            url=self._get_env(
                "DATABASE_URL",
                "postgresql://bi_user:password@localhost:5432/enterprise_bi",
            ),
            pool_size=self._get_int("DATABASE_POOL_SIZE", 10),
            max_overflow=self._get_int("DATABASE_MAX_OVERFLOW", 20),
        )

        # Redis
        self.redis = RedisConfig(
            url=self._get_env("REDIS_URL", "redis://localhost:6379/0"),
            password=self._get_env("REDIS_PASSWORD", None),
            db=self._get_int("REDIS_DB", 0),
            ttl_seconds=self._get_int("REDIS_TTL_SECONDS", 3600),
        )

        # API
        self.api = APIConfig(
            host=self._get_env("API_HOST", "0.0.0.0"),
            port=self._get_int("API_PORT", 8000),
            workers=self._get_int("API_WORKERS", 4),
            timeout=self._get_int("API_TIMEOUT", 30),
            max_requests=self._get_int("API_MAX_REQUESTS", 1000),
        )

        # Security
        self.security = SecurityConfig(
            jwt_secret_key=self._get_env(
                "JWT_SECRET_KEY", "change-this-to-a-secure-random-key"
            ),
            jwt_algorithm=self._get_env("JWT_ALGORITHM", "HS256"),
            jwt_expiry_hours=self._get_int("JWT_EXPIRY_HOURS", 24),
            encryption_key=self._get_env(
                "ENCRYPTION_KEY", "change-this-to-a-secure-random-key"
            ),
        )

        # CRM
        self.crm = CRMConfig(
            sync_enabled=self._get_bool("CRM_SYNC_ENABLED", True),
            sync_interval_minutes=self._get_int("CRM_SYNC_INTERVAL_MINUTES", 30),
            batch_size=self._get_int("CRM_BATCH_SIZE", 1000),
            dynamics365_base_url=self._get_env("DYNAMICS365_BASE_URL"),
            dynamics365_client_id=self._get_env("DYNAMICS365_CLIENT_ID"),
            dynamics365_client_secret=self._get_env("DYNAMICS365_CLIENT_SECRET"),
            dynamics365_tenant_id=self._get_env("DYNAMICS365_TENANT_ID"),
            salesforce_instance_url=self._get_env("SALESFORCE_INSTANCE_URL"),
            salesforce_client_id=self._get_env("SALESFORCE_CLIENT_ID"),
            salesforce_client_secret=self._get_env("SALESFORCE_CLIENT_SECRET"),
            salesforce_username=self._get_env("SALESFORCE_USERNAME"),
            salesforce_password=self._get_env("SALESFORCE_PASSWORD"),
            hubspot_api_key=self._get_env("HUBSPOT_API_KEY"),
            hubspot_base_url=self._get_env(
                "HUBSPOT_BASE_URL", "https://api.hubapi.com"
            ),
        )

        # Monitoring
        self.monitoring = MonitoringConfig(
            enabled=self._get_bool("MONITORING_ENABLED", True),
            email_enabled=self._get_bool("ALERT_EMAIL_ENABLED", True),
            email_smtp_server=self._get_env(
                "ALERT_EMAIL_SMTP_SERVER", "smtp.gmail.com"
            ),
            email_smtp_port=self._get_int("ALERT_EMAIL_SMTP_PORT", 587),
            email_from=self._get_env("ALERT_EMAIL_FROM"),
            email_to=self._get_env("ALERT_EMAIL_TO"),
            email_username=self._get_env("ALERT_EMAIL_USERNAME"),
            email_password=self._get_env("ALERT_EMAIL_PASSWORD"),
            slack_enabled=self._get_bool("ALERT_SLACK_ENABLED", False),
            slack_webhook_url=self._get_env("ALERT_SLACK_WEBHOOK_URL"),
            alert_response_time_ms=self._get_int("ALERT_RESPONSE_TIME_MS", 5000),
            alert_error_rate_percent=self._get_int("ALERT_ERROR_RATE_PERCENT", 5),
            alert_memory_usage_percent=self._get_int("ALERT_MEMORY_USAGE_PERCENT", 80),
            alert_cpu_usage_percent=self._get_int("ALERT_CPU_USAGE_PERCENT", 90),
            alert_disk_usage_percent=self._get_int("ALERT_DISK_USAGE_PERCENT", 85),
        )

        # Rate Limiting
        self.rate_limit = RateLimitConfig(
            enabled=self._get_bool("RATE_LIMIT_ENABLED", True),
            per_minute=self._get_int("RATE_LIMIT_PER_MINUTE", 60),
            per_hour=self._get_int("RATE_LIMIT_PER_HOUR", 1000),
        )

        # CORS
        self.cors_enabled = self._get_bool("CORS_ENABLED", True)
        self.cors_origins = self._get_list("CORS_ORIGINS", ["http://localhost:3000"])

        # Feature Flags
        self.features = {
            "realtime_analytics": self._get_bool("FEATURE_REALTIME_ANALYTICS", True),
            "ab_testing": self._get_bool("FEATURE_AB_TESTING", True),
            "fraud_detection": self._get_bool("FEATURE_FRAUD_DETECTION", True),
            "inventory_optimization": self._get_bool(
                "FEATURE_INVENTORY_OPTIMIZATION", True
            ),
        }

        # External Services
        self.mlflow_tracking_uri = self._get_env(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.prometheus_port = self._get_int("PROMETHEUS_PORT", 9090)
        self.grafana_port = self._get_int("GRAFANA_PORT", 3000)

    def _load_env_file(self):
        """Load .env file if it exists"""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable"""
        value = os.getenv(key, default)
        return value if value else default

    def _get_int(self, key: str, default: int) -> int:
        """Get integer environment variable"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def _get_list(self, key: str, default: List[str]) -> List[str]:
        """Get list environment variable (comma-separated)"""
        value = os.getenv(key)
        if value is None:
            return default
        return [item.strip() for item in value.split(",") if item.strip()]

    def validate(self):
        """Validate critical configuration"""
        errors = []

        # Check security keys in production
        if self.environment == "production":
            if "change-this" in self.security.jwt_secret_key.lower():
                errors.append("JWT_SECRET_KEY must be changed in production")
            if "change-this" in self.security.encryption_key.lower():
                errors.append("ENCRYPTION_KEY must be changed in production")

        # Check required CRM credentials if sync is enabled
        if self.crm.sync_enabled:
            if not any(
                [
                    self.crm.dynamics365_client_id,
                    self.crm.salesforce_client_id,
                    self.crm.hubspot_api_key,
                ]
            ):
                errors.append(
                    "At least one CRM system must be configured when CRM_SYNC_ENABLED is true"
                )

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"


# Global config instance
config = Config()
