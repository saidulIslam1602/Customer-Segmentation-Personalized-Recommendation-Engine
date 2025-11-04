"""
Enterprise Performance Monitor

This module provides comprehensive performance monitoring, alerting, and metrics
collection for the enterprise business intelligence platform. It implements
real-time system monitoring, automated alerting, and performance analytics.

Author: Enterprise Data Science Team
Version: 2.0.0
Created: 2024-11-04
Last Modified: 2024-11-04

Classes:
    EnterprisePerformanceMonitor: Main performance monitoring system
    PerformanceMetric: Data class for individual metrics
    AlertRule: Configuration for automated alerts

Features:
    - Real-time system resource monitoring (CPU, memory, disk)
    - Business KPI tracking and analysis
    - ML model performance metrics collection
    - Automated alerting with configurable thresholds
    - Performance trend analysis and reporting
    - Dashboard data generation for visualization

Dependencies:
    - psutil: System and process monitoring
    - pandas: Data manipulation for metrics
    - threading: Background monitoring operations
    - collections: Efficient data structures for metrics storage
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
import logging
from dataclasses import dataclass, asdict
import threading
from collections import deque


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""

    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    category: str
    threshold: Optional[float] = None
    status: str = "normal"


@dataclass
class AlertRule:
    """Alert rule configuration"""

    metric_name: str
    threshold: float
    comparison: str  # 'greater', 'less', 'equal'
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True


class EnterprisePerformanceMonitor:
    """
    Enterprise Performance Monitoring System

    Features:
    - Real-time performance tracking
    - System resource monitoring
    - ML model performance metrics
    - Business KPI tracking
    - Automated alerting
    - Performance dashboards
    - Historical trend analysis
    """

    def __init__(self, config: Dict):
        self.config = config
        self.performance_tracking = config.get("performance_tracking", True)
        self.alert_thresholds = config.get("alert_thresholds", {})

        # Metrics storage
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k metrics
        self.current_metrics = {}
        self.alerts = []

        # Alert rules
        self.alert_rules = self._setup_default_alert_rules()

        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Start monitoring if enabled
        if self.performance_tracking:
            self.start_monitoring()

    def _setup_default_alert_rules(self) -> List[AlertRule]:
        """Setup default alert rules"""
        return [
            AlertRule("response_time_ms", 5000, "greater", "high"),
            AlertRule("error_rate_percent", 5, "greater", "medium"),
            AlertRule("memory_usage_percent", 80, "greater", "medium"),
            AlertRule("cpu_usage_percent", 90, "greater", "high"),
            AlertRule("disk_usage_percent", 85, "greater", "medium"),
            AlertRule("model_accuracy", 0.8, "less", "high"),
            AlertRule("crm_sync_success_rate", 95, "less", "medium"),
        ]

    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()
            print("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Check alert rules
                self._check_alert_rules()

                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self._record_metric(
            "cpu_usage_percent", cpu_percent, "percent", "system", timestamp
        )

        # Memory metrics
        memory = psutil.virtual_memory()
        self._record_metric(
            "memory_usage_percent", memory.percent, "percent", "system", timestamp
        )
        self._record_metric(
            "memory_available_gb",
            memory.available / (1024**3),
            "GB",
            "system",
            timestamp,
        )

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        self._record_metric(
            "disk_usage_percent", disk_percent, "percent", "system", timestamp
        )

        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self._record_metric(
                "network_bytes_sent", network.bytes_sent, "bytes", "network", timestamp
            )
            self._record_metric(
                "network_bytes_recv", network.bytes_recv, "bytes", "network", timestamp
            )
        except:
            pass

    def _record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        category: str,
        timestamp: datetime = None,
        threshold: float = None,
    ):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()

        # Determine status based on threshold
        status = "normal"
        if threshold and value > threshold:
            status = "warning"

        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            unit=unit,
            category=category,
            threshold=threshold,
            status=status,
        )

        # Store metric
        self.metrics_history.append(metric)
        self.current_metrics[name] = metric

    def collect_metrics(self, core_results: Dict, crm_sync_results: Dict) -> Dict:
        """Collect comprehensive performance metrics"""
        timestamp = datetime.now()

        # Business performance metrics
        business_metrics = self._extract_business_metrics(core_results)

        # CRM integration metrics
        crm_metrics = self._extract_crm_metrics(crm_sync_results)

        # System performance metrics
        system_metrics = self._get_current_system_metrics()

        # ML model performance metrics
        ml_metrics = self._extract_ml_metrics(core_results)

        # Calculate overall performance score
        overall_score = self._calculate_overall_score(
            business_metrics, crm_metrics, system_metrics, ml_metrics
        )

        performance_summary = {
            "timestamp": timestamp.isoformat(),
            "overall_score": overall_score,
            "business_metrics": business_metrics,
            "crm_metrics": crm_metrics,
            "system_metrics": system_metrics,
            "ml_metrics": ml_metrics,
            "alerts_count": len(self.alerts),
            "monitoring_status": "active" if self.monitoring_active else "inactive",
        }

        # Save metrics
        self._save_performance_report(performance_summary)

        return performance_summary

    def _extract_business_metrics(self, core_results: Dict) -> Dict:
        """Extract business performance metrics"""
        return {
            "revenue_impact": core_results.get("revenue_impact", 0),
            "customers_analyzed": core_results.get("customers_analyzed", 0),
            "roi_percentage": core_results.get("roi_percentage", 0),
            "customer_retention_rate": 98.4,  # Mock value
            "processing_time_seconds": core_results.get("processing_time", 0),
        }

    def _extract_crm_metrics(self, crm_sync_results: Dict) -> Dict:
        """Extract CRM integration metrics"""
        if not crm_sync_results:
            return {
                "systems_connected": 0,
                "total_records_synced": 0,
                "sync_success_rate": 0,
                "average_sync_time": 0,
            }

        total_records = sum(
            result.records_synced if hasattr(result, "records_synced") else 0
            for result in crm_sync_results.values()
        )

        successful_syncs = sum(
            1
            for result in crm_sync_results.values()
            if (hasattr(result, "status") and result.status == "success")
            or (isinstance(result, dict) and result.get("status") == "success")
        )

        success_rate = (
            (successful_syncs / len(crm_sync_results)) * 100 if crm_sync_results else 0
        )

        return {
            "systems_connected": len(crm_sync_results),
            "total_records_synced": total_records,
            "sync_success_rate": success_rate,
            "average_sync_time": 2.5,  # Mock value in seconds
        }

    def _get_current_system_metrics(self) -> Dict:
        """Get current system performance metrics"""
        return {
            "cpu_usage_percent": self.current_metrics.get(
                "cpu_usage_percent",
                PerformanceMetric(
                    datetime.now(), "cpu_usage_percent", 0, "percent", "system"
                ),
            ).value,
            "memory_usage_percent": self.current_metrics.get(
                "memory_usage_percent",
                PerformanceMetric(
                    datetime.now(), "memory_usage_percent", 0, "percent", "system"
                ),
            ).value,
            "disk_usage_percent": self.current_metrics.get(
                "disk_usage_percent",
                PerformanceMetric(
                    datetime.now(), "disk_usage_percent", 0, "percent", "system"
                ),
            ).value,
            "uptime_hours": self._get_system_uptime(),
        }

    def _extract_ml_metrics(self, core_results: Dict) -> Dict:
        """Extract ML model performance metrics"""
        return {
            "churn_prediction_accuracy": 0.95,  # Mock value
            "recommendation_precision": 0.18,  # Mock value
            "segmentation_quality": 0.89,  # Mock value
            "model_training_time": 45.2,  # Mock value in seconds
            "prediction_latency_ms": 150,  # Mock value
        }

    def _calculate_overall_score(
        self,
        business_metrics: Dict,
        crm_metrics: Dict,
        system_metrics: Dict,
        ml_metrics: Dict,
    ) -> float:
        """Calculate overall performance score"""
        scores = []

        # Business performance (40% weight)
        business_score = min(
            100, (business_metrics.get("roi_percentage", 0) / 1000) * 100
        )
        scores.append(business_score * 0.4)

        # CRM integration (20% weight)
        crm_score = crm_metrics.get("sync_success_rate", 0)
        scores.append(crm_score * 0.2)

        # System performance (20% weight)
        system_score = 100 - max(
            system_metrics.get("cpu_usage_percent", 0),
            system_metrics.get("memory_usage_percent", 0),
        )
        scores.append(max(0, system_score) * 0.2)

        # ML performance (20% weight)
        ml_score = (
            ml_metrics.get("churn_prediction_accuracy", 0) * 100 * 0.4
            + ml_metrics.get("recommendation_precision", 0) * 100 * 0.3
            + ml_metrics.get("segmentation_quality", 0) * 100 * 0.3
        )
        scores.append(ml_score * 0.2)

        return round(sum(scores), 1)

    def _get_system_uptime(self) -> float:
        """Get system uptime in hours"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            return round(uptime_seconds / 3600, 1)
        except:
            return 0.0

    def _check_alert_rules(self):
        """Check alert rules against current metrics"""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            metric = self.current_metrics.get(rule.metric_name)
            if not metric:
                continue

            triggered = False
            if rule.comparison == "greater" and metric.value > rule.threshold:
                triggered = True
            elif rule.comparison == "less" and metric.value < rule.threshold:
                triggered = True
            elif rule.comparison == "equal" and metric.value == rule.threshold:
                triggered = True

            if triggered:
                self._create_alert(rule, metric)

    def _create_alert(self, rule: AlertRule, metric: PerformanceMetric):
        """Create an alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": rule.metric_name,
            "current_value": metric.value,
            "threshold": rule.threshold,
            "severity": rule.severity,
            "message": f"{rule.metric_name} ({metric.value}{metric.unit}) exceeded threshold ({rule.threshold})",
        }

        self.alerts.append(alert)

        # Log alert
        self.logger.warning(f"ALERT: {alert['message']}")

        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

    def _save_performance_report(self, performance_summary: Dict):
        """Save performance report to file"""
        try:
            os.makedirs("reports/performance", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with open(
                f"reports/performance/performance_report_{timestamp}.json", "w"
            ) as f:
                json.dump(performance_summary, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")

    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                name: asdict(metric) for name, metric in self.current_metrics.items()
            },
            "alerts_count": len(self.alerts),
            "monitoring_active": self.monitoring_active,
        }

    def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_metrics = [
            asdict(metric)
            for metric in self.metrics_history
            if metric.timestamp > cutoff_time
        ]

        return filtered_metrics

    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_alerts = [
            alert
            for alert in self.alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]

        return filtered_alerts

    def add_custom_metric(
        self, name: str, value: float, unit: str, category: str = "custom"
    ):
        """Add a custom performance metric"""
        self._record_metric(name, value, unit, category)

    def create_performance_dashboard(self) -> Dict:
        """Create performance dashboard data"""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_score": self._calculate_current_overall_score(),
                "total_metrics": len(self.current_metrics),
                "active_alerts": len(
                    [a for a in self.alerts if self._is_recent_alert(a)]
                ),
                "monitoring_status": "active" if self.monitoring_active else "inactive",
            },
            "current_metrics": self.get_current_metrics(),
            "recent_alerts": self.get_alerts(hours=1),
            "trends": self._calculate_trends(),
        }

        return dashboard_data

    def _calculate_current_overall_score(self) -> float:
        """Calculate current overall performance score"""
        # Simplified calculation based on current metrics
        scores = []

        # System health
        cpu_metric = self.current_metrics.get("cpu_usage_percent")
        if cpu_metric:
            scores.append(max(0, 100 - cpu_metric.value))

        memory_metric = self.current_metrics.get("memory_usage_percent")
        if memory_metric:
            scores.append(max(0, 100 - memory_metric.value))

        # Default score if no metrics available
        if not scores:
            return 85.0

        return round(sum(scores) / len(scores), 1)

    def _is_recent_alert(self, alert: Dict, hours: int = 1) -> bool:
        """Check if alert is recent"""
        alert_time = datetime.fromisoformat(alert["timestamp"])
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return alert_time > cutoff_time

    def _calculate_trends(self) -> Dict:
        """Calculate performance trends"""
        trends = {}

        # Calculate trends for key metrics
        key_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent",
            "disk_usage_percent",
        ]

        for metric_name in key_metrics:
            recent_values = [
                m.value
                for m in self.metrics_history
                if m.metric_name == metric_name
                and m.timestamp > datetime.now() - timedelta(hours=1)
            ]

            if len(recent_values) >= 2:
                trend = (
                    "increasing"
                    if recent_values[-1] > recent_values[0]
                    else "decreasing"
                )
                change = recent_values[-1] - recent_values[0]
            else:
                trend = "stable"
                change = 0

            trends[metric_name] = {
                "trend": trend,
                "change": round(change, 2),
                "data_points": len(recent_values),
            }

        return trends
