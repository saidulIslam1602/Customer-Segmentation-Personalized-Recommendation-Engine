"""
Real-time Analytics Engine - Streaming Data Processing and Live Insights
Provides real-time monitoring, alerting, and live dashboard capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
from collections import deque
import threading
import queue
import warnings

warnings.filterwarnings("ignore")


class RealTimeAnalyticsEngine:
    """
    Real-time Analytics and Monitoring Engine

    Features:
    - Streaming data processing
    - Real-time KPI monitoring
    - Live alerting system
    - Event-driven analytics
    - Performance dashboards
    - Threshold-based notifications
    - Real-time recommendations
    """

    def __init__(self, alert_thresholds=None, buffer_size=1000):
        """Initialize real-time analytics engine"""

        # Data buffers for streaming
        self.transaction_buffer = deque(maxlen=buffer_size)
        self.kpi_buffer = deque(maxlen=100)  # Store last 100 KPI calculations

        # Real-time metrics
        self.current_metrics = {}
        self.alerts = []
        self.events = queue.Queue()

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "revenue_drop_threshold": -0.20,  # 20% drop in revenue
            "transaction_spike_threshold": 3.0,  # 3x normal transaction volume
            "avg_order_decline_threshold": -0.15,  # 15% decline in AOV
            "customer_activity_drop": -0.25,  # 25% drop in customer activity
            "response_time_threshold": 5.0,  # 5 second response time
            "error_rate_threshold": 0.05,  # 5% error rate
        }

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None

        # Results directory
        self.results_dir = "reports/real_time"
        os.makedirs(self.results_dir, exist_ok=True)

        print(" Real-time Analytics Engine initialized")
        print(f"    Buffer size: {buffer_size}")
        print(f"    Alert thresholds: {len(self.alert_thresholds)} configured")

    def add_transaction(self, transaction_data):
        """Add a new transaction to the real-time stream"""
        # Add timestamp if not present
        if "timestamp" not in transaction_data:
            transaction_data["timestamp"] = datetime.now()

        # Add to buffer
        self.transaction_buffer.append(transaction_data)

        # Trigger real-time processing
        self.events.put(("new_transaction", transaction_data))

        return True

    def simulate_transaction_stream(
        self, base_data_path, duration_minutes=5, transactions_per_minute=10
    ):
        """Simulate real-time transaction stream from historical data"""
        print(f" Starting transaction stream simulation...")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"    Rate: {transactions_per_minute} transactions/minute")

        # Load historical data for simulation
        historical_data = pd.read_csv(base_data_path)

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        transaction_count = 0

        while datetime.now() < end_time and self.is_monitoring:
            # Select random transaction from historical data
            sample_transaction = historical_data.sample(1).iloc[0].to_dict()

            # Modify for real-time simulation
            sample_transaction["timestamp"] = datetime.now()
            sample_transaction["simulation_id"] = transaction_count

            # Add some realistic variations
            variation = np.random.normal(1.0, 0.1)  # 10% variation
            if "total_amount" in sample_transaction:
                sample_transaction["total_amount"] *= max(0.1, variation)

            # Add to stream
            self.add_transaction(sample_transaction)
            transaction_count += 1

            # Wait for next transaction
            time.sleep(60 / transactions_per_minute)  # Convert to seconds

        print(
            f" Stream simulation completed: {transaction_count} transactions processed"
        )
        return transaction_count

    def calculate_real_time_kpis(self):
        """Calculate KPIs from current buffer"""
        if len(self.transaction_buffer) == 0:
            return {}

        # Convert buffer to DataFrame for analysis
        df = pd.DataFrame(list(self.transaction_buffer))

        # Current time window (last 5 minutes)
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=5)

        # Filter to current window
        if "timestamp" in df.columns:
            recent_df = df[df["timestamp"] >= window_start]
        else:
            recent_df = df  # Use all data if no timestamp

        if len(recent_df) == 0:
            return {}

        # Calculate real-time KPIs
        kpis = {
            "timestamp": current_time,
            "transaction_count": len(recent_df),
            "total_revenue": recent_df["total_amount"].sum()
            if "total_amount" in recent_df.columns
            else 0,
            "avg_order_value": recent_df["total_amount"].mean()
            if "total_amount" in recent_df.columns
            else 0,
            "unique_customers": recent_df["customer_id"].nunique()
            if "customer_id" in recent_df.columns
            else 0,
            "transactions_per_minute": len(recent_df) / 5,  # 5-minute window
            "revenue_per_minute": (recent_df["total_amount"].sum() / 5)
            if "total_amount" in recent_df.columns
            else 0,
        }

        # Add to KPI buffer
        self.kpi_buffer.append(kpis)
        self.current_metrics = kpis

        return kpis

    def check_alerts(self, current_kpis):
        """Check for alert conditions based on current KPIs"""
        if len(self.kpi_buffer) < 2:
            return []  # Need at least 2 data points for comparison

        alerts = []
        previous_kpis = list(self.kpi_buffer)[-2]  # Previous calculation

        # Revenue drop alert
        if current_kpis["total_revenue"] > 0 and previous_kpis["total_revenue"] > 0:
            revenue_change = (
                current_kpis["total_revenue"] - previous_kpis["total_revenue"]
            ) / previous_kpis["total_revenue"]

            if revenue_change < self.alert_thresholds["revenue_drop_threshold"]:
                alerts.append(
                    {
                        "type": "revenue_drop",
                        "severity": "high",
                        "timestamp": current_kpis["timestamp"],
                        "message": f"Revenue dropped by {abs(revenue_change)*100:.1f}% in the last 5 minutes",
                        "current_value": current_kpis["total_revenue"],
                        "previous_value": previous_kpis["total_revenue"],
                    }
                )

        # Transaction volume spike
        if previous_kpis["transaction_count"] > 0:
            volume_ratio = (
                current_kpis["transaction_count"] / previous_kpis["transaction_count"]
            )

            if volume_ratio > self.alert_thresholds["transaction_spike_threshold"]:
                alerts.append(
                    {
                        "type": "transaction_spike",
                        "severity": "medium",
                        "timestamp": current_kpis["timestamp"],
                        "message": f"Transaction volume spiked by {(volume_ratio-1)*100:.1f}%",
                        "current_value": current_kpis["transaction_count"],
                        "previous_value": previous_kpis["transaction_count"],
                    }
                )

        # Average order value decline
        if current_kpis["avg_order_value"] > 0 and previous_kpis["avg_order_value"] > 0:
            aov_change = (
                current_kpis["avg_order_value"] - previous_kpis["avg_order_value"]
            ) / previous_kpis["avg_order_value"]

            if aov_change < self.alert_thresholds["avg_order_decline_threshold"]:
                alerts.append(
                    {
                        "type": "aov_decline",
                        "severity": "medium",
                        "timestamp": current_kpis["timestamp"],
                        "message": f"Average order value declined by {abs(aov_change)*100:.1f}%",
                        "current_value": current_kpis["avg_order_value"],
                        "previous_value": previous_kpis["avg_order_value"],
                    }
                )

        # Customer activity drop
        if previous_kpis["unique_customers"] > 0:
            customer_change = (
                current_kpis["unique_customers"] - previous_kpis["unique_customers"]
            ) / previous_kpis["unique_customers"]

            if customer_change < self.alert_thresholds["customer_activity_drop"]:
                alerts.append(
                    {
                        "type": "customer_activity_drop",
                        "severity": "high",
                        "timestamp": current_kpis["timestamp"],
                        "message": f"Customer activity dropped by {abs(customer_change)*100:.1f}%",
                        "current_value": current_kpis["unique_customers"],
                        "previous_value": previous_kpis["unique_customers"],
                    }
                )

        # Add alerts to main alerts list
        self.alerts.extend(alerts)

        return alerts

    def generate_real_time_recommendations(self, current_kpis, recent_alerts):
        """Generate real-time recommendations based on current state"""
        recommendations = []

        # Low transaction volume recommendation
        if current_kpis["transactions_per_minute"] < 2:
            recommendations.append(
                {
                    "type": "marketing",
                    "priority": "medium",
                    "title": "Boost Transaction Volume",
                    "action": "Launch flash sale or send push notifications to increase immediate activity",
                    "expected_impact": "Increase transactions by 20-30%",
                }
            )

        # Low average order value recommendation
        if current_kpis["avg_order_value"] < 20:
            recommendations.append(
                {
                    "type": "sales",
                    "priority": "medium",
                    "title": "Increase Average Order Value",
                    "action": "Promote product bundles or offer free shipping thresholds",
                    "expected_impact": "Increase AOV by 15-25%",
                }
            )

        # Alert-based recommendations
        for alert in recent_alerts:
            if alert["type"] == "revenue_drop":
                recommendations.append(
                    {
                        "type": "urgent",
                        "priority": "high",
                        "title": "Address Revenue Decline",
                        "action": "Investigate system issues, check payment processing, review pricing",
                        "expected_impact": "Prevent further revenue loss",
                    }
                )

            elif alert["type"] == "customer_activity_drop":
                recommendations.append(
                    {
                        "type": "customer",
                        "priority": "high",
                        "title": "Re-engage Customers",
                        "action": "Send personalized offers, check website performance, review user experience",
                        "expected_impact": "Recover 10-15% of lost activity",
                    }
                )

        return recommendations

    def process_real_time_events(self):
        """Process events from the real-time queue"""
        while self.is_monitoring:
            try:
                # Get event from queue (with timeout)
                event_type, event_data = self.events.get(timeout=1)

                if event_type == "new_transaction":
                    # Calculate updated KPIs
                    current_kpis = self.calculate_real_time_kpis()

                    # Check for alerts
                    new_alerts = self.check_alerts(current_kpis)

                    # Generate recommendations if there are alerts
                    if new_alerts:
                        recommendations = self.generate_real_time_recommendations(
                            current_kpis, new_alerts
                        )

                        # Log alerts and recommendations
                        for alert in new_alerts:
                            print(f" ALERT: {alert['message']}")

                        for rec in recommendations:
                            print(f" RECOMMENDATION: {rec['title']} - {rec['action']}")

                # Mark task as done
                self.events.task_done()

            except queue.Empty:
                continue  # No events to process
            except Exception as e:
                print(f" Error processing event: {e}")

    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            print("  Monitoring already active")
            return

        self.is_monitoring = True

        # Start event processing thread
        self.monitoring_thread = threading.Thread(target=self.process_real_time_events)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        print(" Real-time monitoring started")
        return True

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_monitoring:
            print("  Monitoring not active")
            return

        self.is_monitoring = False

        # Wait for thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        print(" Real-time monitoring stopped")
        return True

    def get_live_dashboard_data(self):
        """Get current data for live dashboard"""
        dashboard_data = {
            "timestamp": datetime.now(),
            "current_metrics": self.current_metrics,
            "recent_alerts": self.alerts[-10:]
            if len(self.alerts) > 10
            else self.alerts,
            "kpi_history": list(self.kpi_buffer)[-20:]
            if len(self.kpi_buffer) > 20
            else list(self.kpi_buffer),
            "buffer_status": {
                "transaction_buffer_size": len(self.transaction_buffer),
                "kpi_buffer_size": len(self.kpi_buffer),
                "total_alerts": len(self.alerts),
            },
        }

        return dashboard_data

    def export_real_time_data(self):
        """Export real-time data for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export transactions
        if self.transaction_buffer:
            transactions_df = pd.DataFrame(list(self.transaction_buffer))
            transactions_df.to_csv(
                f"{self.results_dir}/real_time_transactions_{timestamp}.csv",
                index=False,
            )

        # Export KPIs
        if self.kpi_buffer:
            kpis_df = pd.DataFrame(list(self.kpi_buffer))
            kpis_df.to_csv(
                f"{self.results_dir}/real_time_kpis_{timestamp}.csv", index=False
            )

        # Export alerts
        if self.alerts:
            alerts_df = pd.DataFrame(self.alerts)
            alerts_df.to_csv(
                f"{self.results_dir}/real_time_alerts_{timestamp}.csv", index=False
            )

        # Export dashboard data
        dashboard_data = self.get_live_dashboard_data()
        with open(f"{self.results_dir}/dashboard_data_{timestamp}.json", "w") as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        print(f" Real-time data exported with timestamp: {timestamp}")
        return timestamp

    def run_real_time_demo(self, data_path, duration_minutes=2):
        """Run a complete real-time analytics demo"""
        print(" REAL-TIME ANALYTICS DEMO")
        print("=" * 50)

        # Start monitoring
        self.start_monitoring()

        # Start transaction stream simulation in a separate thread
        stream_thread = threading.Thread(
            target=self.simulate_transaction_stream,
            args=(data_path, duration_minutes, 20),  # 20 transactions per minute
        )
        stream_thread.start()

        # Monitor for the duration
        start_time = datetime.now()
        last_status_time = start_time

        while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
            # Print status every 30 seconds
            if (datetime.now() - last_status_time).total_seconds() >= 30:
                dashboard_data = self.get_live_dashboard_data()

                print(f"\n LIVE STATUS ({datetime.now().strftime('%H:%M:%S')}):")
                if dashboard_data["current_metrics"]:
                    metrics = dashboard_data["current_metrics"]
                    print(
                        f"    Revenue (5min): ${metrics.get('total_revenue', 0):.2f}"
                    )
                    print(f"    Transactions: {metrics.get('transaction_count', 0)}")
                    print(f"    Customers: {metrics.get('unique_customers', 0)}")
                    print(f"    Total Alerts: {len(self.alerts)}")

                last_status_time = datetime.now()

            time.sleep(5)  # Check every 5 seconds

        # Wait for stream thread to complete
        stream_thread.join()

        # Stop monitoring
        self.stop_monitoring()

        # Export results
        self.export_real_time_data()

        # Final summary
        print(f"\n REAL-TIME DEMO COMPLETED!")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"    Transactions processed: {len(self.transaction_buffer)}")
        print(f"    KPI calculations: {len(self.kpi_buffer)}")
        print(f"    Alerts generated: {len(self.alerts)}")

        return self.get_live_dashboard_data()


def main():
    """Demo of real-time analytics engine"""
    print(" REAL-TIME ANALYTICS ENGINE DEMO")
    print("=" * 50)

    # Initialize engine
    rt_engine = RealTimeAnalyticsEngine()

    # Run demo with simulated data
    results = rt_engine.run_real_time_demo(
        "data/transactions_real.csv", duration_minutes=1
    )

    print(f"\n DEMO RESULTS:")
    print(f"   Current Metrics: {len(results['current_metrics'])} KPIs")
    print(f"   Recent Alerts: {len(results['recent_alerts'])}")
    print(f"   Buffer Status: {results['buffer_status']}")

    print(f"\n Real-time Analytics Engine Demo Completed!")


if __name__ == "__main__":
    main()
