"""
Executive Business Intelligence Dashboard
Comprehensive business insights and KPI monitoring for retail executives
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    # Optional dependencies for visualization
    pass

# Import business intelligence modules
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.churn_prediction import ChurnPredictionEngine
from models.inventory_optimization import InventoryOptimizationEngine
from models.pricing_optimization import PricingOptimizationEngine
from models.fraud_detection import FraudDetectionEngine
from models.marketing_attribution import MarketingAttributionEngine


class ExecutiveDashboard:
    """
    Executive Dashboard for Comprehensive Business Intelligence

    Features:
    - Real-time KPI monitoring
    - Customer analytics insights
    - Revenue optimization opportunities
    - Risk management alerts
    - Marketing performance tracking
    """

    def __init__(self):
        """Initialize dashboard with all business intelligence engines"""
        self.data_paths = {
            "transactions": "data/transactions_real.csv",
            "customers": "data/customers_real.csv",
            "products": "data/products_real.csv",
            "digital_events": "data/digital_events_real.csv",
        }

        # Initialize engines
        self.churn_engine = None
        self.inventory_engine = None
        self.pricing_engine = None
        self.fraud_engine = None
        self.marketing_engine = None

        # Dashboard data
        self.kpi_data = None
        self.insights_data = None

    def initialize_engines(self):
        """Initialize all business intelligence engines"""
        print(" Initializing business intelligence engines...")

        self.churn_engine = ChurnPredictionEngine(
            self.data_paths["transactions"], self.data_paths["customers"]
        )

        self.inventory_engine = InventoryOptimizationEngine(
            self.data_paths["transactions"], self.data_paths["products"]
        )

        self.pricing_engine = PricingOptimizationEngine(
            self.data_paths["transactions"], self.data_paths["products"]
        )

        self.fraud_engine = FraudDetectionEngine(
            self.data_paths["transactions"], self.data_paths["customers"]
        )

        self.marketing_engine = MarketingAttributionEngine(
            self.data_paths["transactions"],
            self.data_paths["customers"],
            self.data_paths["digital_events"],
        )

        print(" All engines initialized successfully")

    def calculate_executive_kpis(self):
        """Calculate comprehensive executive KPIs"""
        print(" Calculating executive KPIs...")

        # Load base data
        transactions = pd.read_csv(self.data_paths["transactions"])
        customers = pd.read_csv(self.data_paths["customers"])
        products = pd.read_csv(self.data_paths["products"])

        transactions["transaction_date"] = pd.to_datetime(
            transactions["transaction_date"]
        )
        customers["member_since"] = pd.to_datetime(customers["member_since"])

        # Current period (last 30 days)
        current_date = transactions["transaction_date"].max()
        current_period_start = current_date - timedelta(days=30)
        previous_period_start = current_date - timedelta(days=60)

        current_transactions = transactions[
            transactions["transaction_date"] >= current_period_start
        ]
        previous_transactions = transactions[
            (transactions["transaction_date"] >= previous_period_start)
            & (transactions["transaction_date"] < current_period_start)
        ]

        # Revenue KPIs
        current_revenue = current_transactions["total_amount"].sum()
        previous_revenue = previous_transactions["total_amount"].sum()
        revenue_growth = (
            (current_revenue - previous_revenue) / previous_revenue
            if previous_revenue > 0
            else 0
        )

        # Customer KPIs
        current_customers = current_transactions["customer_id"].nunique()
        previous_customers = previous_transactions["customer_id"].nunique()
        customer_growth = (
            (current_customers - previous_customers) / previous_customers
            if previous_customers > 0
            else 0
        )

        # Transaction KPIs
        current_transactions_count = len(current_transactions)
        previous_transactions_count = len(previous_transactions)
        transaction_growth = (
            (current_transactions_count - previous_transactions_count)
            / previous_transactions_count
            if previous_transactions_count > 0
            else 0
        )

        # Average Order Value
        current_aov = current_transactions["total_amount"].mean()
        previous_aov = previous_transactions["total_amount"].mean()
        aov_growth = (
            (current_aov - previous_aov) / previous_aov if previous_aov > 0 else 0
        )

        # Customer Lifetime Value (simplified)
        customer_clv = transactions.groupby("customer_id")["total_amount"].sum().mean()

        # Customer Acquisition Cost (estimated)
        new_customers_current = customers[
            customers["member_since"] >= current_period_start
        ]
        estimated_marketing_spend = (
            current_revenue * 0.05
        )  # Assume 5% of revenue on marketing
        cac = (
            estimated_marketing_spend / len(new_customers_current)
            if len(new_customers_current) > 0
            else 0
        )

        # Inventory metrics
        total_products = len(products)
        active_products = transactions["product_id"].nunique()
        product_utilization = active_products / total_products

        # Channel performance
        channel_revenue = (
            current_transactions.groupby("channel")["total_amount"].sum().to_dict()
        )
        top_channel = max(channel_revenue, key=channel_revenue.get)

        self.kpi_data = {
            "revenue": {
                "current": current_revenue,
                "previous": previous_revenue,
                "growth": revenue_growth,
                "trend": "up" if revenue_growth > 0 else "down",
            },
            "customers": {
                "current": current_customers,
                "previous": previous_customers,
                "growth": customer_growth,
                "total_customers": len(customers),
                "trend": "up" if customer_growth > 0 else "down",
            },
            "transactions": {
                "current": current_transactions_count,
                "previous": previous_transactions_count,
                "growth": transaction_growth,
                "trend": "up" if transaction_growth > 0 else "down",
            },
            "aov": {
                "current": current_aov,
                "previous": previous_aov,
                "growth": aov_growth,
                "trend": "up" if aov_growth > 0 else "down",
            },
            "customer_metrics": {
                "clv": customer_clv,
                "cac": cac,
                "ltv_cac_ratio": customer_clv / cac if cac > 0 else 0,
            },
            "inventory": {
                "total_products": total_products,
                "active_products": active_products,
                "utilization": product_utilization,
            },
            "channels": {"performance": channel_revenue, "top_channel": top_channel},
        }

        print(" Executive KPIs calculated successfully")
        return self.kpi_data

    def generate_business_insights(self):
        """Generate comprehensive business insights from all engines"""
        print(" Generating business insights...")

        if not all(
            [
                self.churn_engine,
                self.inventory_engine,
                self.pricing_engine,
                self.fraud_engine,
                self.marketing_engine,
            ]
        ):
            self.initialize_engines()

        insights = {}

        # Churn insights
        try:
            churn_data = self.churn_engine.create_churn_dashboard_data()
            insights["churn"] = {
                "high_risk_customers": churn_data["churn_summary"][
                    "high_risk_customers"
                ],
                "churn_rate": churn_data["churn_summary"]["churn_rate"],
                "retention_opportunity": churn_data["churn_summary"][
                    "high_risk_customers"
                ]
                * 150,  # Avg customer value
                "status": (
                    "critical"
                    if churn_data["churn_summary"]["churn_rate"] > 0.15
                    else (
                        "warning"
                        if churn_data["churn_summary"]["churn_rate"] > 0.10
                        else "good"
                    )
                ),
            }
        except Exception as e:
            print(f"Warning: Churn analysis failed: {e}")
            insights["churn"] = {"status": "error", "message": str(e)}

        # Inventory insights
        try:
            inventory_recs = (
                self.inventory_engine.generate_procurement_recommendations()
            )
            insights["inventory"] = {
                "high_priority_reorders": inventory_recs["high_priority_reorders"][
                    "count"
                ],
                "overstock_alerts": inventory_recs["overstock_alerts"]["count"],
                "total_reorder_value": inventory_recs["high_priority_reorders"][
                    "total_value"
                ],
                "status": (
                    "critical"
                    if inventory_recs["high_priority_reorders"]["count"] > 50
                    else (
                        "warning"
                        if inventory_recs["high_priority_reorders"]["count"] > 20
                        else "good"
                    )
                ),
            }
        except Exception as e:
            print(f"Warning: Inventory analysis failed: {e}")
            insights["inventory"] = {"status": "error", "message": str(e)}

        # Pricing insights
        try:
            pricing_data = self.pricing_engine.create_pricing_dashboard_data()
            insights["pricing"] = {
                "high_revenue_potential": pricing_data["pricing_summary"][
                    "high_revenue_potential"
                ],
                "elastic_products": pricing_data["pricing_summary"]["elastic_products"],
                "avg_elasticity": pricing_data["pricing_summary"][
                    "avg_price_elasticity"
                ],
                "status": (
                    "opportunity"
                    if pricing_data["pricing_summary"]["high_revenue_potential"] > 10
                    else "good"
                ),
            }
        except Exception as e:
            print(f"Warning: Pricing analysis failed: {e}")
            insights["pricing"] = {"status": "error", "message": str(e)}

        # Fraud insights
        try:
            fraud_data = self.fraud_engine.create_fraud_dashboard_data()
            insights["fraud"] = {
                "high_risk_customers": fraud_data["fraud_summary"][
                    "high_risk_customers"
                ],
                "fraud_rate": fraud_data["fraud_summary"]["fraud_rate"],
                "anomalous_transactions": fraud_data["fraud_summary"][
                    "total_anomalous_transactions"
                ],
                "status": (
                    "critical"
                    if fraud_data["fraud_summary"]["fraud_rate"] > 0.05
                    else (
                        "warning"
                        if fraud_data["fraud_summary"]["fraud_rate"] > 0.02
                        else "good"
                    )
                ),
            }
        except Exception as e:
            print(f"Warning: Fraud analysis failed: {e}")
            insights["fraud"] = {"status": "error", "message": str(e)}

        # Marketing insights
        try:
            marketing_data = self.marketing_engine.create_marketing_dashboard_data()
            insights["marketing"] = {
                "top_channel": marketing_data["attribution_summary"][
                    "top_performing_channel"
                ],
                "best_roi_channel": marketing_data["attribution_summary"][
                    "best_roi_channel"
                ],
                "avg_customer_value": marketing_data["attribution_summary"][
                    "avg_customer_value"
                ],
                "total_attributed_revenue": marketing_data["attribution_summary"][
                    "total_attributed_revenue"
                ],
                "status": "good",
            }
        except Exception as e:
            print(f"Warning: Marketing analysis failed: {e}")
            insights["marketing"] = {"status": "error", "message": str(e)}

        self.insights_data = insights

        print(" Business insights generated successfully")
        return insights

    def create_executive_summary(self):
        """Create executive summary with key findings and recommendations"""
        if self.kpi_data is None:
            self.calculate_executive_kpis()

        if self.insights_data is None:
            self.generate_business_insights()

        # Priority recommendations based on insights
        recommendations = []

        # Revenue optimization
        if self.kpi_data["revenue"]["growth"] < 0:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Revenue",
                    "issue": "Revenue decline detected",
                    "recommendation": "Implement pricing optimization and customer retention campaigns",
                    "impact": "High",
                    "effort": "Medium",
                }
            )

        # Customer retention
        if self.insights_data.get("churn", {}).get("status") in ["critical", "warning"]:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Customer Retention",
                    "issue": f"High churn risk: {self.insights_data['churn'].get('high_risk_customers', 0)} customers",
                    "recommendation": "Launch targeted retention campaigns for high-risk customers",
                    "impact": "High",
                    "effort": "Low",
                }
            )

        # Inventory management
        if self.insights_data.get("inventory", {}).get("status") in [
            "critical",
            "warning",
        ]:
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Inventory",
                    "issue": f"Inventory issues: {self.insights_data['inventory'].get('high_priority_reorders', 0)} urgent reorders",
                    "recommendation": "Optimize inventory levels and implement automated reordering",
                    "impact": "Medium",
                    "effort": "High",
                }
            )

        # Fraud prevention
        if self.insights_data.get("fraud", {}).get("status") in ["critical", "warning"]:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Risk Management",
                    "issue": f"Fraud risk detected: {self.insights_data['fraud'].get('high_risk_customers', 0)} high-risk customers",
                    "recommendation": "Implement enhanced fraud monitoring and verification processes",
                    "impact": "High",
                    "effort": "Medium",
                }
            )

        # Pricing opportunities
        if self.insights_data.get("pricing", {}).get("status") == "opportunity":
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Pricing",
                    "issue": f"Pricing optimization opportunity: {self.insights_data['pricing'].get('high_revenue_potential', 0)} products",
                    "recommendation": "Implement dynamic pricing for high-potential products",
                    "impact": "Medium",
                    "effort": "Medium",
                }
            )

        executive_summary = {
            "period": f"{datetime.now().strftime('%B %Y')}",
            "overall_health": self._calculate_overall_health(),
            "key_metrics": {
                "revenue": self.kpi_data["revenue"],
                "customers": self.kpi_data["customers"],
                "aov": self.kpi_data["aov"],
            },
            "critical_alerts": [
                rec for rec in recommendations if rec["priority"] == "High"
            ],
            "opportunities": [
                rec for rec in recommendations if rec["priority"] == "Medium"
            ],
            "top_achievements": self._identify_achievements(),
            "next_actions": recommendations[:3],  # Top 3 recommendations
        }

        return executive_summary

    def _calculate_overall_health(self):
        """Calculate overall business health score"""
        health_score = 0
        max_score = 0

        # Revenue health (25%)
        if self.kpi_data["revenue"]["growth"] > 0.1:
            health_score += 25
        elif self.kpi_data["revenue"]["growth"] > 0:
            health_score += 15
        elif self.kpi_data["revenue"]["growth"] > -0.05:
            health_score += 10
        max_score += 25

        # Customer health (25%)
        if self.kpi_data["customers"]["growth"] > 0.05:
            health_score += 25
        elif self.kpi_data["customers"]["growth"] > 0:
            health_score += 15
        elif self.kpi_data["customers"]["growth"] > -0.02:
            health_score += 10
        max_score += 25

        # Churn health (20%)
        churn_status = self.insights_data.get("churn", {}).get("status", "good")
        if churn_status == "good":
            health_score += 20
        elif churn_status == "warning":
            health_score += 10
        max_score += 20

        # Fraud health (15%)
        fraud_status = self.insights_data.get("fraud", {}).get("status", "good")
        if fraud_status == "good":
            health_score += 15
        elif fraud_status == "warning":
            health_score += 8
        max_score += 15

        # Inventory health (15%)
        inventory_status = self.insights_data.get("inventory", {}).get("status", "good")
        if inventory_status == "good":
            health_score += 15
        elif inventory_status == "warning":
            health_score += 8
        max_score += 15

        health_percentage = (health_score / max_score) * 100

        if health_percentage >= 80:
            return {"score": health_percentage, "status": "Excellent", "color": "green"}
        elif health_percentage >= 60:
            return {"score": health_percentage, "status": "Good", "color": "yellow"}
        elif health_percentage >= 40:
            return {"score": health_percentage, "status": "Warning", "color": "orange"}
        else:
            return {"score": health_percentage, "status": "Critical", "color": "red"}

    def _identify_achievements(self):
        """Identify top business achievements"""
        achievements = []

        if self.kpi_data["revenue"]["growth"] > 0.1:
            achievements.append(
                f"Strong revenue growth: {self.kpi_data['revenue']['growth']:.1%}"
            )

        if self.kpi_data["customers"]["growth"] > 0.05:
            achievements.append(
                f"Customer base expansion: {self.kpi_data['customers']['growth']:.1%}"
            )

        if self.kpi_data["aov"]["growth"] > 0.05:
            achievements.append(
                f"Increased average order value: {self.kpi_data['aov']['growth']:.1%}"
            )

        if self.kpi_data["customer_metrics"]["ltv_cac_ratio"] > 3:
            achievements.append(
                f"Healthy LTV:CAC ratio: {self.kpi_data['customer_metrics']['ltv_cac_ratio']:.1f}:1"
            )

        return achievements[:3]  # Top 3 achievements

    def save_dashboard_data(self):
        """Save all dashboard data to files"""
        print(" Saving dashboard data...")

        # Ensure we have all data
        if self.kpi_data is None:
            self.calculate_executive_kpis()

        if self.insights_data is None:
            self.generate_business_insights()

        executive_summary = self.create_executive_summary()

        # Save comprehensive dashboard data
        dashboard_export = {
            "generated_at": datetime.now().isoformat(),
            "kpis": self.kpi_data,
            "insights": self.insights_data,
            "executive_summary": executive_summary,
        }

        with open("results/executive_dashboard_data.json", "w") as f:
            json.dump(dashboard_export, f, indent=2, default=str)

        # Save executive summary separately
        with open("results/executive_summary.json", "w") as f:
            json.dump(executive_summary, f, indent=2, default=str)

        print(" Dashboard data saved successfully")
        print(" Files created:")
        print("   - results/executive_dashboard_data.json")
        print("   - results/executive_summary.json")

        return dashboard_export


def main():
    """Main execution function"""
    print(" EXECUTIVE BUSINESS INTELLIGENCE DASHBOARD")
    print("=" * 60)

    # Initialize dashboard
    dashboard = ExecutiveDashboard()

    # Generate comprehensive business intelligence
    dashboard.calculate_executive_kpis()
    dashboard.generate_business_insights()

    # Create and save dashboard data
    dashboard_data = dashboard.save_dashboard_data()

    # Print executive summary
    executive_summary = dashboard.create_executive_summary()

    print(f"\n EXECUTIVE SUMMARY - {executive_summary['period']}")
    print("=" * 50)
    print(
        f"Overall Business Health: {executive_summary['overall_health']['status']} ({executive_summary['overall_health']['score']:.0f}%)"
    )

    print(f"\n Key Metrics:")
    print(
        f"   Revenue Growth: {executive_summary['key_metrics']['revenue']['growth']:.1%}"
    )
    print(
        f"   Customer Growth: {executive_summary['key_metrics']['customers']['growth']:.1%}"
    )
    print(f"   AOV Growth: {executive_summary['key_metrics']['aov']['growth']:.1%}")

    if executive_summary["critical_alerts"]:
        print(f"\n Critical Alerts ({len(executive_summary['critical_alerts'])}):")
        for alert in executive_summary["critical_alerts"]:
            print(f"   • {alert['category']}: {alert['issue']}")

    if executive_summary["opportunities"]:
        print(f"\n Opportunities ({len(executive_summary['opportunities'])}):")
        for opp in executive_summary["opportunities"]:
            print(f"   • {opp['category']}: {opp['recommendation']}")

    if executive_summary["top_achievements"]:
        print(f"\n Top Achievements:")
        for achievement in executive_summary["top_achievements"]:
            print(f"   • {achievement}")

    print(f"\n Next Actions:")
    for i, action in enumerate(executive_summary["next_actions"], 1):
        print(
            f"   {i}. {action['recommendation']} (Impact: {action['impact']}, Effort: {action['effort']})"
        )

    print("\n Executive dashboard analysis completed!")


if __name__ == "__main__":
    main()
