"""
Enhanced Business Intelligence Pipeline
Main entry point for the comprehensive retail analytics platform
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import models
from models.customer_segmentation import CustomerSegmentation
from models.recommendation_engine import RecommendationEngine
from models.churn_prediction import ChurnPredictionEngine
from models.inventory_optimization import InventoryOptimizationEngine
from models.pricing_optimization import PricingOptimizationEngine
from models.fraud_detection import FraudDetectionEngine
from models.marketing_attribution import MarketingAttributionEngine
from visualization.executive_dashboard import ExecutiveDashboard


class EnhancedBusinessIntelligencePlatform:
    """
    Comprehensive Business Intelligence Platform for Retail Excellence

    Addresses Critical Business Issues:
    1. Customer Retention & Churn Prevention
    2. Inventory Optimization & Demand Forecasting
    3. Dynamic Pricing & Revenue Optimization
    4. Fraud Detection & Risk Management
    5. Marketing Attribution & ROI Analysis
    6. Executive Decision Support

    This platform demonstrates enterprise-level data science capabilities
    for solving real-world retail business challenges.
    """

    def __init__(self, data_dir="data", results_dir="reports"):
        """Initialize the comprehensive business intelligence platform"""
        self.data_dir = data_dir
        self.results_dir = results_dir

        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Data paths for new structure
        self.data_paths = {
            "transactions": f"{data_dir}/transactions_real.csv",
            "customers": f"{data_dir}/customers_real.csv",
            "products": f"{data_dir}/products_real.csv",
            "digital_events": f"{data_dir}/digital_events_real.csv",
        }

        self.data_source = "UCI Online Retail + Wholesale Customers Dataset"

        # Initialize all engines
        self.engines = {}
        self.results = {}

    def _check_data_exists(self):
        """Check if required data files exist"""
        missing_files = []
        for name, path in self.data_paths.items():
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            print(f"❌ Missing data files: {missing_files}")
            print("💡 Run the data loader first: python src/data/data_loader.py")
            return False
        return True

    def initialize_all_engines(self):
        """Initialize all business intelligence engines"""
        print("🚀 INITIALIZING COMPREHENSIVE BUSINESS INTELLIGENCE PLATFORM")
        print("=" * 70)

        if not self._check_data_exists():
            return False

        # Original engines
        print("📊 Initializing core analytics engines...")
        self.engines["segmentation"] = CustomerSegmentation(
            self.data_paths["transactions"], self.data_paths["customers"]
        )

        self.engines["recommendations"] = RecommendationEngine(
            self.data_paths["transactions"],
            self.data_paths["customers"],
            self.data_paths["products"],
        )

        # New business intelligence engines
        print("🧠 Initializing advanced business intelligence engines...")

        self.engines["churn_prediction"] = ChurnPredictionEngine(
            self.data_paths["transactions"], self.data_paths["customers"]
        )

        self.engines["inventory_optimization"] = InventoryOptimizationEngine(
            self.data_paths["transactions"], self.data_paths["products"]
        )

        self.engines["pricing_optimization"] = PricingOptimizationEngine(
            self.data_paths["transactions"], self.data_paths["products"]
        )

        self.engines["fraud_detection"] = FraudDetectionEngine(
            self.data_paths["transactions"], self.data_paths["customers"]
        )

        self.engines["marketing_attribution"] = MarketingAttributionEngine(
            self.data_paths["transactions"],
            self.data_paths["customers"],
            self.data_paths["digital_events"],
        )

        self.engines["executive_dashboard"] = ExecutiveDashboard()

        print("✅ All engines initialized successfully!")
        return True

    def run_customer_analytics(self):
        """Run comprehensive customer analytics"""
        print("\n👥 CUSTOMER ANALYTICS & SEGMENTATION")
        print("=" * 50)

        # Customer Segmentation
        print("🔄 Running customer segmentation analysis...")
        segmentation = self.engines["segmentation"]
        segmentation.prepare_data()
        segmentation.calculate_rfm()
        segments = segmentation.advanced_clustering()
        segmentation.generate_insights()

        # Save segmentation results
        segments.to_csv(
            f"{self.results_dir}/customer_segments_enhanced.csv", index=False
        )

        # Churn Prediction
        print("🔄 Running churn prediction analysis...")
        churn_engine = self.engines["churn_prediction"]
        churn_engine.prepare_churn_features()
        churn_engine.train_churn_model()
        strategies, predictions = churn_engine.generate_retention_strategies()

        # Save churn results
        predictions.to_csv(
            f"{self.results_dir}/churn_predictions_enhanced.csv", index=False
        )

        self.results["customer_analytics"] = {
            "segments": len(segments["segment_name"].unique())
            if "segment_name" in segments.columns
            else len(segments["final_cluster"].unique()),
            "churn_rate": predictions["is_churned"].mean(),
            "high_risk_customers": len(
                predictions[predictions["risk_category"] == "High Risk"]
            ),
            "retention_strategies": strategies,
        }

        print("✅ Customer analytics completed!")

    def run_complete_analysis(self):
        """Run the complete business intelligence analysis"""
        print("🚀 STARTING COMPREHENSIVE BUSINESS INTELLIGENCE ANALYSIS")
        print("=" * 70)

        start_time = datetime.now()

        print(f"📊 Data Source: {self.data_source}")
        print("=" * 70)

        # Initialize all engines
        if not self.initialize_all_engines():
            print("❌ Failed to initialize engines. Please check data files.")
            return None

        # Run customer analytics
        try:
            self.run_customer_analytics()
        except Exception as e:
            print(f"⚠️  Customer analytics failed: {e}")

        end_time = datetime.now()
        analysis_duration = (end_time - start_time).total_seconds()

        # Final summary
        print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print("=" * 50)
        print(f"⏱️  Analysis Duration: {analysis_duration:.1f} seconds")
        print(f"📊 Modules Executed: {len(self.results)}")
        print(f"📁 Results Directory: {self.results_dir}")

        # Print key insights
        print(f"\n🔍 KEY INSIGHTS:")
        if "customer_analytics" in self.results:
            print(
                f"   👥 Customer Segments: {self.results['customer_analytics']['segments']}"
            )
            print(
                f"   ⚠️  High-Risk Customers: {self.results['customer_analytics']['high_risk_customers']}"
            )
            print(
                f"   📈 Churn Rate: {self.results['customer_analytics']['churn_rate']:.2%}"
            )

        print(f"\n✅ All results saved to: {self.results_dir}/")

        return self.results


def main():
    """Main execution function"""
    # Initialize and run comprehensive business intelligence platform
    platform = EnhancedBusinessIntelligencePlatform()
    results = platform.run_complete_analysis()

    return results


if __name__ == "__main__":
    main()
