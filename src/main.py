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
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.logger import setup_logging, get_logger
from utils.config import config
from utils.data_validator import validate_data_files, generate_validation_report
from utils.error_handler import error_handler, graceful_shutdown, monitored_execution

# Import models
from models.customer_segmentation import CustomerSegmentation
from models.recommendation_engine import RecommendationEngine
from models.churn_prediction import ChurnPredictionEngine
from models.inventory_optimization import InventoryOptimizationEngine
from models.pricing_optimization import PricingOptimizationEngine
from models.fraud_detection import FraudDetectionEngine
from models.marketing_attribution import MarketingAttributionEngine
from visualization.executive_dashboard import ExecutiveDashboard

# Initialize logging
logger = get_logger(__name__)


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
            logger.error(f"Missing data files: {missing_files}")
            logger.info("Run the data loader first: python src/data/data_loader.py")
            return False
        return True

    @monitored_execution(context="initialization")
    def initialize_all_engines(self):
        """Initialize all business intelligence engines"""
        logger.info("Initializing Comprehensive Business Intelligence Platform")
        logger.info("=" * 70)

        if not self._check_data_exists():
            return False

        # Validate data before processing
        logger.info("Validating data files...")
        validation_results = validate_data_files(self.data_dir)
        
        total_errors = sum(len(r.errors) for r in validation_results.values())
        if total_errors > 0:
            logger.warning(f"Data validation found {total_errors} errors")
            validation_report = generate_validation_report(validation_results)
            logger.debug(validation_report)

        # Original engines
        logger.info("Initializing core analytics engines...")
        self.engines["segmentation"] = CustomerSegmentation(
            self.data_paths["transactions"], self.data_paths["customers"]
        )

        self.engines["recommendations"] = RecommendationEngine(
            self.data_paths["transactions"],
            self.data_paths["customers"],
            self.data_paths["products"],
        )

        # New business intelligence engines
        logger.info("Initializing advanced business intelligence engines...")

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

        logger.info("All engines initialized successfully")
        return True

    @error_handler(default_return=None)
    @monitored_execution(context="customer_analytics")
    def run_customer_analytics(self):
        """Run comprehensive customer analytics"""
        logger.info("Customer Analytics & Segmentation")
        logger.info("=" * 50)

        # Customer Segmentation
        logger.info("Running customer segmentation analysis...")
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
        logger.info("Running churn prediction analysis...")
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

        logger.info("Customer analytics completed")

    @graceful_shutdown
    @monitored_execution(context="complete_analysis")
    def run_complete_analysis(self):
        """Run the complete business intelligence analysis"""
        logger.info("Starting Comprehensive Business Intelligence Analysis")
        logger.info("=" * 70)

        start_time = datetime.now()

        logger.info(f"Data Source: {self.data_source}")
        logger.info("=" * 70)

        # Initialize all engines
        if not self.initialize_all_engines():
            logger.error("Failed to initialize engines. Please check data files.")
            return None

        # Run customer analytics
        try:
            self.run_customer_analytics()
        except Exception as e:
            logger.error(f"Customer analytics failed: {e}", exc_info=True)

        end_time = datetime.now()
        analysis_duration = (end_time - start_time).total_seconds()

        # Final summary
        logger.info("Comprehensive Analysis Completed")
        logger.info("=" * 50)
        logger.info(f"Analysis Duration: {analysis_duration:.1f} seconds")
        logger.info(f"Modules Executed: {len(self.results)}")
        logger.info(f"Results Directory: {self.results_dir}")

        # Print key insights
        logger.info("Key Insights:")
        if "customer_analytics" in self.results:
            logger.info(f"  Customer Segments: {self.results['customer_analytics']['segments']}")
            logger.info(f"  High-Risk Customers: {self.results['customer_analytics']['high_risk_customers']}")
            logger.info(f"  Churn Rate: {self.results['customer_analytics']['churn_rate']:.2%}")

        logger.info(f"All results saved to: {self.results_dir}/")

        return self.results


def main():
    """Main execution function"""
    # Initialize logging
    setup_logging(log_level=config.log_level, log_dir="logs")
    
    logger.info("=" * 70)
    logger.info("Enterprise Business Intelligence Platform v2.0.0")
    logger.info("=" * 70)
    
    # Initialize and run comprehensive business intelligence platform
    platform = EnhancedBusinessIntelligencePlatform(
        data_dir=config.data_dir,
        results_dir=config.results_dir
    )
    results = platform.run_complete_analysis()

    return results


if __name__ == "__main__":
    main()
