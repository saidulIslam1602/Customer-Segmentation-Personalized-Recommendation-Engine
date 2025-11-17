"""
Performance Metrics Generator - Enterprise-Grade Performance Analytics
Generates comprehensive performance metrics, benchmarks, and KPIs for all BI modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings

warnings.filterwarnings("ignore")


class PerformanceMetricsGenerator:
    """
    Enterprise Performance Metrics Generator

    Features:
    - Model Performance Analytics
    - Business KPI Benchmarking
    - Statistical Performance Testing
    - ROI and Revenue Impact Analysis
    - Operational Efficiency Metrics
    - Customer Experience Metrics
    - Real-time Performance Monitoring
    - Comparative Analysis and Benchmarking
    """

    def __init__(self, results_dir="reports/performance_metrics"):
        """Initialize performance metrics generator"""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Performance data storage
        self.model_metrics = {}
        self.business_metrics = {}
        self.operational_metrics = {}
        self.customer_metrics = {}
        self.financial_metrics = {}

        # Benchmarks (industry standards)
        self.benchmarks = {
            "churn_prediction": {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.75,
                "f1_score": 0.77,
                "auc_roc": 0.85,
            },
            "recommendation_engine": {
                "precision_at_k": 0.15,
                "recall_at_k": 0.25,
                "ndcg_at_k": 0.30,
                "coverage": 0.80,
                "diversity": 0.70,
            },
            "segmentation": {
                "silhouette_score": 0.50,
                "calinski_harabasz": 100,
                "davies_bouldin": 1.0,
                "segment_stability": 0.85,
            },
            "business_kpis": {
                "customer_retention_rate": 0.80,
                "customer_lifetime_value_growth": 0.15,
                "revenue_per_customer": 1000,
                "conversion_rate": 0.05,
                "average_order_value_growth": 0.10,
            },
        }

        print(" Performance Metrics Generator initialized")
        print(f"    Results directory: {results_dir}")
        print(f"    Benchmarks loaded: {len(self.benchmarks)} categories")

    def evaluate_churn_prediction_performance(
        self, model, X_test, y_test, X_train=None, y_train=None
    ):
        """Comprehensive churn prediction model evaluation"""
        print(" Evaluating Churn Prediction Performance...")

        # Basic predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Handle single class scenario
        if y_pred_proba.shape[1] == 1:
            y_pred_proba_positive = y_pred_proba[:, 0]
        else:
            y_pred_proba_positive = y_pred_proba[:, 1]

        # Core metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # AUC-ROC (only if more than one class)
        if len(np.unique(y_test)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_test, y_pred_proba_positive)
        else:
            metrics["auc_roc"] = 0.5  # Random performance for single class

        # Business impact metrics
        if len(np.unique(y_test)) > 1:
            # True/False positives and negatives
            from sklearn.metrics import confusion_matrix

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Business metrics
            metrics.update(
                {
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "churn_detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
                    "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                }
            )

        # Cross-validation performance
        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            )
            metrics.update(
                {
                    "cv_mean_accuracy": cv_scores.mean(),
                    "cv_std_accuracy": cv_scores.std(),
                    "cv_scores": cv_scores.tolist(),
                }
            )

        # Performance vs benchmarks
        benchmark_comparison = {}
        for metric, value in metrics.items():
            if metric in self.benchmarks["churn_prediction"]:
                benchmark = self.benchmarks["churn_prediction"][metric]
                benchmark_comparison[f"{metric}_vs_benchmark"] = {
                    "actual": value,
                    "benchmark": benchmark,
                    "performance": "Above" if value > benchmark else "Below",
                    "difference": value - benchmark,
                }

        metrics["benchmark_comparison"] = benchmark_comparison

        # Model confidence analysis
        confidence_analysis = {
            "high_confidence_predictions": np.sum(np.max(y_pred_proba, axis=1) > 0.8),
            "low_confidence_predictions": np.sum(np.max(y_pred_proba, axis=1) < 0.6),
            "average_prediction_confidence": np.mean(np.max(y_pred_proba, axis=1)),
        }
        metrics["confidence_analysis"] = confidence_analysis

        self.model_metrics["churn_prediction"] = metrics

        print(f" Churn Prediction Metrics:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"     F1-Score: {metrics['f1_score']:.4f}")
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")

        return metrics

    def evaluate_recommendation_performance(
        self, recommendations_df, actual_purchases_df, k=10
    ):
        """Evaluate recommendation engine performance"""
        print(" Evaluating Recommendation Engine Performance...")

        metrics = {}

        # Precision@K and Recall@K
        precision_scores = []
        recall_scores = []

        for customer_id in recommendations_df["customer_id"].unique():
            # Get recommendations for this customer
            customer_recs = (
                recommendations_df[recommendations_df["customer_id"] == customer_id][
                    "product_id"
                ]
                .head(k)
                .tolist()
            )

            # Get actual purchases for this customer
            actual_purchases = actual_purchases_df[
                actual_purchases_df["customer_id"] == customer_id
            ]["product_id"].tolist()

            if len(customer_recs) > 0 and len(actual_purchases) > 0:
                # Calculate precision@k
                relevant_recommendations = set(customer_recs) & set(actual_purchases)
                precision = len(relevant_recommendations) / len(customer_recs)
                precision_scores.append(precision)

                # Calculate recall@k
                recall = len(relevant_recommendations) / len(actual_purchases)
                recall_scores.append(recall)

        metrics.update(
            {
                f"precision_at_{k}": np.mean(precision_scores)
                if precision_scores
                else 0,
                f"recall_at_{k}": np.mean(recall_scores) if recall_scores else 0,
                f"precision_scores_distribution": {
                    "mean": np.mean(precision_scores) if precision_scores else 0,
                    "std": np.std(precision_scores) if precision_scores else 0,
                    "min": np.min(precision_scores) if precision_scores else 0,
                    "max": np.max(precision_scores) if precision_scores else 0,
                },
            }
        )

        # Coverage (what percentage of items are recommended)
        total_products = actual_purchases_df["product_id"].nunique()
        recommended_products = recommendations_df["product_id"].nunique()
        metrics["coverage"] = (
            recommended_products / total_products if total_products > 0 else 0
        )

        # Diversity (average pairwise distance between recommendations)
        # Simplified diversity metric based on product categories
        if "category" in recommendations_df.columns:
            diversity_scores = []
            for customer_id in recommendations_df["customer_id"].unique():
                customer_recs = (
                    recommendations_df[
                        recommendations_df["customer_id"] == customer_id
                    ]["category"]
                    .head(k)
                    .tolist()
                )

                if len(customer_recs) > 1:
                    unique_categories = len(set(customer_recs))
                    diversity = unique_categories / len(customer_recs)
                    diversity_scores.append(diversity)

            metrics["diversity"] = np.mean(diversity_scores) if diversity_scores else 0
        else:
            metrics["diversity"] = 0.5  # Default assumption

        # Business impact metrics
        if "score" in recommendations_df.columns:
            metrics.update(
                {
                    "average_recommendation_score": recommendations_df["score"].mean(),
                    "recommendation_score_std": recommendations_df["score"].std(),
                    "high_confidence_recommendations": len(
                        recommendations_df[recommendations_df["score"] > 0.8]
                    ),
                }
            )

        # Benchmark comparison
        benchmark_comparison = {}
        for metric, value in metrics.items():
            if metric in self.benchmarks["recommendation_engine"]:
                benchmark = self.benchmarks["recommendation_engine"][metric]
                benchmark_comparison[f"{metric}_vs_benchmark"] = {
                    "actual": value,
                    "benchmark": benchmark,
                    "performance": "Above" if value > benchmark else "Below",
                    "difference": value - benchmark,
                }

        metrics["benchmark_comparison"] = benchmark_comparison

        self.model_metrics["recommendation_engine"] = metrics

        print(f" Recommendation Engine Metrics:")
        print(f"    Precision@{k}: {metrics[f'precision_at_{k}']:.4f}")
        print(f"    Recall@{k}: {metrics[f'recall_at_{k}']:.4f}")
        print(f"    Coverage: {metrics['coverage']:.4f}")
        print(f"    Diversity: {metrics['diversity']:.4f}")

        return metrics

    def evaluate_segmentation_performance(self, segmentation_data, features_data):
        """Evaluate customer segmentation performance"""
        print(" Evaluating Customer Segmentation Performance...")

        # Extract cluster labels and features
        if "final_cluster" in segmentation_data.columns:
            cluster_labels = segmentation_data["final_cluster"]
        elif "segment_name" in segmentation_data.columns:
            # Convert segment names to numeric labels
            unique_segments = segmentation_data["segment_name"].unique()
            segment_mapping = {segment: i for i, segment in enumerate(unique_segments)}
            cluster_labels = segmentation_data["segment_name"].map(segment_mapping)
        else:
            print("  No cluster labels found in segmentation data")
            return {}

        # Prepare features for evaluation
        feature_columns = ["recency", "frequency", "monetary"]
        available_features = [
            col for col in feature_columns if col in features_data.columns
        ]

        if not available_features:
            print("  No suitable features found for segmentation evaluation")
            return {}

        X = features_data[available_features].fillna(0)

        # Scale features for proper distance calculations
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        metrics = {}

        # Internal clustering metrics
        if len(np.unique(cluster_labels)) > 1:
            metrics.update(
                {
                    "silhouette_score": silhouette_score(X_scaled, cluster_labels),
                    "calinski_harabasz_score": calinski_harabasz_score(
                        X_scaled, cluster_labels
                    ),
                    "davies_bouldin_score": davies_bouldin_score(
                        X_scaled, cluster_labels
                    ),
                }
            )
        else:
            metrics.update(
                {
                    "silhouette_score": 0,
                    "calinski_harabasz_score": 0,
                    "davies_bouldin_score": float("inf"),
                }
            )

        # Segment analysis
        segment_analysis = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = X[cluster_mask]

            segment_analysis[f"cluster_{cluster_id}"] = {
                "size": int(np.sum(cluster_mask)),
                "percentage": float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                "feature_means": cluster_data.mean().to_dict(),
                "feature_stds": cluster_data.std().to_dict(),
            }

        metrics["segment_analysis"] = segment_analysis

        # Business value metrics
        if "monetary" in features_data.columns:
            total_value = features_data["monetary"].sum()
            value_by_segment = {}

            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_value = features_data[cluster_mask]["monetary"].sum()
                value_by_segment[f"cluster_{cluster_id}"] = {
                    "total_value": float(cluster_value),
                    "value_percentage": float(cluster_value / total_value * 100),
                    "avg_customer_value": float(
                        features_data[cluster_mask]["monetary"].mean()
                    ),
                }

            metrics["value_by_segment"] = value_by_segment

        # Segment stability (if historical data available)
        metrics[
            "segment_stability"
        ] = 0.85  # Placeholder - would need historical comparison

        # Benchmark comparison
        benchmark_comparison = {}
        for metric, value in metrics.items():
            if metric in self.benchmarks["segmentation"]:
                benchmark = self.benchmarks["segmentation"][metric]
                benchmark_comparison[f"{metric}_vs_benchmark"] = {
                    "actual": value,
                    "benchmark": benchmark,
                    "performance": "Above" if value > benchmark else "Below",
                    "difference": value - benchmark,
                }

        metrics["benchmark_comparison"] = benchmark_comparison

        self.model_metrics["segmentation"] = metrics

        print(f" Segmentation Metrics:")
        print(f"    Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"    Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
        print(f"    Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
        print(f"    Number of Segments: {len(np.unique(cluster_labels))}")

        return metrics

    def calculate_business_performance_metrics(self, transactions_df, customers_df):
        """Calculate comprehensive business performance metrics"""
        print(" Calculating Business Performance Metrics...")

        # Prepare data
        transactions_df["transaction_date"] = pd.to_datetime(
            transactions_df["transaction_date"]
        )
        current_date = transactions_df["transaction_date"].max()

        metrics = {}

        # Revenue Metrics
        total_revenue = transactions_df["total_amount"].sum()
        monthly_revenue = transactions_df.groupby(
            transactions_df["transaction_date"].dt.to_period("M")
        )["total_amount"].sum()

        if len(monthly_revenue) >= 2:
            revenue_growth = (
                monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]
            ) / monthly_revenue.iloc[-2]
        else:
            revenue_growth = 0

        metrics["revenue"] = {
            "total_revenue": float(total_revenue),
            "monthly_revenue_growth": float(revenue_growth),
            "average_monthly_revenue": float(monthly_revenue.mean()),
            "revenue_volatility": float(monthly_revenue.std() / monthly_revenue.mean())
            if monthly_revenue.mean() > 0
            else 0,
        }

        # Customer Metrics
        total_customers = transactions_df["customer_id"].nunique()
        customer_frequency = transactions_df.groupby("customer_id").size()
        repeat_customers = (customer_frequency > 1).sum()

        # Customer Lifetime Value
        customer_values = transactions_df.groupby("customer_id")["total_amount"].sum()
        avg_clv = customer_values.mean()

        # Customer retention (customers active in last 30 days)
        recent_date = current_date - timedelta(days=30)
        recent_customers = transactions_df[
            transactions_df["transaction_date"] >= recent_date
        ]["customer_id"].nunique()

        metrics["customer"] = {
            "total_customers": int(total_customers),
            "repeat_customer_rate": float(repeat_customers / total_customers)
            if total_customers > 0
            else 0,
            "average_clv": float(avg_clv),
            "customer_retention_rate": float(recent_customers / total_customers)
            if total_customers > 0
            else 0,
            "customer_acquisition_rate": 0.05,  # Placeholder - would need historical data
        }

        # Transaction Metrics
        avg_order_value = transactions_df["total_amount"].mean()
        transaction_frequency = (
            len(transactions_df) / total_customers if total_customers > 0 else 0
        )

        metrics["transactions"] = {
            "total_transactions": len(transactions_df),
            "average_order_value": float(avg_order_value),
            "transaction_frequency_per_customer": float(transaction_frequency),
            "conversion_rate": 0.05,  # Placeholder - would need traffic data
        }

        # Product Performance
        product_performance = (
            transactions_df.groupby("product_id")
            .agg({"total_amount": "sum", "quantity": "sum", "customer_id": "nunique"})
            .reset_index()
        )

        top_products = product_performance.nlargest(10, "total_amount")

        metrics["products"] = {
            "total_products_sold": int(transactions_df["quantity"].sum()),
            "unique_products": int(transactions_df["product_id"].nunique()),
            "avg_products_per_transaction": float(
                transactions_df.groupby("transaction_id")["quantity"].sum().mean()
            ),
            "top_product_revenue_share": float(
                top_products["total_amount"].sum() / total_revenue
            )
            if total_revenue > 0
            else 0,
        }

        # Operational Efficiency
        date_range = (current_date - transactions_df["transaction_date"].min()).days
        daily_transaction_volume = (
            len(transactions_df) / date_range if date_range > 0 else 0
        )

        metrics["operational"] = {
            "daily_transaction_volume": float(daily_transaction_volume),
            "revenue_per_transaction": float(total_revenue / len(transactions_df))
            if len(transactions_df) > 0
            else 0,
            "customer_service_efficiency": 0.85,  # Placeholder
            "order_fulfillment_rate": 0.95,  # Placeholder
        }

        # Benchmark comparisons
        benchmark_comparison = {}
        for category, category_metrics in metrics.items():
            if category in ["customer", "transactions"]:
                for metric, value in category_metrics.items():
                    benchmark_key = (
                        f"{category}_{metric}" if category == "customer" else metric
                    )
                    if benchmark_key in self.benchmarks["business_kpis"]:
                        benchmark = self.benchmarks["business_kpis"][benchmark_key]
                        benchmark_comparison[f"{metric}_vs_benchmark"] = {
                            "actual": value,
                            "benchmark": benchmark,
                            "performance": "Above" if value > benchmark else "Below",
                            "difference": value - benchmark,
                        }

        metrics["benchmark_comparison"] = benchmark_comparison

        self.business_metrics = metrics

        print(f" Business Performance Metrics:")
        print(f"    Total Revenue: ${metrics['revenue']['total_revenue']:,.2f}")
        print(f"    Total Customers: {metrics['customer']['total_customers']:,}")
        print(
            f"    Repeat Customer Rate: {metrics['customer']['repeat_customer_rate']:.2%}"
        )
        print(
            f"    Average Order Value: ${metrics['transactions']['average_order_value']:.2f}"
        )

        return metrics

    def calculate_roi_and_impact_metrics(
        self, baseline_revenue, current_revenue, implementation_cost=50000
    ):
        """Calculate ROI and business impact metrics"""
        print(" Calculating ROI and Business Impact...")

        # Revenue impact
        revenue_increase = current_revenue - baseline_revenue
        revenue_lift_percentage = (
            (revenue_increase / baseline_revenue * 100) if baseline_revenue > 0 else 0
        )

        # ROI calculation
        roi = (
            (revenue_increase - implementation_cost) / implementation_cost * 100
            if implementation_cost > 0
            else 0
        )

        # Payback period (months)
        monthly_revenue_increase = revenue_increase / 12  # Assume annual figures
        payback_months = (
            implementation_cost / monthly_revenue_increase
            if monthly_revenue_increase > 0
            else float("inf")
        )

        # Net Present Value (simplified, 10% discount rate)
        discount_rate = 0.10
        years = 3
        npv = (
            sum(
                [
                    revenue_increase / (1 + discount_rate) ** year
                    for year in range(1, years + 1)
                ]
            )
            - implementation_cost
        )

        metrics = {
            "financial_impact": {
                "baseline_revenue": float(baseline_revenue),
                "current_revenue": float(current_revenue),
                "revenue_increase": float(revenue_increase),
                "revenue_lift_percentage": float(revenue_lift_percentage),
                "implementation_cost": float(implementation_cost),
            },
            "roi_metrics": {
                "roi_percentage": float(roi),
                "payback_period_months": float(payback_months)
                if payback_months != float("inf")
                else None,
                "net_present_value": float(npv),
                "break_even_achieved": roi > 0,
            },
            "business_value": {
                "annual_value_created": float(revenue_increase),
                "monthly_value_created": float(revenue_increase / 12),
                "value_multiple": float(current_revenue / baseline_revenue)
                if baseline_revenue > 0
                else 1,
                "efficiency_gain": float(revenue_lift_percentage / 100),
            },
        }

        self.financial_metrics = metrics

        print(f" ROI and Impact Metrics:")
        print(f"    Revenue Increase: ${revenue_increase:,.2f}")
        print(f"    Revenue Lift: {revenue_lift_percentage:.1f}%")
        print(f"    ROI: {roi:.1f}%")
        print(
            f"     Payback Period: {payback_months:.1f} months"
            if payback_months != float("inf")
            else "     Payback Period: Not achieved"
        )

        return metrics

    def generate_performance_dashboard(self):
        """Generate comprehensive performance dashboard"""
        print(" Generating Performance Dashboard...")

        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Enterprise Performance Dashboard", fontsize=16, fontweight="bold")

        # 1. Model Performance Comparison
        if self.model_metrics:
            model_names = []
            accuracy_scores = []

            for model, metrics in self.model_metrics.items():
                model_names.append(model.replace("_", " ").title())
                if "accuracy" in metrics:
                    accuracy_scores.append(metrics["accuracy"])
                elif "silhouette_score" in metrics:
                    accuracy_scores.append(metrics["silhouette_score"])
                else:
                    accuracy_scores.append(0.5)

            axes[0, 0].bar(
                model_names, accuracy_scores, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            axes[0, 0].set_title("Model Performance Scores")
            axes[0, 0].set_ylabel("Score")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Business Metrics Overview
        if self.business_metrics:
            metrics_names = [
                "Revenue Growth",
                "Customer Retention",
                "AOV Growth",
                "Conversion Rate",
            ]
            metrics_values = [
                self.business_metrics["revenue"].get("monthly_revenue_growth", 0) * 100,
                self.business_metrics["customer"].get("customer_retention_rate", 0)
                * 100,
                10,  # Placeholder
                self.business_metrics["transactions"].get("conversion_rate", 0) * 100,
            ]

            colors = ["green" if v > 0 else "red" for v in metrics_values]
            axes[0, 1].bar(metrics_names, metrics_values, color=colors)
            axes[0, 1].set_title("Business KPIs (%)")
            axes[0, 1].set_ylabel("Percentage")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. ROI Analysis
        if self.financial_metrics:
            roi_data = self.financial_metrics["roi_metrics"]
            roi_value = roi_data.get("roi_percentage", 0)

            axes[0, 2].pie(
                [max(0, roi_value), max(0, 100 - roi_value)],
                labels=["ROI Achieved", "Target Remaining"],
                colors=["#2ca02c", "#d62728"],
                autopct="%1.1f%%",
            )
            axes[0, 2].set_title(f"ROI Achievement: {roi_value:.1f}%")

        # 4. Customer Segmentation Distribution
        if "segmentation" in self.model_metrics:
            segment_data = self.model_metrics["segmentation"].get(
                "segment_analysis", {}
            )
            if segment_data:
                segments = list(segment_data.keys())
                sizes = [segment_data[seg]["size"] for seg in segments]

                axes[1, 0].pie(sizes, labels=segments, autopct="%1.1f%%")
                axes[1, 0].set_title("Customer Segment Distribution")

        # 5. Revenue Trend (placeholder)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        revenue_trend = [100, 105, 110, 108, 115, 120]  # Placeholder data

        axes[1, 1].plot(months, revenue_trend, marker="o", linewidth=2, color="#1f77b4")
        axes[1, 1].set_title("Revenue Trend (Index)")
        axes[1, 1].set_ylabel("Revenue Index")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Performance vs Benchmarks
        if self.model_metrics:
            benchmark_performance = []
            model_names_bench = []

            for model, metrics in self.model_metrics.items():
                if "benchmark_comparison" in metrics:
                    above_benchmark = sum(
                        1
                        for comp in metrics["benchmark_comparison"].values()
                        if comp["performance"] == "Above"
                    )
                    total_benchmarks = len(metrics["benchmark_comparison"])
                    if total_benchmarks > 0:
                        benchmark_performance.append(
                            above_benchmark / total_benchmarks * 100
                        )
                        model_names_bench.append(model.replace("_", " ").title())

            if benchmark_performance:
                axes[1, 2].bar(
                    model_names_bench, benchmark_performance, color="#2ca02c"
                )
                axes[1, 2].set_title("Performance vs Industry Benchmarks (%)")
                axes[1, 2].set_ylabel("% Above Benchmark")
                axes[1, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save dashboard
        dashboard_path = f'{self.results_dir}/performance_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f" Performance dashboard saved: {dashboard_path}")
        return dashboard_path

    def generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        print(" Generating Comprehensive Performance Report...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compile all metrics
        comprehensive_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "Comprehensive Performance Analysis",
                "version": "1.0",
            },
            "executive_summary": {
                "total_models_evaluated": len(self.model_metrics),
                "business_metrics_calculated": len(self.business_metrics),
                "overall_performance_score": self._calculate_overall_performance_score(),
                "key_achievements": self._identify_key_achievements(),
                "improvement_areas": self._identify_improvement_areas(),
            },
            "model_performance": self.model_metrics,
            "business_performance": self.business_metrics,
            "financial_impact": self.financial_metrics,
            "recommendations": self._generate_performance_recommendations(),
        }

        # Save comprehensive report
        report_path = (
            f"{self.results_dir}/comprehensive_performance_report_{timestamp}.json"
        )
        with open(report_path, "w") as f:
            json.dump(comprehensive_report, f, indent=2, default=str)

        # Generate summary report
        summary_report = self._generate_summary_report(comprehensive_report)
        summary_path = f"{self.results_dir}/performance_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(summary_report)

        print(f" Comprehensive report saved: {report_path}")
        print(f" Summary report saved: {summary_path}")

        return comprehensive_report

    def _calculate_overall_performance_score(self):
        """Calculate overall performance score (0-100)"""
        scores = []

        # Model performance scores
        for model, metrics in self.model_metrics.items():
            if "accuracy" in metrics:
                scores.append(metrics["accuracy"] * 100)
            elif "silhouette_score" in metrics:
                scores.append(
                    (metrics["silhouette_score"] + 1) * 50
                )  # Normalize to 0-100

        # Business performance scores
        if self.business_metrics:
            if "customer" in self.business_metrics:
                retention_rate = self.business_metrics["customer"].get(
                    "customer_retention_rate", 0
                )
                scores.append(retention_rate * 100)

        # Financial performance scores
        if self.financial_metrics:
            roi = self.financial_metrics["roi_metrics"].get("roi_percentage", 0)
            scores.append(min(100, max(0, roi)))  # Cap at 100, floor at 0

        return np.mean(scores) if scores else 50

    def _identify_key_achievements(self):
        """Identify key performance achievements"""
        achievements = []

        # Check model performance
        for model, metrics in self.model_metrics.items():
            if "benchmark_comparison" in metrics:
                above_benchmark = sum(
                    1
                    for comp in metrics["benchmark_comparison"].values()
                    if comp["performance"] == "Above"
                )
                if above_benchmark > 0:
                    achievements.append(
                        f"{model.replace('_', ' ').title()}: {above_benchmark} metrics above industry benchmark"
                    )

        # Check business performance
        if self.business_metrics:
            retention_rate = self.business_metrics["customer"].get(
                "customer_retention_rate", 0
            )
            if retention_rate > 0.8:
                achievements.append(
                    f"Excellent customer retention rate: {retention_rate:.1%}"
                )

        # Check financial performance
        if self.financial_metrics:
            roi = self.financial_metrics["roi_metrics"].get("roi_percentage", 0)
            if roi > 100:
                achievements.append(f"Outstanding ROI achieved: {roi:.1f}%")

        return achievements

    def _identify_improvement_areas(self):
        """Identify areas for improvement"""
        improvements = []

        # Check model performance
        for model, metrics in self.model_metrics.items():
            if "benchmark_comparison" in metrics:
                below_benchmark = sum(
                    1
                    for comp in metrics["benchmark_comparison"].values()
                    if comp["performance"] == "Below"
                )
                if below_benchmark > 0:
                    improvements.append(
                        f"{model.replace('_', ' ').title()}: {below_benchmark} metrics below benchmark"
                    )

        # Check business metrics
        if self.business_metrics:
            conversion_rate = self.business_metrics["transactions"].get(
                "conversion_rate", 0
            )
            if conversion_rate < 0.05:
                improvements.append("Conversion rate below industry standard (5%)")

        return improvements

    def _generate_performance_recommendations(self):
        """Generate performance improvement recommendations"""
        recommendations = []

        # Model-specific recommendations
        for model, metrics in self.model_metrics.items():
            if model == "churn_prediction" and metrics.get("recall", 0) < 0.75:
                recommendations.append(
                    {
                        "category": "Model Performance",
                        "priority": "High",
                        "recommendation": "Improve churn prediction recall through feature engineering and class balancing",
                        "expected_impact": "Reduce customer churn by 10-15%",
                    }
                )

            if (
                model == "recommendation_engine"
                and metrics.get("precision_at_10", 0) < 0.15
            ):
                recommendations.append(
                    {
                        "category": "Model Performance",
                        "priority": "Medium",
                        "recommendation": "Enhance recommendation precision through collaborative filtering improvements",
                        "expected_impact": "Increase recommendation relevance by 20%",
                    }
                )

        # Business recommendations
        if self.business_metrics:
            retention_rate = self.business_metrics["customer"].get(
                "customer_retention_rate", 0
            )
            if retention_rate < 0.8:
                recommendations.append(
                    {
                        "category": "Business Performance",
                        "priority": "High",
                        "recommendation": "Implement targeted retention campaigns for at-risk customers",
                        "expected_impact": "Improve retention rate to 85%+",
                    }
                )

        return recommendations

    def _generate_summary_report(self, comprehensive_report):
        """Generate human-readable summary report"""
        summary = f"""
ENTERPRISE PERFORMANCE METRICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

EXECUTIVE SUMMARY
Overall Performance Score: {comprehensive_report['executive_summary']['overall_performance_score']:.1f}/100

KEY ACHIEVEMENTS:
"""

        for achievement in comprehensive_report["executive_summary"][
            "key_achievements"
        ]:
            summary += f" {achievement}\n"

        summary += f"""
IMPROVEMENT AREAS:
"""

        for improvement in comprehensive_report["executive_summary"][
            "improvement_areas"
        ]:
            summary += f"  {improvement}\n"

        summary += f"""
MODEL PERFORMANCE SUMMARY:
"""

        for model, metrics in comprehensive_report["model_performance"].items():
            summary += f"\n{model.replace('_', ' ').title()}:\n"
            if "accuracy" in metrics:
                summary += f"  Accuracy: {metrics['accuracy']:.4f}\n"
            if "precision" in metrics:
                summary += f"  Precision: {metrics['precision']:.4f}\n"
            if "recall" in metrics:
                summary += f"  Recall: {metrics['recall']:.4f}\n"

        if comprehensive_report["financial_impact"]:
            roi = comprehensive_report["financial_impact"]["roi_metrics"].get(
                "roi_percentage", 0
            )
            summary += f"""
FINANCIAL IMPACT:
ROI: {roi:.1f}%
Revenue Impact: ${comprehensive_report['financial_impact']['financial_impact'].get('revenue_increase', 0):,.2f}
"""

        summary += f"""
RECOMMENDATIONS:
"""

        for rec in comprehensive_report["recommendations"]:
            summary += f" {rec['recommendation']} (Priority: {rec['priority']})\n"

        return summary

    def run_complete_performance_analysis(self, data_sources):
        """Run complete performance analysis"""
        print(" RUNNING COMPLETE PERFORMANCE ANALYSIS")
        print("=" * 70)

        # Load data sources
        transactions_df = pd.read_csv(data_sources["transactions"])
        customers_df = pd.read_csv(data_sources["customers"])

        # Calculate business performance metrics
        self.calculate_business_performance_metrics(transactions_df, customers_df)

        # Calculate ROI metrics (using sample data)
        baseline_revenue = 8000000  # $8M baseline
        current_revenue = transactions_df["total_amount"].sum()
        self.calculate_roi_and_impact_metrics(baseline_revenue, current_revenue)

        # Generate dashboard
        dashboard_path = self.generate_performance_dashboard()

        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report()

        print(f"\n PERFORMANCE ANALYSIS COMPLETED!")
        print(
            f"    Overall Score: {comprehensive_report['executive_summary']['overall_performance_score']:.1f}/100"
        )
        print(
            f"    Key Achievements: {len(comprehensive_report['executive_summary']['key_achievements'])}"
        )
        print(f"    Recommendations: {len(comprehensive_report['recommendations'])}")

        return comprehensive_report


def main():
    """Demo of performance metrics generator"""
    print(" PERFORMANCE METRICS GENERATOR DEMO")
    print("=" * 50)

    # Initialize generator
    metrics_generator = PerformanceMetricsGenerator()

    # Run complete analysis
    data_sources = {
        "transactions": "data/transactions_real.csv",
        "customers": "data/customers_real.csv",
    }

    results = metrics_generator.run_complete_performance_analysis(data_sources)

    print(f"\n ANALYSIS RESULTS:")
    print(
        f"   Overall Performance: {results['executive_summary']['overall_performance_score']:.1f}/100"
    )
    print(
        f"   Models Evaluated: {results['executive_summary']['total_models_evaluated']}"
    )
    print(
        f"   Key Achievements: {len(results['executive_summary']['key_achievements'])}"
    )

    print(f"\n Performance Metrics Generator Demo Completed!")


if __name__ == "__main__":
    main()
