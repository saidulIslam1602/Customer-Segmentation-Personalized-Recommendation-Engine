"""
Customer Segmentation using RFM Analysis and Advanced Clustering
Implements state-of-the-art segmentation techniques for Coop Norge business case
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import os
import json
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class CustomerSegmentation:
    """
    Enhanced Customer Segmentation for Retail Business Intelligence

    Advanced Features:
    - RFM Analysis with CLV Integration
    - Behavioral Clustering (Purchase Patterns)
    - Customer Lifetime Value (CLV) Prediction
    - Dynamic Re-segmentation Triggers
    - Segment-Specific Retention Strategies
    - Anomaly Detection for VIP Customers
    - Predictive Segment Migration
    - Advanced Feature Engineering
    """

    def __init__(self, transaction_data_path, customer_data_path):
        """Initialize with enhanced segmentation capabilities"""
        self.transactions = pd.read_csv(transaction_data_path)
        self.customers = pd.read_csv(customer_data_path)

        # Core segmentation data
        self.rfm_data = None
        self.segments = None
        self.model = None

        # Enhanced features
        self.clv_data = None
        self.behavioral_features = None
        self.clv_model = None
        self.segment_profiles = {}
        self.retention_strategies = {}

        # Anomaly detection
        self.anomaly_detector = None
        self.vip_customers = None

        # Performance tracking
        self.segmentation_history = []
        self.segment_stability = {}

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

    def prepare_data(self):
        """Prepare and clean data for segmentation"""
        print(" Preparing data for segmentation...")

        # Convert date columns
        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"]
        )

        # Filter out any negative amounts or quantities
        self.transactions = self.transactions[
            (self.transactions["total_amount"] > 0)
            & (self.transactions["quantity"] > 0)
        ]

        # Calculate reference date (most recent transaction + 1 day)
        self.reference_date = self.transactions["transaction_date"].max() + timedelta(
            days=1
        )

        print(
            f" Data prepared: {len(self.transactions)} transactions, {len(self.customers)} customers"
        )

    def calculate_rfm(self):
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        print(" Calculating RFM metrics...")

        # Calculate RFM metrics
        rfm = (
            self.transactions.groupby("customer_id")
            .agg(
                {
                    "transaction_date": [
                        ("recency", lambda x: (self.reference_date - x.max()).days),
                        ("frequency", "count"),
                    ],
                    "total_amount": [("monetary", "sum"), ("avg_order_value", "mean")],
                }
            )
            .round(2)
        )

        # Flatten column names
        rfm.columns = ["recency", "frequency", "monetary", "avg_order_value"]
        rfm = rfm.reset_index()

        # Add customer demographics
        rfm = rfm.merge(
            self.customers[["customer_id", "loyalty_tier", "age", "household_size"]],
            on="customer_id",
            how="left",
        )

        # Calculate additional metrics
        rfm["days_since_first_purchase"] = (
            self.transactions.groupby("customer_id")["transaction_date"]
            .agg(lambda x: (self.reference_date - x.min()).days)
            .values
        )

        rfm["purchase_frequency_per_month"] = rfm["frequency"] / (
            rfm["days_since_first_purchase"] / 30
        )

        # Proper CLV calculation with forward-looking components
        # CLV = (Average Order Value × Purchase Frequency × Gross Margin × Customer Lifespan) - Acquisition Cost

        # Calculate average time between purchases (in days)
        avg_days_between_purchases = rfm["days_since_first_purchase"] / (
            rfm["frequency"] + 1
        )

        # Estimate annual purchase frequency
        annual_frequency = 365 / avg_days_between_purchases.replace(
            [np.inf, 0], 365
        )  # Handle edge cases

        # Calculate average order value
        rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
        rfm["avg_order_value"] = rfm["avg_order_value"].fillna(
            rfm["monetary"]
        )  # For single purchases

        # Estimate customer lifespan (simplified model based on recency and frequency)
        # More frequent, recent customers likely to have longer lifespan
        recency_factor = 1 / (1 + rfm["recency"] / 365)  # Decay function
        frequency_factor = np.log1p(rfm["frequency"]) / np.log1p(rfm["frequency"].max())
        estimated_lifespan_years = (
            2 * recency_factor * frequency_factor + 0.5
        )  # 0.5 to 2.5 years

        # Gross margin assumption (typical retail 20-30%)
        gross_margin = 0.25

        # Acquisition cost assumption (typical retail 50-100 NOK)
        acquisition_cost = 75

        # Calculate forward-looking CLV
        rfm["customer_lifetime_value"] = (
            rfm["avg_order_value"]
            * annual_frequency
            * gross_margin
            * estimated_lifespan_years
        ) - acquisition_cost

        # Ensure CLV is positive (minimum 0)
        rfm["customer_lifetime_value"] = np.maximum(rfm["customer_lifetime_value"], 0)

        # Handle any missing values
        rfm = rfm.fillna(0)

        self.rfm_data = rfm
        print(f" RFM metrics calculated for {len(rfm)} customers")

        return rfm

    def create_rfm_scores(self):
        """Create RFM scores using quantile-based scoring"""
        print(" Creating RFM scores...")

        if self.rfm_data is None:
            self.calculate_rfm()

        # Create RFM scores (1-5 scale) with robust binning
        # Note: For Recency, lower values are better (more recent)
        try:
            self.rfm_data["recency_score"] = pd.qcut(
                self.rfm_data["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop"
            )
        except ValueError:
            # Fallback for edge cases where qcut fails
            self.rfm_data["recency_score"] = pd.cut(
                self.rfm_data["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop"
            )

        try:
            self.rfm_data["frequency_score"] = pd.qcut(
                self.rfm_data["frequency"].rank(method="first"),
                5,
                labels=[1, 2, 3, 4, 5],
                duplicates="drop",
            )
        except ValueError:
            # Fallback for edge cases where qcut fails
            self.rfm_data["frequency_score"] = pd.cut(
                self.rfm_data["frequency"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            )

        try:
            self.rfm_data["monetary_score"] = pd.qcut(
                self.rfm_data["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            )
        except ValueError:
            # Fallback for edge cases where qcut fails
            self.rfm_data["monetary_score"] = pd.cut(
                self.rfm_data["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            )

        # Convert to numeric
        self.rfm_data["recency_score"] = self.rfm_data["recency_score"].astype(int)
        self.rfm_data["frequency_score"] = self.rfm_data["frequency_score"].astype(int)
        self.rfm_data["monetary_score"] = self.rfm_data["monetary_score"].astype(int)

        # Create combined RFM score
        self.rfm_data["rfm_score"] = (
            self.rfm_data["recency_score"].astype(str)
            + self.rfm_data["frequency_score"].astype(str)
            + self.rfm_data["monetary_score"].astype(str)
        )

        # Create segment based on RFM score
        self.rfm_data["rfm_segment"] = self.rfm_data["rfm_score"].apply(
            self._categorize_rfm
        )

        print(" RFM scores created")
        return self.rfm_data

    def _categorize_rfm(self, rfm_score):
        """Categorize customers based on RFM score"""
        if rfm_score in ["555", "554", "544", "545", "454", "455", "445"]:
            return "Champions"
        elif rfm_score in ["543", "444", "435", "355", "354", "345", "344", "335"]:
            return "Loyal Customers"
        elif rfm_score in [
            "553",
            "551",
            "552",
            "541",
            "542",
            "533",
            "532",
            "531",
            "452",
            "451",
        ]:
            return "Potential Loyalists"
        elif rfm_score in ["512", "511", "422", "421", "412", "411", "311"]:
            return "New Customers"
        elif rfm_score in ["155", "154", "144", "214", "215", "115", "114"]:
            return "At Risk"
        elif rfm_score in ["155", "254", "144", "214", "215", "115", "114"]:
            return "Cannot Lose Them"
        elif rfm_score in ["331", "321", "231", "241", "251"]:
            return "Hibernating"
        else:
            return "Others"

    def advanced_clustering(self, n_clusters=None):
        """Perform advanced clustering using multiple algorithms"""
        print("🤖 Performing advanced clustering...")

        if self.rfm_data is None:
            self.create_rfm_scores()

        # Prepare features for clustering
        features = [
            "recency",
            "frequency",
            "monetary",
            "avg_order_value",
            "purchase_frequency_per_month",
        ]
        X = self.rfm_data[features].copy()

        # Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_data["cluster"] = kmeans.fit_predict(X_scaled)

        # Apply Gaussian Mixture Model for comparison
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        self.rfm_data["cluster_gmm"] = gmm.fit_predict(X_scaled)

        # Evaluate clustering quality
        silhouette_kmeans = silhouette_score(X_scaled, self.rfm_data["cluster"])
        silhouette_gmm = silhouette_score(X_scaled, self.rfm_data["cluster_gmm"])

        # Choose best clustering method
        if silhouette_kmeans >= silhouette_gmm:
            self.rfm_data["final_cluster"] = self.rfm_data["cluster"]
            self.model = kmeans
            print(f" K-Means selected (Silhouette Score: {silhouette_kmeans:.3f})")
        else:
            self.rfm_data["final_cluster"] = self.rfm_data["cluster_gmm"]
            self.model = gmm
            print(f" GMM selected (Silhouette Score: {silhouette_gmm:.3f})")

        # Create business-friendly segment names
        self._create_segment_names()

        return self.rfm_data

    def _find_optimal_clusters(self, X_scaled, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(X_scaled) // 2))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Find optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f" Optimal number of clusters: {optimal_k}")
        return optimal_k

    def _create_segment_names(self):
        """Create business-friendly names for clusters based on characteristics"""
        cluster_summary = (
            self.rfm_data.groupby("final_cluster")
            .agg(
                {
                    "recency": "mean",
                    "frequency": "mean",
                    "monetary": "mean",
                    "customer_lifetime_value": "mean",
                }
            )
            .round(2)
        )

        # Name segments based on characteristics
        segment_names = {}
        for cluster in cluster_summary.index:
            row = cluster_summary.loc[cluster]

            if row["monetary"] > cluster_summary["monetary"].quantile(0.8):
                if row["frequency"] > cluster_summary["frequency"].quantile(0.7):
                    segment_names[cluster] = "VIP Champions"
                else:
                    segment_names[cluster] = "Big Spenders"
            elif row["frequency"] > cluster_summary["frequency"].quantile(0.8):
                segment_names[cluster] = "Frequent Buyers"
            elif row["recency"] < cluster_summary["recency"].quantile(0.3):
                segment_names[cluster] = "Recent Customers"
            elif row["recency"] > cluster_summary["recency"].quantile(0.7):
                segment_names[cluster] = "At Risk"
            else:
                segment_names[cluster] = f"Regular Customers"

        self.rfm_data["segment_name"] = self.rfm_data["final_cluster"].map(
            segment_names
        )

    def generate_insights(self):
        """Generate actionable business insights from segmentation"""
        print(" Generating business insights...")

        if self.rfm_data is None:
            self.advanced_clustering()

        insights = {}

        # Overall statistics
        insights["total_customers"] = len(self.rfm_data)
        insights["total_revenue"] = self.rfm_data["monetary"].sum()
        insights["avg_clv"] = self.rfm_data["customer_lifetime_value"].mean()

        # Segment analysis
        segment_analysis = (
            self.rfm_data.groupby("segment_name")
            .agg(
                {
                    "customer_id": "count",
                    "monetary": ["sum", "mean"],
                    "frequency": "mean",
                    "recency": "mean",
                    "customer_lifetime_value": "mean",
                }
            )
            .round(2)
        )

        segment_analysis.columns = [
            "customers",
            "total_revenue",
            "avg_revenue",
            "avg_frequency",
            "avg_recency",
            "avg_clv",
        ]
        segment_analysis["revenue_percentage"] = (
            segment_analysis["total_revenue"]
            / segment_analysis["total_revenue"].sum()
            * 100
        ).round(1)
        segment_analysis["customer_percentage"] = (
            segment_analysis["customers"] / segment_analysis["customers"].sum() * 100
        ).round(1)

        insights["segment_analysis"] = segment_analysis

        # Top segments by revenue
        insights["top_revenue_segments"] = segment_analysis.nlargest(
            3, "total_revenue"
        ).index.tolist()

        # At-risk customers
        at_risk = self.rfm_data[
            self.rfm_data["recency"] > 90
        ]  # Haven't purchased in 90+ days
        insights["at_risk_customers"] = len(at_risk)
        insights["at_risk_revenue"] = at_risk["monetary"].sum()

        # Loyalty tier analysis
        loyalty_analysis = (
            self.rfm_data.groupby("loyalty_tier")
            .agg({"monetary": ["mean", "sum"], "frequency": "mean"})
            .round(2)
        )

        insights["loyalty_analysis"] = loyalty_analysis

        self.insights = insights
        return insights

    def plot_segmentation_analysis(self, save_path=None):
        """Create comprehensive segmentation visualizations"""
        if self.rfm_data is None:
            self.advanced_clustering()

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Coop Norge - Customer Segmentation Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. RFM Distribution
        axes[0, 0].hist(self.rfm_data["recency"], bins=30, alpha=0.7, color="skyblue")
        axes[0, 0].set_title("Recency Distribution")
        axes[0, 0].set_xlabel("Days Since Last Purchase")

        axes[0, 1].hist(
            self.rfm_data["frequency"], bins=30, alpha=0.7, color="lightgreen"
        )
        axes[0, 1].set_title("Frequency Distribution")
        axes[0, 1].set_xlabel("Number of Purchases")

        axes[0, 2].hist(self.rfm_data["monetary"], bins=30, alpha=0.7, color="salmon")
        axes[0, 2].set_title("Monetary Distribution")
        axes[0, 2].set_xlabel("Total Spent (NOK)")

        # 2. Segment Analysis
        segment_counts = self.rfm_data["segment_name"].value_counts()
        axes[1, 0].pie(
            segment_counts.values, labels=segment_counts.index, autopct="%1.1f%%"
        )
        axes[1, 0].set_title("Customer Segments Distribution")

        # 3. Segment Revenue
        segment_revenue = (
            self.rfm_data.groupby("segment_name")["monetary"]
            .sum()
            .sort_values(ascending=True)
        )
        axes[1, 1].barh(segment_revenue.index, segment_revenue.values)
        axes[1, 1].set_title("Revenue by Segment (NOK)")
        axes[1, 1].set_xlabel("Total Revenue")

        # 4. RFM Scatter
        scatter = axes[1, 2].scatter(
            self.rfm_data["frequency"],
            self.rfm_data["monetary"],
            c=self.rfm_data["final_cluster"],
            cmap="viridis",
            alpha=0.6,
        )
        axes[1, 2].set_title("Frequency vs Monetary by Cluster")
        axes[1, 2].set_xlabel("Frequency")
        axes[1, 2].set_ylabel("Monetary (NOK)")
        plt.colorbar(scatter, ax=axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f" Analysis plots saved to {save_path}")

        plt.show()

    def get_marketing_recommendations(self):
        """Generate specific marketing recommendations for each segment"""
        if not hasattr(self, "insights"):
            self.generate_insights()

        recommendations = {}

        for segment in self.rfm_data["segment_name"].unique():
            segment_data = self.rfm_data[self.rfm_data["segment_name"] == segment]
            avg_recency = segment_data["recency"].mean()
            avg_frequency = segment_data["frequency"].mean()
            avg_monetary = segment_data["monetary"].mean()

            if "VIP" in segment or "Champions" in segment:
                recommendations[segment] = {
                    "strategy": "Retain and Reward",
                    "tactics": [
                        "Exclusive member events and early access",
                        "Premium customer service",
                        "Personalized high-value product recommendations",
                        "Loyalty program upgrades",
                    ],
                    "channel": "Personal touch + digital",
                    "frequency": "Monthly",
                }
            elif "At Risk" in segment:
                recommendations[segment] = {
                    "strategy": "Win-back Campaign",
                    "tactics": [
                        "Discount coupons for next purchase",
                        "Email series highlighting new products",
                        "Survey to understand issues",
                        "Limited-time offers",
                    ],
                    "channel": "Email + SMS",
                    "frequency": "Bi-weekly",
                }
            elif "Recent" in segment or "New" in segment:
                recommendations[segment] = {
                    "strategy": "Onboarding and Education",
                    "tactics": [
                        "Welcome series explaining Coop benefits",
                        "Tutorial on loyalty program",
                        "Cross-selling complementary products",
                        "First purchase incentives",
                    ],
                    "channel": "Mobile app + email",
                    "frequency": "Weekly for first month",
                }
            else:
                recommendations[segment] = {
                    "strategy": "Engagement and Growth",
                    "tactics": [
                        "Product recommendations based on purchase history",
                        "Seasonal campaign participation",
                        "Category-specific promotions",
                        "Loyalty point boosters",
                    ],
                    "channel": "App notifications + email",
                    "frequency": "Bi-weekly",
                }

        return recommendations

    def export_segments(self, filename="customer_segments.csv"):
        """Export segmentation results for business use"""
        if self.rfm_data is None:
            self.advanced_clustering()

        # Dynamically select available columns
        required_cols = ["customer_id", "recency", "frequency", "monetary"]
        optional_cols = [
            "segment_name",
            "final_cluster",
            "customer_lifetime_value",
            "rfm_score",
            "loyalty_tier",
            "age",
            "household_size",
        ]

        export_cols = required_cols.copy()
        for col in optional_cols:
            if col in self.rfm_data.columns:
                export_cols.append(col)

        export_data = self.rfm_data[export_cols].copy()

        export_data.to_csv(filename, index=False)
        print(f" Segmentation results exported to {filename}")
        print(f"   Columns exported: {', '.join(export_cols)}")
        return export_data

    def calculate_customer_lifetime_value(self):
        """Calculate Customer Lifetime Value (CLV) using advanced modeling"""
        print(" Calculating Customer Lifetime Value (CLV)...")

        if self.rfm_data is None:
            self.calculate_rfm()

        # Enhanced CLV features
        clv_features = self.rfm_data.copy()

        # Historical value metrics
        clv_features["historical_clv"] = (
            clv_features["monetary"] * clv_features["frequency"]
        )

        # Predictive features for CLV modeling
        clv_features["avg_days_between_purchases"] = (
            clv_features["recency"] / clv_features["frequency"]
        )
        clv_features["purchase_acceleration"] = np.where(
            clv_features["avg_days_between_purchases"] > 0,
            1 / clv_features["avg_days_between_purchases"],
            0,
        )

        # Seasonal purchasing patterns
        seasonal_data = self.transactions.copy()
        seasonal_data["month"] = pd.to_datetime(
            seasonal_data["transaction_date"]
        ).dt.month
        seasonal_data["quarter"] = pd.to_datetime(
            seasonal_data["transaction_date"]
        ).dt.quarter

        seasonal_patterns = (
            seasonal_data.groupby(["customer_id", "quarter"])
            .agg({"total_amount": "sum", "transaction_date": "count"})
            .reset_index()
        )

        seasonal_variance = (
            seasonal_patterns.groupby("customer_id")
            .agg({"total_amount": "std", "transaction_date": "std"})
            .fillna(0)
        )

        clv_features = clv_features.merge(
            seasonal_variance.rename(
                columns={
                    "total_amount": "seasonal_spend_variance",
                    "transaction_date": "seasonal_frequency_variance",
                }
            ),
            left_on="customer_id",
            right_index=True,
            how="left",
        ).fillna(0)

        # Customer tenure and lifecycle stage
        first_purchase = self.transactions.groupby("customer_id")[
            "transaction_date"
        ].min()
        last_purchase = self.transactions.groupby("customer_id")[
            "transaction_date"
        ].max()

        clv_features["customer_tenure_days"] = (
            pd.to_datetime("today") - pd.to_datetime(first_purchase)
        ).dt.days

        clv_features["days_since_last_purchase"] = (
            pd.to_datetime("today") - pd.to_datetime(last_purchase)
        ).dt.days

        # Lifecycle stage classification
        clv_features["lifecycle_stage"] = pd.cut(
            clv_features["customer_tenure_days"],
            bins=[0, 30, 90, 365, 730, np.inf],
            labels=["New", "Developing", "Established", "Mature", "Veteran"],
        )

        # Train CLV prediction model
        self._train_clv_model(clv_features)

        self.clv_data = clv_features

        print(f" CLV calculated for {len(clv_features)} customers")
        print(f"   Average CLV: ${clv_features['historical_clv'].mean():.2f}")
        print(
            f"   CLV Range: ${clv_features['historical_clv'].min():.2f} - ${clv_features['historical_clv'].max():.2f}"
        )

        return clv_features

    def _train_clv_model(self, clv_features):
        """Train predictive CLV model"""
        print("🤖 Training CLV prediction model...")

        # Prepare features for modeling
        feature_columns = [
            "recency",
            "frequency",
            "monetary",
            "avg_order_value",
            "purchase_frequency_per_month",
            "avg_days_between_purchases",
            "purchase_acceleration",
            "seasonal_spend_variance",
            "seasonal_frequency_variance",
            "customer_tenure_days",
            "days_since_last_purchase",
        ]

        # Encode categorical features
        le = LabelEncoder()
        clv_features["lifecycle_stage_encoded"] = le.fit_transform(
            clv_features["lifecycle_stage"].astype(str)
        )
        feature_columns.append("lifecycle_stage_encoded")

        # Prepare training data
        X = clv_features[feature_columns].fillna(0)
        y = clv_features["historical_clv"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest model
        self.clv_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.clv_model.fit(X_train, y_train)

        # Evaluate model
        train_score = self.clv_model.score(X_train, y_train)
        test_score = self.clv_model.score(X_test, y_test)

        # Predict future CLV
        clv_features["predicted_clv"] = self.clv_model.predict(X)

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": self.clv_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(f"    CLV model trained - R²: {test_score:.4f}")
        print(f"    Top CLV predictors:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")

        # Save model
        joblib.dump(self.clv_model, "models/clv_prediction_model.joblib")

        return self.clv_model

    def create_behavioral_features(self):
        """Create advanced behavioral features for clustering"""
        print("Creating behavioral features...")

        # Purchase behavior patterns
        behavioral_data = (
            self.transactions.groupby("customer_id")
            .agg(
                {
                    "total_amount": ["sum", "mean", "std", "min", "max"],
                    "quantity": ["sum", "mean", "std"],
                    "transaction_date": ["count", "min", "max"],
                    "product_id": "nunique",
                    "store_id": "nunique",
                    "channel": lambda x: (
                        x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
                    ),
                    "payment_method": lambda x: (
                        x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
                    ),
                }
            )
            .round(2)
        )

        # Flatten column names
        behavioral_data.columns = [
            "_".join(col).strip() for col in behavioral_data.columns
        ]
        behavioral_data = behavioral_data.reset_index()

        # Time-based patterns
        transactions_with_time = self.transactions.copy()
        transactions_with_time["transaction_date"] = pd.to_datetime(
            transactions_with_time["transaction_date"]
        )
        transactions_with_time["hour"] = transactions_with_time[
            "transaction_date"
        ].dt.hour
        transactions_with_time["day_of_week"] = transactions_with_time[
            "transaction_date"
        ].dt.dayofweek
        transactions_with_time["month"] = transactions_with_time[
            "transaction_date"
        ].dt.month

        # Preferred shopping times
        time_patterns = (
            transactions_with_time.groupby("customer_id")
            .agg(
                {
                    "hour": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12,
                    "day_of_week": lambda x: (
                        x.mode().iloc[0] if len(x.mode()) > 0 else 1
                    ),
                    "month": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 6,
                }
            )
            .reset_index()
        )

        time_patterns.columns = [
            "customer_id",
            "preferred_hour",
            "preferred_day",
            "preferred_month",
        ]

        # Shopping consistency metrics
        consistency_metrics = (
            transactions_with_time.groupby("customer_id")
            .agg(
                {
                    "hour": "std",
                    "day_of_week": "std",
                    "total_amount": lambda x: (
                        x.std() / x.mean() if x.mean() > 0 else 0
                    ),  # Coefficient of variation
                }
            )
            .fillna(0)
            .reset_index()
        )

        consistency_metrics.columns = [
            "customer_id",
            "hour_consistency",
            "day_consistency",
            "amount_consistency",
        ]

        # Purchase intervals
        purchase_intervals = (
            transactions_with_time.groupby("customer_id")["transaction_date"]
            .apply(lambda x: x.sort_values().diff().dt.days.mean())
            .fillna(0)
            .reset_index()
        )
        purchase_intervals.columns = ["customer_id", "avg_purchase_interval"]

        # Combine all behavioral features
        behavioral_features = behavioral_data.merge(
            time_patterns, on="customer_id", how="left"
        )
        behavioral_features = behavioral_features.merge(
            consistency_metrics, on="customer_id", how="left"
        )
        behavioral_features = behavioral_features.merge(
            purchase_intervals, on="customer_id", how="left"
        )

        # Product diversity metrics
        product_diversity = (
            self.transactions.groupby("customer_id")["product_id"]
            .apply(
                lambda x: (
                    len(set(x)) / len(x) if len(x) > 0 else 0
                )  # Unique products ratio
            )
            .reset_index()
        )
        product_diversity.columns = ["customer_id", "product_diversity_ratio"]

        behavioral_features = behavioral_features.merge(
            product_diversity, on="customer_id", how="left"
        )

        # Fill missing values
        behavioral_features = behavioral_features.fillna(0)

        self.behavioral_features = behavioral_features

        print(f" Behavioral features created for {len(behavioral_features)} customers")
        print(f"   Features: {behavioral_features.shape[1] - 1}")  # Exclude customer_id

        return behavioral_features

    def detect_vip_customers(self):
        """Detect VIP customers using anomaly detection"""
        print(" Detecting VIP customers...")

        if self.clv_data is None:
            self.calculate_customer_lifetime_value()

        # Features for VIP detection
        vip_features = self.clv_data[
            [
                "monetary",
                "frequency",
                "historical_clv",
                "predicted_clv",
                "avg_order_value",
                "customer_tenure_days",
            ]
        ].fillna(0)

        # Scale features
        scaler = StandardScaler()
        vip_features_scaled = scaler.fit_transform(vip_features)

        # Use Isolation Forest for anomaly detection (VIP = positive outliers)
        self.anomaly_detector = IsolationForest(
            contamination=0.05, random_state=42  # Expect 5% VIPs
        )

        anomaly_scores = self.anomaly_detector.fit_predict(vip_features_scaled)
        anomaly_scores_continuous = self.anomaly_detector.score_samples(
            vip_features_scaled
        )

        # VIP customers are those with high positive anomaly scores
        self.clv_data["vip_score"] = anomaly_scores_continuous
        self.clv_data["is_vip"] = (anomaly_scores == -1) & (
            self.clv_data["monetary"] > self.clv_data["monetary"].quantile(0.8)
        )

        vip_customers = self.clv_data[self.clv_data["is_vip"]].copy()
        self.vip_customers = vip_customers

        print(f" VIP detection completed")
        print(f"    VIP customers identified: {len(vip_customers)}")
        print(f"    Average VIP CLV: ${vip_customers['historical_clv'].mean():.2f}")
        print(
            f"    VIP contribution: {(vip_customers['historical_clv'].sum() / self.clv_data['historical_clv'].sum() * 100):.1f}% of total value"
        )

        return vip_customers

    def create_segment_profiles(self):
        """Create detailed profiles for each segment"""
        print(" Creating segment profiles...")

        if self.rfm_data is None or "segment_name" not in self.rfm_data.columns:
            print("  Segmentation not completed. Run advanced_clustering() first.")
            return None

        # Combine all data for profiling
        profile_data = self.rfm_data.copy()

        if self.clv_data is not None:
            profile_data = profile_data.merge(
                self.clv_data[
                    [
                        "customer_id",
                        "historical_clv",
                        "predicted_clv",
                        "lifecycle_stage",
                        "is_vip",
                    ]
                ],
                on="customer_id",
                how="left",
            )

        if self.behavioral_features is not None:
            profile_data = profile_data.merge(
                self.behavioral_features, on="customer_id", how="left"
            )

        # Create profiles by segment
        segments = profile_data["segment_name"].unique()

        for segment in segments:
            segment_data = profile_data[profile_data["segment_name"] == segment]

            profile = {
                "segment_name": segment,
                "size": len(segment_data),
                "percentage": len(segment_data) / len(profile_data) * 100,
                # RFM characteristics
                "avg_recency": segment_data["recency"].mean(),
                "avg_frequency": segment_data["frequency"].mean(),
                "avg_monetary": segment_data["monetary"].mean(),
                "avg_order_value": segment_data["avg_order_value"].mean(),
                # CLV characteristics
                "avg_historical_clv": (
                    segment_data["historical_clv"].mean()
                    if "historical_clv" in segment_data.columns
                    else 0
                ),
                "avg_predicted_clv": (
                    segment_data["predicted_clv"].mean()
                    if "predicted_clv" in segment_data.columns
                    else 0
                ),
                "vip_percentage": (
                    segment_data["is_vip"].mean() * 100
                    if "is_vip" in segment_data.columns
                    else 0
                ),
                # Behavioral characteristics
                "product_diversity": (
                    segment_data["product_diversity_ratio"].mean()
                    if "product_diversity_ratio" in segment_data.columns
                    else 0
                ),
                "purchase_consistency": (
                    segment_data["amount_consistency"].mean()
                    if "amount_consistency" in segment_data.columns
                    else 0
                ),
                # Business value
                "total_value": segment_data["monetary"].sum(),
                "value_contribution": segment_data["monetary"].sum()
                / profile_data["monetary"].sum()
                * 100,
            }

            self.segment_profiles[segment] = profile

        print(f" Segment profiles created for {len(segments)} segments")

        # Print summary
        print(f"\n SEGMENT PROFILE SUMMARY:")
        for segment, profile in self.segment_profiles.items():
            print(f"   {segment}:")
            print(f"     Size: {profile['size']:,} ({profile['percentage']:.1f}%)")
            print(f"     Avg CLV: ${profile['avg_historical_clv']:.2f}")
            print(f"     Value Contribution: {profile['value_contribution']:.1f}%")
            print(f"     VIP %: {profile['vip_percentage']:.1f}%")

        return self.segment_profiles

    def generate_retention_strategies(self):
        """Generate segment-specific retention strategies"""
        print(" Generating retention strategies...")

        if not self.segment_profiles:
            self.create_segment_profiles()

        for segment, profile in self.segment_profiles.items():
            strategy = {
                "segment": segment,
                "priority": self._calculate_segment_priority(profile),
                "recommended_actions": [],
                "budget_allocation": 0,
                "expected_roi": 0,
            }

            # Strategy based on segment characteristics
            if profile["avg_recency"] > 90:  # Inactive customers
                strategy["recommended_actions"].extend(
                    [
                        "Win-back email campaign",
                        "Special discount offers (15-20%)",
                        "Personalized product recommendations",
                        "Survey to understand absence reasons",
                    ]
                )
                strategy["budget_allocation"] = profile["size"] * 5  # $5 per customer

            elif profile["avg_frequency"] < 2:  # Low frequency
                strategy["recommended_actions"].extend(
                    [
                        "Loyalty program enrollment",
                        "Cross-selling campaigns",
                        "Educational content about products",
                        "Reminder notifications",
                    ]
                )
                strategy["budget_allocation"] = profile["size"] * 3  # $3 per customer

            elif profile["vip_percentage"] > 10:  # High VIP concentration
                strategy["recommended_actions"].extend(
                    [
                        "VIP exclusive events",
                        "Premium customer service",
                        "Early access to new products",
                        "Personalized shopping experiences",
                    ]
                )
                strategy["budget_allocation"] = profile["size"] * 15  # $15 per customer

            else:  # Regular customers
                strategy["recommended_actions"].extend(
                    [
                        "Regular newsletter with offers",
                        "Seasonal promotions",
                        "Product bundling suggestions",
                        "Referral incentives",
                    ]
                )
                strategy["budget_allocation"] = profile["size"] * 2  # $2 per customer

            # Calculate expected ROI based on segment value
            if strategy["budget_allocation"] > 0:
                expected_retention_rate = 0.15  # 15% improvement
                expected_revenue_increase = (
                    profile["total_value"] * expected_retention_rate
                )
                strategy["expected_roi"] = (
                    expected_revenue_increase / strategy["budget_allocation"] - 1
                ) * 100

            self.retention_strategies[segment] = strategy

        print(
            f" Retention strategies generated for {len(self.retention_strategies)} segments"
        )

        # Print strategy summary
        total_budget = sum(
            s["budget_allocation"] for s in self.retention_strategies.values()
        )
        weighted_roi = (
            sum(
                s["expected_roi"] * s["budget_allocation"]
                for s in self.retention_strategies.values()
            )
            / total_budget
            if total_budget > 0
            else 0
        )

        print(f"\n RETENTION STRATEGY SUMMARY:")
        print(f"   Total Budget: ${total_budget:,.2f}")
        print(f"   Expected Weighted ROI: {weighted_roi:.1f}%")

        for segment, strategy in self.retention_strategies.items():
            print(f"\n   {segment} ({strategy['priority']} Priority):")
            print(f"     Budget: ${strategy['budget_allocation']:,.2f}")
            print(f"     Expected ROI: {strategy['expected_roi']:.1f}%")
            print(f"     Actions: {len(strategy['recommended_actions'])} strategies")

        return self.retention_strategies

    def _calculate_segment_priority(self, profile):
        """Calculate priority level for segment"""
        # Priority based on value contribution and VIP percentage
        value_score = profile["value_contribution"] / 25  # Normalize to 0-4 scale
        vip_score = profile["vip_percentage"] / 25  # Normalize to 0-4 scale

        total_score = value_score + vip_score

        if total_score >= 3:
            return "High"
        elif total_score >= 1.5:
            return "Medium"
        else:
            return "Low"

    def run_enhanced_segmentation(self):
        """Run complete enhanced segmentation analysis"""
        print(" RUNNING ENHANCED CUSTOMER SEGMENTATION")
        print("=" * 60)

        # Step 1: Basic segmentation
        self.prepare_data()
        self.calculate_rfm()
        segments = self.advanced_clustering()

        # Step 2: Enhanced features
        self.calculate_customer_lifetime_value()
        self.create_behavioral_features()

        # Step 3: VIP detection
        self.detect_vip_customers()

        # Step 4: Segment profiling
        self.create_segment_profiles()

        # Step 5: Retention strategies
        self.generate_retention_strategies()

        # Step 6: Save results
        self.save_enhanced_results()

        print(f"\n ENHANCED SEGMENTATION COMPLETED!")
        print(f"    Segments: {len(self.segment_profiles)}")
        print(
            f"    VIP Customers: {len(self.vip_customers) if self.vip_customers is not None else 0}"
        )
        print(f"    Retention Strategies: {len(self.retention_strategies)}")

        return {
            "segments": segments,
            "clv_data": self.clv_data,
            "vip_customers": self.vip_customers,
            "segment_profiles": self.segment_profiles,
            "retention_strategies": self.retention_strategies,
        }

    def save_enhanced_results(self):
        """Save all enhanced segmentation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save segmentation data
        if self.rfm_data is not None:
            self.rfm_data.to_csv(
                f"reports/enhanced_segments_{timestamp}.csv", index=False
            )

        # Save CLV data
        if self.clv_data is not None:
            self.clv_data.to_csv(f"reports/customer_clv_{timestamp}.csv", index=False)

        # Save VIP customers
        if self.vip_customers is not None:
            self.vip_customers.to_csv(
                f"reports/vip_customers_{timestamp}.csv", index=False
            )

        # Save segment profiles and strategies
        results_summary = {
            "timestamp": timestamp,
            "segment_profiles": self.segment_profiles,
            "retention_strategies": self.retention_strategies,
            "total_customers": len(self.rfm_data) if self.rfm_data is not None else 0,
            "total_vips": (
                len(self.vip_customers) if self.vip_customers is not None else 0
            ),
        }

        with open(f"reports/segmentation_summary_{timestamp}.json", "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        # Save models
        if self.clv_model is not None:
            joblib.dump(self.clv_model, f"models/clv_model_{timestamp}.joblib")

        if self.anomaly_detector is not None:
            joblib.dump(
                self.anomaly_detector, f"models/vip_detector_{timestamp}.joblib"
            )

        print(f" Enhanced segmentation results saved with timestamp: {timestamp}")

        return timestamp


if __name__ == "__main__":
    # Example usage
    print(" Coop Norge - Customer Segmentation Analysis")
    print("=" * 50)

    # Initialize segmentation
    segmentation = CustomerSegmentation(
        "../data/transactions_real.csv", "../data/customers_real.csv"
    )

    # Run complete analysis
    segmentation.prepare_data()
    segmentation.calculate_rfm()
    segmentation.create_rfm_scores()
    segmentation.advanced_clustering()

    # Generate insights
    insights = segmentation.generate_insights()
    print(f"\n KEY INSIGHTS:")
    print(f"• Total Customers: {insights['total_customers']:,}")
    print(f"• Total Revenue: {insights['total_revenue']:,.2f} NOK")
    print(f"• Average CLV: {insights['avg_clv']:,.2f} NOK")
    print(f"• At-Risk Customers: {insights['at_risk_customers']:,}")

    # Get marketing recommendations
    recommendations = segmentation.get_marketing_recommendations()
    print(f"\n MARKETING RECOMMENDATIONS GENERATED FOR {len(recommendations)} SEGMENTS")

    # Export results
    segmentation.export_segments("../results/customer_segments.csv")

    # Create visualizations
    segmentation.plot_segmentation_analysis("../results/segmentation_analysis.png")
