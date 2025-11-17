"""
Advanced Customer Churn Prediction and Retention Strategy Engine
Addresses critical business issue: Customer retention and churn prevention
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os
import json
import warnings

warnings.filterwarnings("ignore")


class ChurnPredictionEngine:
    """
    Advanced Churn Prediction System for Retail Business

    Business Problems Addressed:
    - Customer retention and churn prevention
    - Proactive intervention strategies
    - Customer lifetime value protection
    - Targeted retention campaigns
    """

    def __init__(self, transactions_path, customers_path):
        """Initialize with transaction and customer data"""
        self.transactions = pd.read_csv(transactions_path)
        self.customers = pd.read_csv(customers_path)
        self.churn_features = None
        self.churn_model = None
        self.scaler = StandardScaler()
        self.model_metadata = {}

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

    def prepare_churn_features(self, churn_threshold_days=90):
        """
        Create comprehensive features for churn prediction

        Args:
            churn_threshold_days: Days without purchase to consider churned
        """
        print(" Creating advanced churn prediction features...")

        # Convert dates
        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"]
        )
        self.customers["member_since"] = pd.to_datetime(self.customers["member_since"])

        # Calculate reference date
        reference_date = self.transactions["transaction_date"].max()

        # Customer transaction aggregations
        customer_stats = (
            self.transactions.groupby("customer_id")
            .agg(
                {
                    "transaction_date": [
                        ("last_purchase_date", "max"),
                        ("first_purchase_date", "min"),
                        ("total_transactions", "count"),
                        ("purchase_frequency", lambda x: len(x.unique())),
                    ],
                    "total_amount": [
                        ("total_spent", "sum"),
                        ("avg_transaction_value", "mean"),
                        ("std_transaction_value", "std"),
                        ("max_transaction_value", "max"),
                        ("min_transaction_value", "min"),
                    ],
                    "quantity": [
                        ("total_items_purchased", "sum"),
                        ("avg_items_per_transaction", "mean"),
                    ],
                    "store_id": [("unique_stores_visited", "nunique")],
                    "channel": [("unique_channels_used", "nunique")],
                    "discount_applied": [("discount_usage_rate", "mean")],
                    "discount_amount": [("total_discounts_received", "sum")],
                }
            )
            .round(2)
        )

        # Flatten column names
        customer_stats.columns = [
            col[1] if col[1] else col[0] for col in customer_stats.columns
        ]
        customer_stats = customer_stats.reset_index()

        # Calculate time-based features
        customer_stats["days_since_last_purchase"] = (
            reference_date - customer_stats["last_purchase_date"]
        ).dt.days
        customer_stats["customer_lifetime_days"] = (
            customer_stats["last_purchase_date"] - customer_stats["first_purchase_date"]
        ).dt.days
        customer_stats["purchase_frequency_per_month"] = customer_stats[
            "total_transactions"
        ] / (customer_stats["customer_lifetime_days"] / 30 + 1)

        # Behavioral features
        customer_stats["avg_days_between_purchases"] = customer_stats[
            "customer_lifetime_days"
        ] / (customer_stats["total_transactions"] + 1)
        customer_stats["spending_acceleration"] = customer_stats["total_spent"] / (
            customer_stats["customer_lifetime_days"] + 1
        )
        customer_stats["transaction_value_consistency"] = customer_stats[
            "std_transaction_value"
        ] / (customer_stats["avg_transaction_value"] + 1)

        # Recent behavior (last 30 days)
        recent_date = reference_date - timedelta(days=30)
        recent_transactions = self.transactions[
            self.transactions["transaction_date"] >= recent_date
        ]

        recent_stats = (
            recent_transactions.groupby("customer_id")
            .agg({"total_amount": ["sum", "count"], "quantity": "sum"})
            .round(2)
        )

        recent_stats.columns = [
            "recent_30d_spent",
            "recent_30d_transactions",
            "recent_30d_items",
        ]
        recent_stats = recent_stats.reset_index()

        # Merge with main features
        customer_stats = customer_stats.merge(
            recent_stats, on="customer_id", how="left"
        )
        customer_stats[
            ["recent_30d_spent", "recent_30d_transactions", "recent_30d_items"]
        ] = customer_stats[
            ["recent_30d_spent", "recent_30d_transactions", "recent_30d_items"]
        ].fillna(
            0
        )

        # Add customer demographics
        customer_stats = customer_stats.merge(
            self.customers, on="customer_id", how="left"
        )

        # Calculate customer tenure
        customer_stats["customer_tenure_days"] = (
            reference_date - customer_stats["member_since"]
        ).dt.days

        # Create churn label
        customer_stats["is_churned"] = (
            customer_stats["days_since_last_purchase"] > churn_threshold_days
        ).astype(int)

        # Advanced behavioral indicators
        customer_stats["declining_spend_trend"] = (
            customer_stats["recent_30d_spent"] < customer_stats["avg_transaction_value"]
        ).astype(int)
        customer_stats["low_engagement_score"] = (
            (customer_stats["recent_30d_transactions"] == 0)
            & (customer_stats["days_since_last_purchase"] > 30)
        ).astype(int)

        # Loyalty and engagement features
        le = LabelEncoder()
        customer_stats["loyalty_tier_encoded"] = le.fit_transform(
            customer_stats["loyalty_tier"]
        )
        customer_stats["gender_encoded"] = le.fit_transform(customer_stats["gender"])

        # Risk scoring features
        customer_stats["churn_risk_score"] = (
            (customer_stats["days_since_last_purchase"] / 90) * 0.3
            + (
                1
                - customer_stats["recent_30d_transactions"]
                / customer_stats["purchase_frequency_per_month"].clip(lower=1)
            )
            * 0.2
            + (customer_stats["declining_spend_trend"]) * 0.2
            + (customer_stats["low_engagement_score"]) * 0.3
        ).clip(0, 1)

        self.churn_features = customer_stats

        print(f" Churn features created for {len(customer_stats)} customers")
        print(f" Churn rate: {customer_stats['is_churned'].mean():.2%}")

        return customer_stats

    def train_churn_model(self, test_size=0.2, hyperparameter_tuning=True):
        """Train advanced churn prediction model with hyperparameter optimization"""
        print("🤖 Training enhanced churn prediction model...")

        if self.churn_features is None:
            self.prepare_churn_features()

        # Select features for modeling
        feature_columns = [
            "total_transactions",
            "total_spent",
            "avg_transaction_value",
            "std_transaction_value",
            "total_items_purchased",
            "avg_items_per_transaction",
            "unique_stores_visited",
            "unique_channels_used",
            "discount_usage_rate",
            "total_discounts_received",
            "days_since_last_purchase",
            "customer_lifetime_days",
            "purchase_frequency_per_month",
            "avg_days_between_purchases",
            "spending_acceleration",
            "transaction_value_consistency",
            "recent_30d_spent",
            "recent_30d_transactions",
            "recent_30d_items",
            "age",
            "household_size",
            "estimated_income",
            "customer_tenure_days",
            "loyalty_tier_encoded",
            "gender_encoded",
            "declining_spend_trend",
            "low_engagement_score",
            "churn_risk_score",
        ]

        # Prepare data
        X = self.churn_features[feature_columns].fillna(0)
        y = self.churn_features["is_churned"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Enhanced model configurations with hyperparameter tuning
        if hyperparameter_tuning:
            print(" Performing hyperparameter optimization...")
            models = self._get_tuned_models(X_train, y_train)
        else:
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    random_state=42, class_weight="balanced"
                ),
            }

        best_model = None
        best_score = 0
        model_performances = {}

        for name, model in models.items():
            try:
                if name == "Logistic Regression":
                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                    if y_pred_proba.shape[1] > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                    else:
                        y_pred_proba = y_pred_proba[:, 0]
                else:
                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba.shape[1] > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                    else:
                        y_pred_proba = y_pred_proba[:, 0]

                # Handle case where all samples are from one class
                if len(set(y_test)) > 1:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                else:
                    auc_score = 0.5  # Random performance for single class

                print(f"   {name} AUC: {auc_score:.4f}")

                if auc_score > best_score:
                    best_score = auc_score
                    best_model = model
                    self.best_model_name = name

            except Exception as e:
                print(f"   {name} failed: {str(e)}")
                continue

        self.churn_model = best_model
        self.feature_columns = feature_columns

        # Feature importance
        if hasattr(best_model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            print("\n Top 10 Churn Prediction Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

        print(f" Best model: {self.best_model_name} (AUC: {best_score:.4f})")

        # Save model and metadata
        self._save_model_artifacts(feature_columns, model_performances)

        return best_model, best_score

    def _get_tuned_models(self, X_train, y_train):
        """Get hyperparameter-tuned models"""
        tuned_models = {}

        # Random Forest hyperparameter tuning
        rf_params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            rf_params,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        tuned_models["Random Forest"] = rf_grid.best_estimator_

        # Gradient Boosting hyperparameter tuning
        gb_params = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        }

        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        gb_grid.fit(X_train, y_train)
        tuned_models["Gradient Boosting"] = gb_grid.best_estimator_

        # Logistic Regression hyperparameter tuning
        lr_params = {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }

        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
            lr_params,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        lr_grid.fit(self.scaler.fit_transform(X_train), y_train)
        tuned_models["Logistic Regression"] = lr_grid.best_estimator_

        return tuned_models

    def _save_model_artifacts(self, feature_columns, model_performances):
        """Save model, scaler, and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = f"models/churn_model_{timestamp}.joblib"
        joblib.dump(self.churn_model, model_path)

        # Save scaler
        scaler_path = f"models/churn_scaler_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        self.model_metadata = {
            "model_type": self.best_model_name,
            "training_date": timestamp,
            "feature_columns": feature_columns,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "performance_metrics": model_performances,
            "data_shape": self.churn_features.shape,
            "churn_rate": float(self.churn_features["is_churned"].mean()),
        }

        metadata_path = f"models/churn_metadata_{timestamp}.json"
        with open(metadata_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2)

        print(f" Model artifacts saved:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Metadata: {metadata_path}")

    def load_model(self, model_timestamp=None):
        """Load a previously trained model"""
        if model_timestamp is None:
            # Find the latest model
            model_files = [
                f for f in os.listdir("models") if f.startswith("churn_model_")
            ]
            if not model_files:
                raise FileNotFoundError("No trained churn models found")
            model_timestamp = max(model_files).split("_")[2].split(".")[0]

        # Load model artifacts
        model_path = f"models/churn_model_{model_timestamp}.joblib"
        scaler_path = f"models/churn_scaler_{model_timestamp}.joblib"
        metadata_path = f"models/churn_metadata_{model_timestamp}.json"

        self.churn_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(metadata_path, "r") as f:
            self.model_metadata = json.load(f)

        print(f" Loaded churn model from {self.model_metadata['training_date']}")
        print(f"   Model type: {self.model_metadata['model_type']}")
        print(f"   Training churn rate: {self.model_metadata['churn_rate']:.2%}")

        return self.churn_model

    def predict_churn_risk(self, customer_ids=None):
        """Predict churn risk for customers"""
        if self.churn_model is None:
            raise ValueError("Model not trained. Call train_churn_model() first.")

        if customer_ids is not None:
            data = self.churn_features[
                self.churn_features["customer_id"].isin(customer_ids)
            ]
        else:
            data = self.churn_features

        X = data[self.feature_columns].fillna(0)

        if self.best_model_name == "Logistic Regression":
            X = self.scaler.transform(X)

        churn_proba = self.churn_model.predict_proba(X)
        if churn_proba.shape[1] > 1:
            churn_probabilities = churn_proba[:, 1]
        else:
            churn_probabilities = churn_proba[:, 0]

        results = data[["customer_id", "is_churned"]].copy()
        results["churn_probability"] = churn_probabilities
        results["risk_category"] = pd.cut(
            churn_probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )

        return results

    def generate_retention_strategies(self):
        """Generate targeted retention strategies for different risk segments"""
        print(" Generating retention strategies...")

        churn_predictions = self.predict_churn_risk()

        strategies = {
            "High Risk": {
                "customers": len(
                    churn_predictions[churn_predictions["risk_category"] == "High Risk"]
                ),
                "strategies": [
                    "Immediate personal outreach with exclusive offers",
                    "Win-back campaign with 20-30% discount",
                    "Free premium service upgrade for 3 months",
                    "Personal shopping assistance",
                    "Loyalty points bonus (2x points for next 3 purchases)",
                ],
                "expected_retention_rate": 0.35,
                "campaign_cost_per_customer": 150,
            },
            "Medium Risk": {
                "customers": len(
                    churn_predictions[
                        churn_predictions["risk_category"] == "Medium Risk"
                    ]
                ),
                "strategies": [
                    "Targeted email campaign with personalized offers",
                    "10-15% discount on favorite product categories",
                    "Early access to sales and new products",
                    "Mobile app engagement campaign",
                    "Referral incentives",
                ],
                "expected_retention_rate": 0.60,
                "campaign_cost_per_customer": 75,
            },
            "Low Risk": {
                "customers": len(
                    churn_predictions[churn_predictions["risk_category"] == "Low Risk"]
                ),
                "strategies": [
                    "Loyalty program enhancement",
                    "Cross-selling campaigns",
                    "Seasonal promotions",
                    "Community engagement programs",
                    "Product recommendation improvements",
                ],
                "expected_retention_rate": 0.85,
                "campaign_cost_per_customer": 25,
            },
        }

        # Calculate business impact
        total_at_risk_customers = (
            strategies["High Risk"]["customers"]
            + strategies["Medium Risk"]["customers"]
        )
        total_campaign_cost = (
            strategies["High Risk"]["customers"]
            * strategies["High Risk"]["campaign_cost_per_customer"]
            + strategies["Medium Risk"]["customers"]
            * strategies["Medium Risk"]["campaign_cost_per_customer"]
            + strategies["Low Risk"]["customers"]
            * strategies["Low Risk"]["campaign_cost_per_customer"]
        )

        # Estimate average customer value
        avg_customer_value = self.churn_features["total_spent"].mean()

        expected_retained_customers = (
            strategies["High Risk"]["customers"]
            * strategies["High Risk"]["expected_retention_rate"]
            + strategies["Medium Risk"]["customers"]
            * strategies["Medium Risk"]["expected_retention_rate"]
            + strategies["Low Risk"]["customers"]
            * strategies["Low Risk"]["expected_retention_rate"]
        )

        expected_revenue_saved = expected_retained_customers * avg_customer_value
        roi = (expected_revenue_saved - total_campaign_cost) / total_campaign_cost

        print(f"\n Retention Strategy Business Impact:")
        print(f"   Total at-risk customers: {total_at_risk_customers:,}")
        print(f"   Expected customers retained: {expected_retained_customers:,.0f}")
        print(f"   Total campaign cost: ${total_campaign_cost:,.0f}")
        print(f"   Expected revenue saved: ${expected_revenue_saved:,.0f}")
        print(f"   ROI: {roi:.1%}")

        return strategies, churn_predictions

    def create_churn_dashboard_data(self):
        """Create data for churn prediction dashboard"""
        churn_predictions = self.predict_churn_risk()

        dashboard_data = {
            "churn_summary": {
                "total_customers": len(churn_predictions),
                "churned_customers": len(
                    churn_predictions[churn_predictions["is_churned"] == 1]
                ),
                "churn_rate": churn_predictions["is_churned"].mean(),
                "high_risk_customers": len(
                    churn_predictions[churn_predictions["risk_category"] == "High Risk"]
                ),
                "medium_risk_customers": len(
                    churn_predictions[
                        churn_predictions["risk_category"] == "Medium Risk"
                    ]
                ),
                "low_risk_customers": len(
                    churn_predictions[churn_predictions["risk_category"] == "Low Risk"]
                ),
            },
            "risk_distribution": churn_predictions["risk_category"]
            .value_counts()
            .to_dict(),
            "high_risk_customers": churn_predictions[
                churn_predictions["risk_category"] == "High Risk"
            ][["customer_id", "churn_probability"]]
            .head(20)
            .to_dict("records"),
        }

        return dashboard_data


def main():
    """Main execution function"""
    print(" CUSTOMER CHURN PREDICTION & RETENTION ENGINE")
    print("=" * 60)

    # Initialize churn prediction engine
    churn_engine = ChurnPredictionEngine(
        transactions_path="data/transactions_real.csv",
        customers_path="data/customers_real.csv",
    )

    # Prepare features and train model
    churn_engine.prepare_churn_features()
    churn_engine.train_churn_model()

    # Generate retention strategies
    strategies, predictions = churn_engine.generate_retention_strategies()

    # Save results
    predictions.to_csv("results/churn_predictions.csv", index=False)

    # Create dashboard data
    dashboard_data = churn_engine.create_churn_dashboard_data()

    import json

    with open("results/churn_dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print("\n Churn prediction analysis completed!")
    print(" Results saved to:")
    print("   - results/churn_predictions.csv")
    print("   - results/churn_dashboard_data.json")


if __name__ == "__main__":
    main()
