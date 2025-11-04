"""
Advanced Inventory Optimization and Demand Forecasting Engine
Addresses critical business issue: Inventory management and demand planning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")


class InventoryOptimizationEngine:
    """
    Advanced Inventory Optimization System for Retail Business

    Business Problems Addressed:
    - Demand forecasting and inventory planning
    - Stockout prevention and overstock reduction
    - Seasonal demand patterns
    - Product lifecycle management
    - Supply chain optimization
    """

    def __init__(self, transactions_path, products_path):
        """Initialize with transaction and product data"""
        self.transactions = pd.read_csv(transactions_path)
        self.products = pd.read_csv(products_path)
        self.demand_data = None
        self.forecast_models = {}
        self.inventory_recommendations = None

    def prepare_demand_data(self):
        """Prepare comprehensive demand data for forecasting"""
        print("🔄 Preparing demand forecasting data...")

        # Convert dates
        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"]
        )

        # Create daily demand data
        daily_demand = (
            self.transactions.groupby(["product_id", "transaction_date"])
            .agg({"quantity": "sum", "total_amount": "sum", "transaction_id": "count"})
            .rename(columns={"transaction_id": "transaction_count"})
            .reset_index()
        )

        # Add product information
        daily_demand = daily_demand.merge(self.products, on="product_id", how="left")

        # Create time-based features
        daily_demand["year"] = daily_demand["transaction_date"].dt.year
        daily_demand["month"] = daily_demand["transaction_date"].dt.month
        daily_demand["day_of_week"] = daily_demand["transaction_date"].dt.dayofweek
        daily_demand["day_of_month"] = daily_demand["transaction_date"].dt.day
        daily_demand["is_weekend"] = (
            daily_demand["day_of_week"].isin([5, 6]).astype(int)
        )
        daily_demand["is_month_start"] = (daily_demand["day_of_month"] <= 7).astype(int)
        daily_demand["is_month_end"] = (daily_demand["day_of_month"] >= 24).astype(int)

        # Seasonal features
        daily_demand["season"] = daily_demand["month"].map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
            }
        )

        # Create complete date range for each product
        date_range = pd.date_range(
            start=daily_demand["transaction_date"].min(),
            end=daily_demand["transaction_date"].max(),
            freq="D",
        )

        # Get all unique products
        products = daily_demand["product_id"].unique()

        # Create complete product-date combinations
        complete_data = []
        for product in products:
            product_data = pd.DataFrame(
                {"product_id": product, "transaction_date": date_range}
            )
            complete_data.append(product_data)

        complete_data = pd.concat(complete_data, ignore_index=True)

        # Merge with actual demand data
        complete_demand = complete_data.merge(
            daily_demand, on=["product_id", "transaction_date"], how="left"
        )
        complete_demand[
            ["quantity", "total_amount", "transaction_count"]
        ] = complete_demand[["quantity", "total_amount", "transaction_count"]].fillna(0)

        # Fill product information
        product_info = self.products.set_index("product_id")
        for col in [
            "product_name",
            "category",
            "subcategory",
            "brand",
            "unit_price",
            "supplier_id",
        ]:
            complete_demand[col] = complete_demand["product_id"].map(product_info[col])

        # Recreate time features for complete data
        complete_demand["year"] = complete_demand["transaction_date"].dt.year
        complete_demand["month"] = complete_demand["transaction_date"].dt.month
        complete_demand["day_of_week"] = complete_demand[
            "transaction_date"
        ].dt.dayofweek
        complete_demand["day_of_month"] = complete_demand["transaction_date"].dt.day
        complete_demand["is_weekend"] = (
            complete_demand["day_of_week"].isin([5, 6]).astype(int)
        )
        complete_demand["is_month_start"] = (
            complete_demand["day_of_month"] <= 7
        ).astype(int)
        complete_demand["is_month_end"] = (
            complete_demand["day_of_month"] >= 24
        ).astype(int)

        complete_demand["season"] = complete_demand["month"].map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
            }
        )

        # Calculate rolling averages and trends
        complete_demand = complete_demand.sort_values(
            ["product_id", "transaction_date"]
        )

        for window in [7, 14, 30]:
            complete_demand[f"quantity_rolling_{window}d"] = complete_demand.groupby(
                "product_id"
            )["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            complete_demand[f"quantity_trend_{window}d"] = complete_demand.groupby(
                "product_id"
            )[f"quantity_rolling_{window}d"].transform(
                lambda x: x.pct_change(periods=window).fillna(0)
            )

        # Calculate demand volatility
        complete_demand["demand_volatility"] = complete_demand.groupby("product_id")[
            "quantity"
        ].transform(lambda x: x.rolling(window=30, min_periods=1).std().fillna(0))

        # Product lifecycle features
        product_stats = (
            complete_demand.groupby("product_id")
            .agg(
                {
                    "quantity": ["sum", "mean", "std", "max"],
                    "transaction_date": ["min", "max"],
                }
            )
            .round(2)
        )

        product_stats.columns = [
            "total_demand",
            "avg_daily_demand",
            "demand_std",
            "max_daily_demand",
            "first_sale_date",
            "last_sale_date",
        ]
        product_stats = product_stats.reset_index()

        # Calculate product age and lifecycle stage
        reference_date = complete_demand["transaction_date"].max()
        product_stats["product_age_days"] = (
            reference_date - product_stats["first_sale_date"]
        ).dt.days
        product_stats["days_since_last_sale"] = (
            reference_date - product_stats["last_sale_date"]
        ).dt.days

        # Classify lifecycle stage
        def classify_lifecycle(row):
            if row["days_since_last_sale"] > 90:
                return "Declining"
            elif row["product_age_days"] < 30:
                return "Introduction"
            elif row["avg_daily_demand"] > row["demand_std"] * 2:
                return "Growth"
            else:
                return "Maturity"

        product_stats["lifecycle_stage"] = product_stats.apply(
            classify_lifecycle, axis=1
        )

        # Merge lifecycle information
        complete_demand = complete_demand.merge(
            product_stats[["product_id", "lifecycle_stage", "avg_daily_demand"]],
            on="product_id",
            how="left",
        )

        self.demand_data = complete_demand

        print(
            f"✅ Demand data prepared for {len(products)} products over {len(date_range)} days"
        )
        print(f"📊 Total demand records: {len(complete_demand):,}")

        return complete_demand

    def train_demand_forecasting_models(self, forecast_horizon=30):
        """Train demand forecasting models for different product categories"""
        print("🤖 Training demand forecasting models...")

        if self.demand_data is None:
            self.prepare_demand_data()

        # Feature columns for modeling
        feature_columns = [
            "month",
            "day_of_week",
            "day_of_month",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "quantity_rolling_7d",
            "quantity_rolling_14d",
            "quantity_rolling_30d",
            "quantity_trend_7d",
            "quantity_trend_14d",
            "quantity_trend_30d",
            "demand_volatility",
            "unit_price",
        ]

        # Encode categorical variables
        category_dummies = pd.get_dummies(
            self.demand_data["category"], prefix="category"
        )
        season_dummies = pd.get_dummies(self.demand_data["season"], prefix="season")
        lifecycle_dummies = pd.get_dummies(
            self.demand_data["lifecycle_stage"], prefix="lifecycle"
        )

        # Combine features
        X_features = pd.concat(
            [
                self.demand_data[feature_columns],
                category_dummies,
                season_dummies,
                lifecycle_dummies,
            ],
            axis=1,
        ).fillna(0)

        y = self.demand_data["quantity"]

        # Split data temporally
        split_date = self.demand_data["transaction_date"].quantile(0.8)
        train_mask = self.demand_data["transaction_date"] <= split_date

        X_train, X_test = X_features[train_mask], X_features[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]

        # Train different models
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "Linear Regression": LinearRegression(),
        }

        best_model = None
        best_score = float("inf")

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"   {name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

            if mae < best_score:
                best_score = mae
                best_model = model
                self.best_model_name = name

        self.forecast_models["demand"] = best_model
        self.feature_columns = X_features.columns.tolist()

        print(f"✅ Best model: {self.best_model_name} (MAE: {best_score:.2f})")

        return best_model, best_score

    def generate_demand_forecasts(self, forecast_days=30):
        """Generate demand forecasts for all products"""
        print(f"📈 Generating {forecast_days}-day demand forecasts...")

        if "demand" not in self.forecast_models:
            self.train_demand_forecasting_models()

        # Get latest data for each product
        latest_data = self.demand_data.groupby("product_id").last().reset_index()

        forecasts = []

        for _, product_row in latest_data.iterrows():
            product_id = product_row["product_id"]
            last_date = product_row["transaction_date"]

            # Generate forecast dates
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1), periods=forecast_days, freq="D"
            )

            for forecast_date in forecast_dates:
                # Create features for forecast date
                forecast_features = {
                    "month": forecast_date.month,
                    "day_of_week": forecast_date.dayofweek,
                    "day_of_month": forecast_date.day,
                    "is_weekend": int(forecast_date.dayofweek in [5, 6]),
                    "is_month_start": int(forecast_date.day <= 7),
                    "is_month_end": int(forecast_date.day >= 24),
                    "quantity_rolling_7d": product_row["quantity_rolling_7d"],
                    "quantity_rolling_14d": product_row["quantity_rolling_14d"],
                    "quantity_rolling_30d": product_row["quantity_rolling_30d"],
                    "quantity_trend_7d": product_row["quantity_trend_7d"],
                    "quantity_trend_14d": product_row["quantity_trend_14d"],
                    "quantity_trend_30d": product_row["quantity_trend_30d"],
                    "demand_volatility": product_row["demand_volatility"],
                    "unit_price": product_row["unit_price"],
                }

                # Add categorical features
                for col in self.feature_columns:
                    if col.startswith(("category_", "season_", "lifecycle_")):
                        if col in [
                            f"category_{product_row['category']}",
                            f"season_{forecast_date.month//3}",
                            f"lifecycle_{product_row['lifecycle_stage']}",
                        ]:
                            forecast_features[col] = 1
                        else:
                            forecast_features[col] = 0

                # Create feature vector
                feature_vector = pd.DataFrame([forecast_features])[
                    self.feature_columns
                ].fillna(0)

                # Make prediction
                predicted_demand = max(
                    0, self.forecast_models["demand"].predict(feature_vector)[0]
                )

                forecasts.append(
                    {
                        "product_id": product_id,
                        "forecast_date": forecast_date,
                        "predicted_demand": predicted_demand,
                        "product_name": product_row["product_name"],
                        "category": product_row["category"],
                        "unit_price": product_row["unit_price"],
                        "current_avg_demand": product_row["avg_daily_demand"],
                    }
                )

        forecast_df = pd.DataFrame(forecasts)

        print(f"✅ Generated forecasts for {len(latest_data)} products")

        return forecast_df

    def optimize_inventory_levels(self, service_level=0.95, lead_time_days=7):
        """Calculate optimal inventory levels and reorder points"""
        print("📦 Optimizing inventory levels...")

        forecasts = self.generate_demand_forecasts(forecast_days=30)

        # Calculate inventory metrics for each product
        inventory_optimization = []

        for product_id in forecasts["product_id"].unique():
            product_forecasts = forecasts[forecasts["product_id"] == product_id]
            product_info = self.demand_data[
                self.demand_data["product_id"] == product_id
            ].iloc[-1]

            # Calculate demand statistics
            avg_daily_demand = product_forecasts["predicted_demand"].mean()
            demand_std = product_info["demand_volatility"]
            max_daily_demand = product_forecasts["predicted_demand"].max()

            # Safety stock calculation (considering service level and demand variability)
            from scipy import stats

            z_score = stats.norm.ppf(service_level)
            safety_stock = z_score * demand_std * np.sqrt(lead_time_days)

            # Reorder point
            reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

            # Economic Order Quantity (simplified)
            annual_demand = avg_daily_demand * 365
            holding_cost_rate = 0.25  # 25% of product value per year
            ordering_cost = 50  # Fixed cost per order

            if annual_demand > 0 and product_info["unit_price"] > 0:
                eoq = np.sqrt(
                    (2 * annual_demand * ordering_cost)
                    / (holding_cost_rate * product_info["unit_price"])
                )
            else:
                eoq = avg_daily_demand * 30  # 30-day supply as fallback

            # Maximum inventory level
            max_inventory = reorder_point + eoq

            # Inventory turnover
            if avg_daily_demand > 0:
                inventory_turnover = annual_demand / (eoq / 2)  # Average inventory
            else:
                inventory_turnover = 0

            # Days of supply
            days_of_supply = eoq / max(avg_daily_demand, 0.1)

            # Risk assessment
            stockout_risk = (
                "High"
                if demand_std > avg_daily_demand
                else "Medium"
                if demand_std > avg_daily_demand * 0.5
                else "Low"
            )
            overstock_risk = (
                "High"
                if days_of_supply > 60
                else "Medium"
                if days_of_supply > 30
                else "Low"
            )

            inventory_optimization.append(
                {
                    "product_id": product_id,
                    "product_name": product_info["product_name"],
                    "category": product_info["category"],
                    "unit_price": product_info["unit_price"],
                    "avg_daily_demand": avg_daily_demand,
                    "demand_volatility": demand_std,
                    "safety_stock": safety_stock,
                    "reorder_point": reorder_point,
                    "economic_order_quantity": eoq,
                    "max_inventory_level": max_inventory,
                    "inventory_turnover": inventory_turnover,
                    "days_of_supply": days_of_supply,
                    "stockout_risk": stockout_risk,
                    "overstock_risk": overstock_risk,
                    "total_30day_demand": product_forecasts["predicted_demand"].sum(),
                }
            )

        self.inventory_recommendations = pd.DataFrame(inventory_optimization)

        print(
            f"✅ Inventory optimization completed for {len(inventory_optimization)} products"
        )

        return self.inventory_recommendations

    def generate_procurement_recommendations(self):
        """Generate procurement recommendations and alerts"""
        print("🚨 Generating procurement recommendations...")

        if self.inventory_recommendations is None:
            self.optimize_inventory_levels()

        # Categorize recommendations
        high_priority = self.inventory_recommendations[
            (self.inventory_recommendations["stockout_risk"] == "High")
            | (
                self.inventory_recommendations["avg_daily_demand"]
                > self.inventory_recommendations["avg_daily_demand"].quantile(0.8)
            )
        ].sort_values("avg_daily_demand", ascending=False)

        medium_priority = self.inventory_recommendations[
            (self.inventory_recommendations["stockout_risk"] == "Medium")
            & (
                self.inventory_recommendations["avg_daily_demand"]
                > self.inventory_recommendations["avg_daily_demand"].quantile(0.5)
            )
        ].sort_values("avg_daily_demand", ascending=False)

        overstock_alerts = self.inventory_recommendations[
            self.inventory_recommendations["overstock_risk"] == "High"
        ].sort_values("days_of_supply", ascending=False)

        recommendations = {
            "high_priority_reorders": {
                "count": len(high_priority),
                "products": high_priority.head(20).to_dict("records"),
                "total_value": (
                    high_priority["economic_order_quantity"]
                    * high_priority["unit_price"]
                ).sum(),
            },
            "medium_priority_reorders": {
                "count": len(medium_priority),
                "products": medium_priority.head(20).to_dict("records"),
                "total_value": (
                    medium_priority["economic_order_quantity"]
                    * medium_priority["unit_price"]
                ).sum(),
            },
            "overstock_alerts": {
                "count": len(overstock_alerts),
                "products": overstock_alerts.head(10).to_dict("records"),
                "total_value": (
                    overstock_alerts["economic_order_quantity"]
                    * overstock_alerts["unit_price"]
                ).sum(),
            },
        }

        # Calculate business impact
        total_products = len(self.inventory_recommendations)
        avg_inventory_turnover = self.inventory_recommendations[
            "inventory_turnover"
        ].mean()

        print(f"\n📊 Procurement Recommendations Summary:")
        print(
            f"   High Priority Reorders: {recommendations['high_priority_reorders']['count']} products"
        )
        print(
            f"   Medium Priority Reorders: {recommendations['medium_priority_reorders']['count']} products"
        )
        print(
            f"   Overstock Alerts: {recommendations['overstock_alerts']['count']} products"
        )
        print(f"   Average Inventory Turnover: {avg_inventory_turnover:.1f}x per year")

        return recommendations


def main():
    """Main execution function"""
    print("🚀 INVENTORY OPTIMIZATION & DEMAND FORECASTING ENGINE")
    print("=" * 60)

    # Initialize inventory optimization engine
    inventory_engine = InventoryOptimizationEngine(
        transactions_path="data/transactions.csv", products_path="data/products.csv"
    )

    # Prepare data and train models
    inventory_engine.prepare_demand_data()
    inventory_engine.train_demand_forecasting_models()

    # Generate forecasts and optimize inventory
    forecasts = inventory_engine.generate_demand_forecasts()
    inventory_recommendations = inventory_engine.optimize_inventory_levels()
    procurement_recommendations = (
        inventory_engine.generate_procurement_recommendations()
    )

    # Save results
    forecasts.to_csv("results/demand_forecasts.csv", index=False)
    inventory_recommendations.to_csv("results/inventory_optimization.csv", index=False)

    import json

    with open("results/procurement_recommendations.json", "w") as f:
        json.dump(procurement_recommendations, f, indent=2, default=str)

    print("\n✅ Inventory optimization analysis completed!")
    print("📁 Results saved to:")
    print("   - results/demand_forecasts.csv")
    print("   - results/inventory_optimization.csv")
    print("   - results/procurement_recommendations.json")


if __name__ == "__main__":
    main()
