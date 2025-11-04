"""
Advanced Pricing Optimization and Price Elasticity Analysis Engine
Addresses critical business issue: Dynamic pricing and revenue optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings("ignore")


class PricingOptimizationEngine:
    """
    Advanced Pricing Optimization System for Retail Business

    Business Problems Addressed:
    - Dynamic pricing strategies
    - Price elasticity analysis
    - Revenue optimization
    - Competitive pricing
    - Promotional pricing effectiveness
    """

    def __init__(self, transactions_path, products_path):
        """Initialize with transaction and product data"""
        self.transactions = pd.read_csv(transactions_path)
        self.products = pd.read_csv(products_path)
        self.pricing_data = None
        self.elasticity_models = {}
        self.pricing_recommendations = None

    def prepare_pricing_data(self):
        """Prepare comprehensive pricing analysis data"""
        print("🔄 Preparing pricing analysis data...")

        # Convert dates
        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"]
        )

        # Calculate effective price (considering discounts)
        self.transactions["effective_price"] = self.transactions["unit_price"] - (
            self.transactions["discount_amount"] / self.transactions["quantity"]
        ).fillna(0)

        # Weekly aggregation for price elasticity analysis
        self.transactions["year_week"] = self.transactions[
            "transaction_date"
        ].dt.strftime("%Y-%U")

        weekly_data = (
            self.transactions.groupby(["product_id", "year_week"])
            .agg(
                {
                    "quantity": "sum",
                    "total_amount": "sum",
                    "effective_price": "mean",
                    "unit_price": "mean",
                    "discount_amount": "sum",
                    "transaction_id": "count",
                    "transaction_date": "first",
                }
            )
            .rename(columns={"transaction_id": "transaction_count"})
            .reset_index()
        )

        # Add product information
        weekly_data = weekly_data.merge(self.products, on="product_id", how="left")

        # Calculate price metrics
        weekly_data["discount_percentage"] = (
            weekly_data["discount_amount"] / weekly_data["total_amount"]
        ).fillna(0)
        weekly_data["revenue"] = (
            weekly_data["quantity"] * weekly_data["effective_price"]
        )
        weekly_data["units_per_transaction"] = (
            weekly_data["quantity"] / weekly_data["transaction_count"]
        )

        # Time-based features
        weekly_data["transaction_date"] = pd.to_datetime(
            weekly_data["transaction_date"]
        )
        weekly_data["month"] = weekly_data["transaction_date"].dt.month
        weekly_data["quarter"] = weekly_data["transaction_date"].dt.quarter
        weekly_data["is_holiday_season"] = (
            weekly_data["month"].isin([11, 12]).astype(int)
        )
        weekly_data["is_summer"] = weekly_data["month"].isin([6, 7, 8]).astype(int)

        # Calculate price changes and trends
        weekly_data = weekly_data.sort_values(["product_id", "transaction_date"])

        weekly_data["price_change"] = (
            weekly_data.groupby("product_id")["effective_price"].pct_change().fillna(0)
        )
        weekly_data["price_trend_4w"] = weekly_data.groupby("product_id")[
            "effective_price"
        ].transform(
            lambda x: x.rolling(window=4, min_periods=1).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
            )
        )

        # Competitive pricing features (simplified)
        category_avg_price = (
            weekly_data.groupby(["category", "year_week"])["effective_price"]
            .mean()
            .reset_index()
        )
        category_avg_price.columns = ["category", "year_week", "category_avg_price"]
        weekly_data = weekly_data.merge(
            category_avg_price, on=["category", "year_week"], how="left"
        )

        weekly_data["price_vs_category_avg"] = (
            weekly_data["effective_price"] / weekly_data["category_avg_price"]
        ) - 1
        weekly_data["is_premium_priced"] = (
            weekly_data["price_vs_category_avg"] > 0.1
        ).astype(int)
        weekly_data["is_discount_priced"] = (
            weekly_data["price_vs_category_avg"] < -0.1
        ).astype(int)

        # Calculate demand elasticity features
        for window in [4, 8, 12]:  # 4, 8, 12 weeks
            weekly_data[f"avg_quantity_{window}w"] = weekly_data.groupby("product_id")[
                "quantity"
            ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            weekly_data[f"avg_price_{window}w"] = weekly_data.groupby("product_id")[
                "effective_price"
            ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

        # Product lifecycle and performance metrics
        product_performance = (
            weekly_data.groupby("product_id")
            .agg(
                {
                    "quantity": ["sum", "mean", "std"],
                    "revenue": ["sum", "mean"],
                    "effective_price": ["mean", "std", "min", "max"],
                    "transaction_count": "sum",
                    "transaction_date": ["min", "max"],
                }
            )
            .round(2)
        )

        product_performance.columns = [
            "total_quantity",
            "avg_weekly_quantity",
            "quantity_volatility",
            "total_revenue",
            "avg_weekly_revenue",
            "avg_price",
            "price_volatility",
            "min_price",
            "max_price",
            "total_transactions",
            "first_sale_date",
            "last_sale_date",
        ]
        product_performance = product_performance.reset_index()

        # Calculate price elasticity indicators
        reference_date = weekly_data["transaction_date"].max()
        product_performance["weeks_on_market"] = (
            (reference_date - product_performance["first_sale_date"]).dt.days / 7
        ).round(0)
        product_performance["price_range"] = (
            product_performance["max_price"] - product_performance["min_price"]
        )
        product_performance["price_coefficient_variation"] = (
            product_performance["price_volatility"] / product_performance["avg_price"]
        )

        # Merge performance metrics
        weekly_data = weekly_data.merge(
            product_performance[
                [
                    "product_id",
                    "weeks_on_market",
                    "price_range",
                    "price_coefficient_variation",
                ]
            ],
            on="product_id",
            how="left",
        )

        self.pricing_data = weekly_data

        print(f"✅ Pricing data prepared: {len(weekly_data)} product-week observations")
        print(f"📊 Products analyzed: {weekly_data['product_id'].nunique()}")
        print(
            f"📅 Time period: {weekly_data['transaction_date'].min().date()} to {weekly_data['transaction_date'].max().date()}"
        )

        return weekly_data

    def calculate_price_elasticity(self):
        """Calculate price elasticity of demand for each product"""
        print("📈 Calculating price elasticity of demand...")

        if self.pricing_data is None:
            self.prepare_pricing_data()

        elasticity_results = []

        # Calculate elasticity for each product with sufficient data
        for product_id in self.pricing_data["product_id"].unique():
            product_data = self.pricing_data[
                self.pricing_data["product_id"] == product_id
            ].copy()

            # Need at least 8 weeks of data for reliable elasticity calculation
            if len(product_data) < 8:
                continue

            # Remove outliers (quantities beyond 3 standard deviations)
            q_mean = product_data["quantity"].mean()
            q_std = product_data["quantity"].std()
            product_data = product_data[
                (product_data["quantity"] >= q_mean - 3 * q_std)
                & (product_data["quantity"] <= q_mean + 3 * q_std)
            ]

            if len(product_data) < 6:
                continue

            # Log-log regression for elasticity calculation
            # log(Quantity) = α + β * log(Price) + controls
            product_data["log_quantity"] = np.log(product_data["quantity"] + 1)
            product_data["log_price"] = np.log(product_data["effective_price"])

            # Control variables
            control_vars = [
                "month",
                "is_holiday_season",
                "discount_percentage",
                "price_vs_category_avg",
            ]

            # Prepare features
            X_features = product_data[["log_price"] + control_vars].fillna(0)
            y = product_data["log_quantity"]

            # Fit elasticity model
            try:
                model = LinearRegression()
                model.fit(X_features, y)

                # Price elasticity is the coefficient of log_price
                price_elasticity = model.coef_[0]
                r2_score_val = model.score(X_features, y)

                # Additional elasticity calculations
                # Point elasticity at mean price and quantity
                mean_price = product_data["effective_price"].mean()
                mean_quantity = product_data["quantity"].mean()

                # Arc elasticity (using price range)
                if product_data["effective_price"].std() > 0:
                    price_corr = product_data["effective_price"].corr(
                        product_data["quantity"]
                    )

                    # Simple arc elasticity calculation
                    price_change_pct = (
                        product_data["effective_price"].max()
                        - product_data["effective_price"].min()
                    ) / product_data["effective_price"].mean()
                    quantity_change_pct = (
                        product_data["quantity"].max() - product_data["quantity"].min()
                    ) / product_data["quantity"].mean()

                    if price_change_pct != 0:
                        arc_elasticity = quantity_change_pct / price_change_pct
                    else:
                        arc_elasticity = 0
                else:
                    arc_elasticity = 0
                    price_corr = 0

                # Classify elasticity
                if abs(price_elasticity) > 1:
                    elasticity_type = "Elastic"
                elif abs(price_elasticity) > 0.5:
                    elasticity_type = "Moderately Elastic"
                else:
                    elasticity_type = "Inelastic"

                # Get product info
                product_info = product_data.iloc[0]

                elasticity_results.append(
                    {
                        "product_id": product_id,
                        "product_name": product_info["product_name"],
                        "category": product_info["category"],
                        "price_elasticity": price_elasticity,
                        "arc_elasticity": arc_elasticity,
                        "elasticity_type": elasticity_type,
                        "price_quantity_correlation": price_corr,
                        "model_r2": r2_score_val,
                        "avg_price": mean_price,
                        "avg_quantity": mean_quantity,
                        "price_volatility": product_info["price_coefficient_variation"],
                        "weeks_analyzed": len(product_data),
                        "total_revenue": product_data["revenue"].sum(),
                        "revenue_potential": "High"
                        if abs(price_elasticity) > 0.8
                        and mean_quantity > product_data["quantity"].median()
                        else "Medium"
                        if abs(price_elasticity) > 0.5
                        else "Low",
                    }
                )

            except Exception as e:
                print(
                    f"   Warning: Could not calculate elasticity for product {product_id}: {str(e)}"
                )
                continue

        elasticity_df = pd.DataFrame(elasticity_results)

        if len(elasticity_df) > 0:
            print(f"✅ Price elasticity calculated for {len(elasticity_df)} products")
            print(f"📊 Elasticity distribution:")
            print(
                f"   Elastic products: {len(elasticity_df[elasticity_df['elasticity_type'] == 'Elastic'])}"
            )
            print(
                f"   Moderately Elastic: {len(elasticity_df[elasticity_df['elasticity_type'] == 'Moderately Elastic'])}"
            )
            print(
                f"   Inelastic products: {len(elasticity_df[elasticity_df['elasticity_type'] == 'Inelastic'])}"
            )

        return elasticity_df

    def optimize_pricing_strategy(self, target_metric="revenue"):
        """Generate optimal pricing strategies based on elasticity analysis"""
        print("💰 Optimizing pricing strategies...")

        elasticity_data = self.calculate_price_elasticity()

        if len(elasticity_data) == 0:
            print("❌ No elasticity data available for pricing optimization")
            return None

        pricing_strategies = []

        for _, product in elasticity_data.iterrows():
            current_price = product["avg_price"]
            elasticity = product["price_elasticity"]
            current_quantity = product["avg_quantity"]

            # Calculate optimal price based on elasticity
            if target_metric == "revenue":
                # For revenue maximization: optimal price when elasticity = -1
                if elasticity < -0.1:  # Elastic product
                    # Revenue maximizing price change
                    optimal_price_multiplier = 1 / (1 + elasticity)
                    suggested_price = current_price * optimal_price_multiplier

                    # Limit price changes to reasonable bounds (±30%)
                    suggested_price = max(
                        current_price * 0.7, min(current_price * 1.3, suggested_price)
                    )

                    price_change_pct = (suggested_price - current_price) / current_price

                    # Estimate impact
                    estimated_quantity_change = elasticity * price_change_pct
                    estimated_new_quantity = current_quantity * (
                        1 + estimated_quantity_change
                    )
                    estimated_revenue_change = (
                        suggested_price * estimated_new_quantity
                    ) / (current_price * current_quantity) - 1

                else:  # Inelastic product
                    # For inelastic products, consider small price increases
                    suggested_price = current_price * 1.05  # 5% increase
                    price_change_pct = 0.05
                    estimated_quantity_change = elasticity * price_change_pct
                    estimated_new_quantity = current_quantity * (
                        1 + estimated_quantity_change
                    )
                    estimated_revenue_change = (
                        suggested_price * estimated_new_quantity
                    ) / (current_price * current_quantity) - 1

            # Determine pricing strategy
            if price_change_pct > 0.02:
                strategy = "Price Increase"
                rationale = f"Inelastic demand (elasticity: {elasticity:.2f}) allows for price increases"
            elif price_change_pct < -0.02:
                strategy = "Price Decrease"
                rationale = f"Elastic demand (elasticity: {elasticity:.2f}) suggests lower prices increase revenue"
            else:
                strategy = "Maintain Current Price"
                rationale = "Current price is near optimal"

            # Risk assessment
            if abs(elasticity) > 1.5:
                risk_level = "High"
            elif abs(elasticity) > 0.8:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            # Confidence based on model quality
            if product["model_r2"] > 0.7 and product["weeks_analyzed"] > 12:
                confidence = "High"
            elif product["model_r2"] > 0.5 and product["weeks_analyzed"] > 8:
                confidence = "Medium"
            else:
                confidence = "Low"

            pricing_strategies.append(
                {
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "current_price": current_price,
                    "suggested_price": suggested_price,
                    "price_change_pct": price_change_pct,
                    "strategy": strategy,
                    "rationale": rationale,
                    "price_elasticity": elasticity,
                    "elasticity_type": product["elasticity_type"],
                    "estimated_revenue_change": estimated_revenue_change,
                    "estimated_quantity_change": estimated_quantity_change,
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "current_weekly_revenue": product["total_revenue"]
                    / product["weeks_analyzed"],
                    "revenue_potential": product["revenue_potential"],
                }
            )

        pricing_strategies_df = pd.DataFrame(pricing_strategies)

        # Prioritize recommendations
        pricing_strategies_df["priority_score"] = (
            (pricing_strategies_df["estimated_revenue_change"].abs() * 0.4)
            + (
                pricing_strategies_df["current_weekly_revenue"]
                / pricing_strategies_df["current_weekly_revenue"].max()
                * 0.3
            )
            + ((pricing_strategies_df["confidence"] == "High").astype(int) * 0.3)
        )

        pricing_strategies_df = pricing_strategies_df.sort_values(
            "priority_score", ascending=False
        )

        self.pricing_recommendations = pricing_strategies_df

        # Summary statistics
        total_potential_revenue_impact = (
            pricing_strategies_df["estimated_revenue_change"]
            * pricing_strategies_df["current_weekly_revenue"]
        ).sum() * 52  # Annualized

        high_impact_products = len(
            pricing_strategies_df[
                pricing_strategies_df["estimated_revenue_change"].abs() > 0.1
            ]
        )

        print(
            f"✅ Pricing strategies generated for {len(pricing_strategies_df)} products"
        )
        print(
            f"💰 Estimated annual revenue impact: ${total_potential_revenue_impact:,.0f}"
        )
        print(f"🎯 High-impact opportunities: {high_impact_products} products")

        return pricing_strategies_df

    def generate_promotional_pricing_analysis(self):
        """Analyze promotional pricing effectiveness"""
        print("🎉 Analyzing promotional pricing effectiveness...")

        if self.pricing_data is None:
            self.prepare_pricing_data()

        # Analyze discount effectiveness
        promotional_analysis = []

        for product_id in self.pricing_data["product_id"].unique():
            product_data = self.pricing_data[
                self.pricing_data["product_id"] == product_id
            ].copy()

            if len(product_data) < 4:
                continue

            # Separate promotional and regular periods
            promotional_periods = product_data[
                product_data["discount_percentage"] > 0.05
            ]  # >5% discount
            regular_periods = product_data[product_data["discount_percentage"] <= 0.05]

            if len(promotional_periods) == 0 or len(regular_periods) == 0:
                continue

            # Calculate metrics
            promo_avg_quantity = promotional_periods["quantity"].mean()
            regular_avg_quantity = regular_periods["quantity"].mean()

            promo_avg_revenue = promotional_periods["revenue"].mean()
            regular_avg_revenue = regular_periods["revenue"].mean()

            promo_avg_discount = promotional_periods["discount_percentage"].mean()

            # Calculate lift
            quantity_lift = (
                (promo_avg_quantity - regular_avg_quantity) / regular_avg_quantity
                if regular_avg_quantity > 0
                else 0
            )
            revenue_lift = (
                (promo_avg_revenue - regular_avg_revenue) / regular_avg_revenue
                if regular_avg_revenue > 0
                else 0
            )

            # ROI calculation
            discount_cost = promo_avg_revenue * promo_avg_discount
            additional_revenue = promo_avg_revenue - regular_avg_revenue
            promo_roi = (
                (additional_revenue - discount_cost) / discount_cost
                if discount_cost > 0
                else 0
            )

            # Get product info
            product_info = product_data.iloc[0]

            promotional_analysis.append(
                {
                    "product_id": product_id,
                    "product_name": product_info["product_name"],
                    "category": product_info["category"],
                    "promotional_periods": len(promotional_periods),
                    "avg_discount_percentage": promo_avg_discount,
                    "quantity_lift": quantity_lift,
                    "revenue_lift": revenue_lift,
                    "promotional_roi": promo_roi,
                    "promo_effectiveness": "High"
                    if quantity_lift > 0.3 and promo_roi > 0
                    else "Medium"
                    if quantity_lift > 0.1
                    else "Low",
                    "regular_avg_quantity": regular_avg_quantity,
                    "promo_avg_quantity": promo_avg_quantity,
                    "regular_avg_revenue": regular_avg_revenue,
                    "promo_avg_revenue": promo_avg_revenue,
                }
            )

        promotional_df = pd.DataFrame(promotional_analysis)

        if len(promotional_df) > 0:
            print(
                f"✅ Promotional analysis completed for {len(promotional_df)} products"
            )
            print(
                f"📊 Average quantity lift: {promotional_df['quantity_lift'].mean():.1%}"
            )
            print(
                f"💰 Average revenue lift: {promotional_df['revenue_lift'].mean():.1%}"
            )
            print(
                f"📈 Average promotional ROI: {promotional_df['promotional_roi'].mean():.1%}"
            )

        return promotional_df

    def create_pricing_dashboard_data(self):
        """Create comprehensive data for pricing dashboard"""
        elasticity_data = self.calculate_price_elasticity()
        pricing_strategies = self.optimize_pricing_strategy()
        promotional_analysis = self.generate_promotional_pricing_analysis()

        dashboard_data = {
            "pricing_summary": {
                "total_products_analyzed": len(elasticity_data)
                if elasticity_data is not None
                else 0,
                "elastic_products": len(
                    elasticity_data[elasticity_data["elasticity_type"] == "Elastic"]
                )
                if elasticity_data is not None
                else 0,
                "inelastic_products": len(
                    elasticity_data[elasticity_data["elasticity_type"] == "Inelastic"]
                )
                if elasticity_data is not None
                else 0,
                "high_revenue_potential": len(
                    pricing_strategies[
                        pricing_strategies["revenue_potential"] == "High"
                    ]
                )
                if pricing_strategies is not None
                else 0,
                "avg_price_elasticity": elasticity_data["price_elasticity"].mean()
                if elasticity_data is not None
                else 0,
            },
            "top_pricing_opportunities": pricing_strategies.head(20).to_dict("records")
            if pricing_strategies is not None
            else [],
            "promotional_insights": {
                "avg_quantity_lift": promotional_analysis["quantity_lift"].mean()
                if promotional_analysis is not None
                else 0,
                "avg_revenue_lift": promotional_analysis["revenue_lift"].mean()
                if promotional_analysis is not None
                else 0,
                "high_effectiveness_promos": len(
                    promotional_analysis[
                        promotional_analysis["promo_effectiveness"] == "High"
                    ]
                )
                if promotional_analysis is not None
                else 0,
            },
        }

        return dashboard_data


def main():
    """Main execution function"""
    print("🚀 PRICING OPTIMIZATION & ELASTICITY ANALYSIS ENGINE")
    print("=" * 60)

    # Initialize pricing optimization engine
    pricing_engine = PricingOptimizationEngine(
        transactions_path="data/transactions_real.csv",
        products_path="data/products_real.csv",
    )

    # Prepare data and calculate elasticity
    pricing_engine.prepare_pricing_data()
    elasticity_data = pricing_engine.calculate_price_elasticity()

    # Generate pricing strategies
    pricing_strategies = pricing_engine.optimize_pricing_strategy()
    promotional_analysis = pricing_engine.generate_promotional_pricing_analysis()

    # Save results
    if elasticity_data is not None:
        elasticity_data.to_csv("results/price_elasticity_analysis.csv", index=False)

    if pricing_strategies is not None:
        pricing_strategies.to_csv(
            "results/pricing_optimization_strategies.csv", index=False
        )

    if promotional_analysis is not None:
        promotional_analysis.to_csv(
            "results/promotional_pricing_analysis.csv", index=False
        )

    # Create dashboard data
    dashboard_data = pricing_engine.create_pricing_dashboard_data()

    import json

    with open("results/pricing_dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print("\n✅ Pricing optimization analysis completed!")
    print("📁 Results saved to:")
    print("   - results/price_elasticity_analysis.csv")
    print("   - results/pricing_optimization_strategies.csv")
    print("   - results/promotional_pricing_analysis.csv")
    print("   - results/pricing_dashboard_data.json")


if __name__ == "__main__":
    main()
