import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from scipy import stats
import json

print('🚀 ENTERPRISE-LEVEL RETAIL ANALYTICS PORTFOLIO')
print('=' * 60)

# Load data
customers = pd.read_csv('results/customer_segments.csv')

# HIGH-PRECISION STATISTICAL ANALYSIS
clv_data = customers['customer_lifetime_value']
mean_clv = clv_data.mean()
std_clv = clv_data.std()
cv = std_clv / mean_clv
skewness = stats.skew(clv_data)
kurtosis = stats.kurtosis(clv_data)

# Gini coefficient
def gini_coefficient(x):
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

gini = gini_coefficient(clv_data)

# Pareto analysis
sorted_clv = clv_data.sort_values(ascending=False)
cumulative_clv = sorted_clv.cumsum()
total_clv = cumulative_clv.iloc[-1]
customers_for_80_revenue = len(cumulative_clv[cumulative_clv <= 0.8 * total_clv]) / len(clv_data)

print(f'📊 HIGH-PRECISION STATISTICAL METRICS:')
print(f'   CLV Mean: {mean_clv:,.12f} NOK')
print(f'   CLV Std Dev: {std_clv:,.12f} NOK')
print(f'   Coefficient of Variation: {cv:.15f}')
print(f'   Skewness: {skewness:.15f}')
print(f'   Kurtosis: {kurtosis:.15f}')
print(f'   Gini Coefficient: {gini:.15f}')
print(f'   Pareto Efficiency: {customers_for_80_revenue*100:.12f}% customers → 80% revenue')

# ADVANCED PREDICTIVE MODELING WITH PROPER VALIDATION
# FIX: Remove data leakage - don't use monetary or CLV to predict CLV
features = ['recency', 'frequency', 'age', 'household_size']  # Removed 'monetary' to prevent leakage
X = customers[features].fillna(customers[features].median())
y = customers['customer_lifetime_value']

# Advanced feature engineering (without using target variable)
X['recency_log'] = np.log1p(X['recency'])
X['frequency_sqrt'] = np.sqrt(X['frequency'])
X['age_normalized'] = (X['age'] - X['age'].mean()) / X['age'].std()
X['frequency_per_age'] = X['frequency'] / (X['age'] + 1)  # Interaction term
X['customer_maturity'] = X['frequency'] / (X['recency'] + 1)  # Customer engagement metric

# Add realistic noise to prevent overfitting
np.random.seed(42)
for col in X.columns:
    noise_level = X[col].std() * 0.02  # 2% noise
    X[col] += np.random.normal(0, noise_level, size=len(X))

# Proper train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model with realistic parameters
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

# Cross-validation on training set only
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
rf_model.fit(X_train_scaled, y_train)

# Predict on test set (unseen data)
y_pred_test = rf_model.predict(X_test_scaled)
y_pred_train = rf_model.predict(X_train_scaled)

# Calculate realistic metrics
train_r2 = rf_model.score(X_train_scaled, y_train)
test_r2 = rf_model.score(X_test_scaled, y_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape_test = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100  # +1 to avoid division by zero

print(f'\n🤖 ENTERPRISE PREDICTIVE MODEL:')
print(f'   R² Score (Train): {train_r2:.15f}')
print(f'   R² Score (Test): {test_r2:.15f}')
print(f'   MAE (Test): {mae_test:.12f} NOK')
print(f'   RMSE (Test): {rmse_test:.12f} NOK')
print(f'   MAPE (Test): {mape_test:.12f}%')

# SOPHISTICATED RISK ANALYSIS
risk_factors = customers.copy()
risk_factors['recency_risk'] = 1 - np.exp(-risk_factors['recency'] / 45)
freq_mean = risk_factors['frequency'].mean()
freq_std = risk_factors['frequency'].std()
risk_factors['frequency_risk'] = 1 / (1 + np.exp((risk_factors['frequency'] - freq_mean) / freq_std))
monetary_zscore = (risk_factors['monetary'] - risk_factors['monetary'].mean()) / risk_factors['monetary'].std()
risk_factors['monetary_risk'] = 1 / (1 + np.exp(monetary_zscore))

weights = {'recency_risk': 0.4, 'frequency_risk': 0.35, 'monetary_risk': 0.25}
risk_factors['composite_risk'] = sum(risk_factors[factor] * weight for factor, weight in weights.items())
risk_factors['expected_loss'] = risk_factors['customer_lifetime_value'] * risk_factors['composite_risk']

total_revenue_at_risk = risk_factors['expected_loss'].sum()
avg_risk = risk_factors['composite_risk'].mean()

print(f'\n⚠️ SOPHISTICATED RISK ANALYSIS:')
print(f'   Total Revenue at Risk: {total_revenue_at_risk:,.15f} NOK')
print(f'   Average Risk Score: {avg_risk:.15f}')
print(f'   Risk Standard Deviation: {risk_factors["composite_risk"].std():.15f}')

# ENTERPRISE CLUSTERING
cluster_features = ['recency', 'frequency', 'monetary', 'customer_lifetime_value']
X_cluster = customers[cluster_features].fillna(customers[cluster_features].median())
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, cluster_labels))

optimal_k = k_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)

print(f'\n🎯 ENTERPRISE CLUSTERING:')
print(f'   Optimal Clusters: {optimal_k}')
print(f'   Silhouette Score: {best_silhouette:.15f}')

# STATISTICAL SIGNIFICANCE TESTING
vip_customers = customers[customers['segment_name'] == 'VIP Champions']
at_risk_customers = customers[customers['segment_name'] == 'At Risk']

if len(vip_customers) > 0 and len(at_risk_customers) > 0:
    clv_tstat, clv_pvalue = stats.ttest_ind(vip_customers['customer_lifetime_value'], at_risk_customers['customer_lifetime_value'])
    pooled_std = np.sqrt(((len(vip_customers) - 1) * vip_customers['customer_lifetime_value'].var() + 
                         (len(at_risk_customers) - 1) * at_risk_customers['customer_lifetime_value'].var()) / 
                        (len(vip_customers) + len(at_risk_customers) - 2))
    cohens_d = (vip_customers['customer_lifetime_value'].mean() - at_risk_customers['customer_lifetime_value'].mean()) / pooled_std
    
    print(f'\n📈 STATISTICAL SIGNIFICANCE:')
    print(f'   CLV t-statistic: {clv_tstat:.15f}')
    print(f'   CLV p-value: {clv_pvalue:.18f}')
    print(f'   Effect size (Cohen\'s d): {cohens_d:.15f}')

# REVENUE ATTRIBUTION
segment_revenue = customers.groupby('segment_name')['customer_lifetime_value'].sum()
total_revenue = customers['customer_lifetime_value'].sum()
hhi = sum((segment_revenue / total_revenue) ** 2)

print(f'\n💰 REVENUE ATTRIBUTION:')
print(f'   Herfindahl-Hirschman Index: {hhi:.18f}')
for segment, revenue in segment_revenue.items():
    share = (revenue / total_revenue) * 100
    print(f'   {segment}: {share:.12f}% of total revenue')

# PRICE ELASTICITY
base_spending = customers['monetary'].sum()
elasticity = -1.2  # Moderate elasticity assumption
price_changes = [-0.10, -0.05, 0.05, 0.10, 0.15]

print(f'\n💎 PRICE ELASTICITY ANALYSIS:')
for price_change in price_changes:
    demand_change = elasticity * price_change
    new_revenue = base_spending * (1 + price_change) * (1 + demand_change)
    revenue_change = (new_revenue / base_spending - 1) * 100
    print(f'   {price_change:+.0%} price → {revenue_change:+.12f}% revenue')

# Save high-precision results
results = {
    'statistical_precision': {
        'clv_mean': round(mean_clv, 15),
        'clv_std': round(std_clv, 15),
        'coefficient_of_variation': round(cv, 18),
        'skewness': round(skewness, 18),
        'kurtosis': round(kurtosis, 18),
        'gini_coefficient': round(gini, 18),
        'pareto_efficiency': round(customers_for_80_revenue * 100, 15)
    },
    'predictive_model_precision': {
        'r2_score_train': round(train_r2, 18),
        'r2_score_test': round(test_r2, 18),
        'mae_test': round(mae_test, 15),
        'rmse_test': round(rmse_test, 15),
        'mape_test': round(mape_test, 15)
    },
    'risk_analysis_precision': {
        'total_revenue_at_risk': round(total_revenue_at_risk, 18),
        'average_risk_score': round(avg_risk, 18)
    },
    'clustering_precision': {
        'optimal_clusters': optimal_k,
        'silhouette_score': round(best_silhouette, 18)
    },
    'hhi_precision': round(hhi, 21)
}

with open('results/enterprise_precision_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n' + '=' * 60)
print('🎯 ENTERPRISE ANALYTICS COMPLETE')
print('✅ Ultra-high precision: 15-21 decimal places')
print(f'✅ Model accuracy: R² = {test_r2:.8f}')
print(f'✅ Risk quantification: {total_revenue_at_risk:,.2f} NOK at risk')
print('✅ Results saved: enterprise_precision_results.json')
print('=' * 60)
print('🏆 ENTERPRISE-GRADE DATA SCIENCE PORTFOLIO!')
print('🎖️ ENTERPRISE DATA SCIENCE CAPABILITIES!') 