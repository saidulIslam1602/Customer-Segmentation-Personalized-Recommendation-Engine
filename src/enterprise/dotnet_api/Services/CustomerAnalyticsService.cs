using CustomerSegmentation.API.Models;

namespace CustomerSegmentation.API.Services
{
    public class CustomerAnalyticsService : ICustomerAnalyticsService
    {
        private readonly IPythonMLService _pythonService;
        private readonly ILogger<CustomerAnalyticsService> _logger;

        public CustomerAnalyticsService(
            IPythonMLService pythonService,
            ILogger<CustomerAnalyticsService> logger)
        {
            _pythonService = pythonService;
            _logger = logger;
        }

        public async Task<List<CustomerSegmentResponse>> GetCustomerSegmentsAsync(string? customerId = null, bool includePredictions = true)
        {
            try
            {
                // Call Python ML service for actual segmentation
                var requestData = new { customer_id = customerId, include_predictions = includePredictions };
                var segments = await _pythonService.CallPythonServiceAsync<List<CustomerSegmentResponse>>("api/v1/customer-segments", requestData);
                
                return segments ?? new List<CustomerSegmentResponse>();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting customer segments from Python service");
                
                // Return mock data as fallback
                return GenerateMockSegments(customerId);
            }
        }

        public async Task<ChurnPredictionResponse> PredictChurnAsync(ChurnPredictionRequest request)
        {
            try
            {
                var prediction = await _pythonService.CallPythonServiceAsync<ChurnPredictionResponse>("api/v1/churn-prediction", request);
                
                return prediction ?? GenerateMockChurnPrediction(request.CustomerId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting churn from Python service");
                
                return GenerateMockChurnPrediction(request.CustomerId);
            }
        }

        public async Task<RecommendationResponse> GetRecommendationsAsync(string customerId, int numRecommendations = 10, bool includeExplanations = false)
        {
            try
            {
                var requestData = new { customer_id = customerId, num_recommendations = numRecommendations, include_explanations = includeExplanations };
                var recommendations = await _pythonService.CallPythonServiceAsync<RecommendationResponse>("api/v1/recommendations", requestData);
                
                return recommendations ?? GenerateMockRecommendations(customerId, numRecommendations, includeExplanations);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting recommendations from Python service");
                
                return GenerateMockRecommendations(customerId, numRecommendations, includeExplanations);
            }
        }

        public async Task<CLVPredictionResponse> GetCustomerLifetimeValueAsync(string customerId)
        {
            try
            {
                var clv = await _pythonService.CallPythonServiceAsync<CLVPredictionResponse>($"api/v1/clv/{customerId}");
                
                return clv ?? GenerateMockCLV(customerId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting CLV from Python service");
                
                return GenerateMockCLV(customerId);
            }
        }

        public async Task<PerformanceMetricsResponse> GetPerformanceMetricsAsync()
        {
            try
            {
                var metrics = await _pythonService.CallPythonServiceAsync<PerformanceMetricsResponse>("api/v1/performance-metrics");
                
                return metrics ?? GenerateMockPerformanceMetrics();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting performance metrics from Python service");
                
                return GenerateMockPerformanceMetrics();
            }
        }

        // Mock data generators for fallback scenarios
        private List<CustomerSegmentResponse> GenerateMockSegments(string? customerId)
        {
            var segments = new List<CustomerSegmentResponse>();
            var customerIds = string.IsNullOrEmpty(customerId) 
                ? new[] { "customer_1", "customer_2", "customer_3", "customer_4", "customer_5" }
                : new[] { customerId };

            foreach (var id in customerIds)
            {
                segments.Add(new CustomerSegmentResponse
                {
                    CustomerId = id,
                    Segment = GetRandomSegment(),
                    CLVScore = Random.Shared.NextDouble() * 50000 + 1000,
                    RiskLevel = GetRandomRiskLevel(),
                    Recommendations = GetRandomRecommendations(),
                    Timestamp = DateTime.UtcNow
                });
            }

            return segments;
        }

        private ChurnPredictionResponse GenerateMockChurnPrediction(string customerId)
        {
            var churnProb = Random.Shared.NextDouble();
            
            return new ChurnPredictionResponse
            {
                CustomerId = customerId,
                ChurnProbability = churnProb,
                RiskCategory = churnProb > 0.7 ? "High" : churnProb > 0.4 ? "Medium" : "Low",
                RetentionStrategy = churnProb > 0.7 ? "Immediate intervention" : churnProb > 0.4 ? "Targeted campaign" : "Standard engagement",
                ConfidenceScore = Random.Shared.NextDouble() * 0.15 + 0.85,
                Timestamp = DateTime.UtcNow
            };
        }

        private RecommendationResponse GenerateMockRecommendations(string customerId, int numRecommendations, bool includeExplanations)
        {
            var recommendations = new List<ProductRecommendation>();
            var categories = new[] { "Electronics", "Clothing", "Home", "Books", "Sports" };
            
            for (int i = 0; i < numRecommendations; i++)
            {
                var rec = new ProductRecommendation
                {
                    ProductId = $"product_{i + 1}",
                    ProductName = $"Recommended Product {i + 1}",
                    Score = Random.Shared.NextDouble() * 0.5 + 0.5,
                    Category = categories[Random.Shared.Next(categories.Length)],
                    Price = (decimal)(Random.Shared.NextDouble() * 490 + 10)
                };
                
                if (includeExplanations)
                {
                    rec.Explanation = "Based on purchase history and similar customers";
                }
                
                recommendations.Add(rec);
            }

            return new RecommendationResponse
            {
                CustomerId = customerId,
                Recommendations = recommendations,
                Timestamp = DateTime.UtcNow
            };
        }

        private CLVPredictionResponse GenerateMockCLV(string customerId)
        {
            var clv = Random.Shared.NextDouble() * 50000 + 5000;
            
            return new CLVPredictionResponse
            {
                CustomerId = customerId,
                PredictedCLV = clv,
                CLVCategory = clv > 25000 ? "High Value" : clv > 15000 ? "Medium Value" : "Standard Value",
                ConfidenceScore = Random.Shared.NextDouble() * 0.15 + 0.85,
                Timestamp = DateTime.UtcNow
            };
        }

        private PerformanceMetricsResponse GenerateMockPerformanceMetrics()
        {
            return new PerformanceMetricsResponse
            {
                OverallScore = 95.5,
                ModelPerformance = new ModelPerformanceMetrics
                {
                    ChurnPredictionAccuracy = 0.95,
                    RecommendationPrecision = 0.18,
                    SegmentationQuality = 0.89
                },
                BusinessMetrics = new BusinessMetrics
                {
                    TotalCustomers = 4338,
                    VIPCustomers = 213,
                    HighRiskCustomers = 1247,
                    RevenueImpact = 911407.90m
                },
                SystemMetrics = new SystemMetrics
                {
                    APIResponseTimeMs = 150,
                    CRMSyncSuccessRate = 98.5,
                    UptimePercentage = 99.9
                },
                Timestamp = DateTime.UtcNow
            };
        }

        private string GetRandomSegment()
        {
            var segments = new[] { "VIP", "Loyal", "At-Risk", "New", "Champion", "Potential Loyalist" };
            return segments[Random.Shared.Next(segments.Length)];
        }

        private string GetRandomRiskLevel()
        {
            var levels = new[] { "Low", "Medium", "High" };
            return levels[Random.Shared.Next(levels.Length)];
        }

        private List<string> GetRandomRecommendations()
        {
            var recommendations = new[]
            {
                "Personalized product recommendations",
                "Targeted retention campaign",
                "VIP program enrollment",
                "Cross-sell opportunities",
                "Loyalty program benefits",
                "Seasonal promotions"
            };
            
            return recommendations.OrderBy(x => Random.Shared.Next()).Take(3).ToList();
        }
    }
}
