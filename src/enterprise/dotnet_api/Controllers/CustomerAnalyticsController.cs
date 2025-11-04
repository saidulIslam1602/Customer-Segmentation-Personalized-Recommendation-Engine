using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using CustomerSegmentation.API.Services;
using CustomerSegmentation.API.Models;

namespace CustomerSegmentation.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class CustomerAnalyticsController : ControllerBase
    {
        private readonly ICustomerAnalyticsService _analyticsService;
        private readonly ILogger<CustomerAnalyticsController> _logger;

        public CustomerAnalyticsController(
            ICustomerAnalyticsService analyticsService,
            ILogger<CustomerAnalyticsController> logger)
        {
            _analyticsService = analyticsService;
            _logger = logger;
        }

        /// <summary>
        /// Get customer segments with ML-driven insights
        /// </summary>
        [HttpGet("segments")]
        public async Task<ActionResult<CustomerSegmentResponse>> GetCustomerSegments(
            [FromQuery] string? customerId = null,
            [FromQuery] bool includePredictions = true)
        {
            try
            {
                _logger.LogInformation("Getting customer segments for customer: {CustomerId}", customerId);
                
                var segments = await _analyticsService.GetCustomerSegmentsAsync(customerId, includePredictions);
                return Ok(segments);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting customer segments");
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Predict customer churn probability
        /// </summary>
        [HttpPost("churn-prediction")]
        public async Task<ActionResult<ChurnPredictionResponse>> PredictChurn(
            [FromBody] ChurnPredictionRequest request)
        {
            try
            {
                _logger.LogInformation("Predicting churn for customer: {CustomerId}", request.CustomerId);
                
                var prediction = await _analyticsService.PredictChurnAsync(request);
                return Ok(prediction);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting churn for customer: {CustomerId}", request.CustomerId);
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Get personalized product recommendations
        /// </summary>
        [HttpGet("recommendations/{customerId}")]
        public async Task<ActionResult<RecommendationResponse>> GetRecommendations(
            string customerId,
            [FromQuery] int numRecommendations = 10,
            [FromQuery] bool includeExplanations = false)
        {
            try
            {
                _logger.LogInformation("Getting recommendations for customer: {CustomerId}", customerId);
                
                var recommendations = await _analyticsService.GetRecommendationsAsync(
                    customerId, numRecommendations, includeExplanations);
                
                return Ok(recommendations);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting recommendations for customer: {CustomerId}", customerId);
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Get customer lifetime value prediction
        /// </summary>
        [HttpGet("clv/{customerId}")]
        public async Task<ActionResult<CLVPredictionResponse>> GetCustomerLifetimeValue(string customerId)
        {
            try
            {
                _logger.LogInformation("Getting CLV for customer: {CustomerId}", customerId);
                
                var clv = await _analyticsService.GetCustomerLifetimeValueAsync(customerId);
                return Ok(clv);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting CLV for customer: {CustomerId}", customerId);
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Get business performance metrics
        /// </summary>
        [HttpGet("performance-metrics")]
        public async Task<ActionResult<PerformanceMetricsResponse>> GetPerformanceMetrics()
        {
            try
            {
                _logger.LogInformation("Getting performance metrics");
                
                var metrics = await _analyticsService.GetPerformanceMetricsAsync();
                return Ok(metrics);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting performance metrics");
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }
    }
}
