using CustomerSegmentation.API.Models;

namespace CustomerSegmentation.API.Services
{
    public interface ICustomerAnalyticsService
    {
        Task<List<CustomerSegmentResponse>> GetCustomerSegmentsAsync(string? customerId = null, bool includePredictions = true);
        Task<ChurnPredictionResponse> PredictChurnAsync(ChurnPredictionRequest request);
        Task<RecommendationResponse> GetRecommendationsAsync(string customerId, int numRecommendations = 10, bool includeExplanations = false);
        Task<CLVPredictionResponse> GetCustomerLifetimeValueAsync(string customerId);
        Task<PerformanceMetricsResponse> GetPerformanceMetricsAsync();
    }

    public interface ICRMIntegrationService
    {
        Task<CRMSyncResponse> SyncCustomerSegmentsAsync(CRMSyncRequest request);
        Task<CampaignResponse> TriggerRetentionCampaignsAsync(RetentionCampaignRequest request);
        Task<CRMSyncResponse> UpdateLeadScoresAsync(LeadScoringRequest request);
        Task<CRMStatusResponse> GetCRMStatusAsync();
        Task<SyncHistoryResponse> GetSyncHistoryAsync(DateTime? startDate, DateTime? endDate, int limit);
        Task<ConnectionTestResponse> TestConnectionAsync(string crmSystem);
    }

    public interface IPythonMLService
    {
        Task<T> CallPythonServiceAsync<T>(string endpoint, object? data = null);
        Task<bool> IsServiceHealthyAsync();
    }
}
