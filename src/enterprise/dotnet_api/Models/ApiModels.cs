using System.ComponentModel.DataAnnotations;

namespace CustomerSegmentation.API.Models
{
    // Request Models
    public class ChurnPredictionRequest
    {
        [Required]
        public string CustomerId { get; set; } = string.Empty;
        
        public Dictionary<string, object> Features { get; set; } = new();
    }

    public class CRMSyncRequest
    {
        [Required]
        public List<string> Systems { get; set; } = new();
        
        [Required]
        public List<string> DataTypes { get; set; } = new();
        
        public bool ForceSync { get; set; } = false;
    }

    public class RetentionCampaignRequest
    {
        [Required]
        public List<string> CustomerIds { get; set; } = new();
        
        public string CampaignType { get; set; } = "retention";
        
        public Dictionary<string, object> CampaignParameters { get; set; } = new();
    }

    public class LeadScoringRequest
    {
        [Required]
        public List<LeadScore> LeadScores { get; set; } = new();
        
        public List<string> TargetSystems { get; set; } = new();
    }

    public class ConnectionTestRequest
    {
        [Required]
        public string CRMSystem { get; set; } = string.Empty;
    }

    // Response Models
    public class CustomerSegmentResponse
    {
        public string CustomerId { get; set; } = string.Empty;
        public string Segment { get; set; } = string.Empty;
        public double CLVScore { get; set; }
        public string RiskLevel { get; set; } = string.Empty;
        public List<string> Recommendations { get; set; } = new();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class ChurnPredictionResponse
    {
        public string CustomerId { get; set; } = string.Empty;
        public double ChurnProbability { get; set; }
        public string RiskCategory { get; set; } = string.Empty;
        public string RetentionStrategy { get; set; } = string.Empty;
        public double ConfidenceScore { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class RecommendationResponse
    {
        public string CustomerId { get; set; } = string.Empty;
        public List<ProductRecommendation> Recommendations { get; set; } = new();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class CLVPredictionResponse
    {
        public string CustomerId { get; set; } = string.Empty;
        public double PredictedCLV { get; set; }
        public string CLVCategory { get; set; } = string.Empty;
        public double ConfidenceScore { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class PerformanceMetricsResponse
    {
        public double OverallScore { get; set; }
        public ModelPerformanceMetrics ModelPerformance { get; set; } = new();
        public BusinessMetrics BusinessMetrics { get; set; } = new();
        public SystemMetrics SystemMetrics { get; set; } = new();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class CRMSyncResponse
    {
        public string Status { get; set; } = string.Empty;
        public List<SystemSyncResult> Results { get; set; } = new();
        public int TotalRecordsSynced { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class CampaignResponse
    {
        public string Status { get; set; } = string.Empty;
        public List<CampaignResult> Campaigns { get; set; } = new();
        public int TotalCampaignsCreated { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class CRMStatusResponse
    {
        public Dictionary<string, SystemStatus> Systems { get; set; } = new();
        public string OverallStatus { get; set; } = string.Empty;
        public int TotalRecordsSynced { get; set; }
        public double SyncSuccessRate { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class SyncHistoryResponse
    {
        public List<SyncHistoryEntry> History { get; set; } = new();
        public int TotalEntries { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class ConnectionTestResponse
    {
        public string CRMSystem { get; set; } = string.Empty;
        public bool IsConnected { get; set; }
        public string Status { get; set; } = string.Empty;
        public string? ErrorMessage { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    // Supporting Models
    public class ProductRecommendation
    {
        public string ProductId { get; set; } = string.Empty;
        public string ProductName { get; set; } = string.Empty;
        public double Score { get; set; }
        public string Category { get; set; } = string.Empty;
        public decimal Price { get; set; }
        public string? Explanation { get; set; }
    }

    public class LeadScore
    {
        public string LeadId { get; set; } = string.Empty;
        public double Score { get; set; }
        public string Category { get; set; } = string.Empty;
        public Dictionary<string, object> Attributes { get; set; } = new();
    }

    public class ModelPerformanceMetrics
    {
        public double ChurnPredictionAccuracy { get; set; }
        public double RecommendationPrecision { get; set; }
        public double SegmentationQuality { get; set; }
    }

    public class BusinessMetrics
    {
        public int TotalCustomers { get; set; }
        public int VIPCustomers { get; set; }
        public int HighRiskCustomers { get; set; }
        public decimal RevenueImpact { get; set; }
    }

    public class SystemMetrics
    {
        public double APIResponseTimeMs { get; set; }
        public double CRMSyncSuccessRate { get; set; }
        public double UptimePercentage { get; set; }
    }

    public class SystemSyncResult
    {
        public string SystemName { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public int RecordsSynced { get; set; }
        public string? ErrorMessage { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class CampaignResult
    {
        public string CampaignId { get; set; } = string.Empty;
        public string CampaignType { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public int TargetCustomers { get; set; }
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }

    public class SystemStatus
    {
        public string Status { get; set; } = string.Empty;
        public DateTime LastSync { get; set; }
        public int RecordsSynced { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class SyncHistoryEntry
    {
        public string Id { get; set; } = string.Empty;
        public string SystemName { get; set; } = string.Empty;
        public string DataType { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public int RecordsSynced { get; set; }
        public DateTime Timestamp { get; set; }
        public string? ErrorMessage { get; set; }
    }
}
