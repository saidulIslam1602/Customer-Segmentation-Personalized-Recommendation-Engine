using CustomerSegmentation.API.Models;

namespace CustomerSegmentation.API.Services
{
    public class CRMIntegrationService : ICRMIntegrationService
    {
        private readonly IPythonMLService _pythonService;
        private readonly ILogger<CRMIntegrationService> _logger;
        private readonly List<SyncHistoryEntry> _syncHistory;

        public CRMIntegrationService(
            IPythonMLService pythonService,
            ILogger<CRMIntegrationService> logger)
        {
            _pythonService = pythonService;
            _logger = logger;
            _syncHistory = new List<SyncHistoryEntry>();
        }

        public async Task<CRMSyncResponse> SyncCustomerSegmentsAsync(CRMSyncRequest request)
        {
            try
            {
                _logger.LogInformation("Syncing customer segments to systems: {Systems}", string.Join(", ", request.Systems));

                var results = new List<SystemSyncResult>();
                var totalRecords = 0;

                foreach (var system in request.Systems)
                {
                    var result = await SyncToSystem(system, "customer_segments", request.ForceSync);
                    results.Add(result);
                    totalRecords += result.RecordsSynced;

                    // Add to sync history
                    _syncHistory.Add(new SyncHistoryEntry
                    {
                        Id = Guid.NewGuid().ToString(),
                        SystemName = system,
                        DataType = "customer_segments",
                        Status = result.Status,
                        RecordsSynced = result.RecordsSynced,
                        Timestamp = DateTime.UtcNow,
                        ErrorMessage = result.ErrorMessage
                    });
                }

                return new CRMSyncResponse
                {
                    Status = results.All(r => r.Status == "success") ? "success" : "partial_success",
                    Results = results,
                    TotalRecordsSynced = totalRecords,
                    Timestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error syncing customer segments");
                throw;
            }
        }

        public async Task<CampaignResponse> TriggerRetentionCampaignsAsync(RetentionCampaignRequest request)
        {
            try
            {
                _logger.LogInformation("Triggering retention campaigns for {Count} customers", request.CustomerIds.Count);

                var campaigns = new List<CampaignResult>();

                // Group customers by risk level for targeted campaigns
                var highRiskCustomers = request.CustomerIds.Take(request.CustomerIds.Count / 3).ToList();
                var mediumRiskCustomers = request.CustomerIds.Skip(request.CustomerIds.Count / 3).Take(request.CustomerIds.Count / 3).ToList();
                var lowRiskCustomers = request.CustomerIds.Skip(2 * request.CustomerIds.Count / 3).ToList();

                // Create campaigns for different risk levels
                if (highRiskCustomers.Any())
                {
                    campaigns.Add(new CampaignResult
                    {
                        CampaignId = Guid.NewGuid().ToString(),
                        CampaignType = "high_risk_retention",
                        Status = "created",
                        TargetCustomers = highRiskCustomers.Count,
                        CreatedAt = DateTime.UtcNow
                    });
                }

                if (mediumRiskCustomers.Any())
                {
                    campaigns.Add(new CampaignResult
                    {
                        CampaignId = Guid.NewGuid().ToString(),
                        CampaignType = "medium_risk_retention",
                        Status = "created",
                        TargetCustomers = mediumRiskCustomers.Count,
                        CreatedAt = DateTime.UtcNow
                    });
                }

                if (lowRiskCustomers.Any())
                {
                    campaigns.Add(new CampaignResult
                    {
                        CampaignId = Guid.NewGuid().ToString(),
                        CampaignType = "engagement_campaign",
                        Status = "created",
                        TargetCustomers = lowRiskCustomers.Count,
                        CreatedAt = DateTime.UtcNow
                    });
                }

                // Simulate delay for campaign creation
                await Task.Delay(500);

                return new CampaignResponse
                {
                    Status = "success",
                    Campaigns = campaigns,
                    TotalCampaignsCreated = campaigns.Count,
                    Timestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error triggering retention campaigns");
                throw;
            }
        }

        public async Task<CRMSyncResponse> UpdateLeadScoresAsync(LeadScoringRequest request)
        {
            try
            {
                _logger.LogInformation("Updating lead scores for {Count} leads", request.LeadScores.Count);

                var results = new List<SystemSyncResult>();
                var totalRecords = 0;

                var targetSystems = request.TargetSystems.Any() ? request.TargetSystems : new[] { "dynamics365", "salesforce", "hubspot" };

                foreach (var system in targetSystems)
                {
                    var result = await SyncToSystem(system, "lead_scores", false);
                    results.Add(result);
                    totalRecords += result.RecordsSynced;

                    // Add to sync history
                    _syncHistory.Add(new SyncHistoryEntry
                    {
                        Id = Guid.NewGuid().ToString(),
                        SystemName = system,
                        DataType = "lead_scores",
                        Status = result.Status,
                        RecordsSynced = result.RecordsSynced,
                        Timestamp = DateTime.UtcNow,
                        ErrorMessage = result.ErrorMessage
                    });
                }

                return new CRMSyncResponse
                {
                    Status = results.All(r => r.Status == "success") ? "success" : "partial_success",
                    Results = results,
                    TotalRecordsSynced = totalRecords,
                    Timestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating lead scores");
                throw;
            }
        }

        public async Task<CRMStatusResponse> GetCRMStatusAsync()
        {
            try
            {
                var systems = new Dictionary<string, SystemStatus>
                {
                    ["dynamics365"] = new SystemStatus
                    {
                        Status = "connected",
                        LastSync = DateTime.UtcNow.AddMinutes(-Random.Shared.Next(1, 60)),
                        RecordsSynced = Random.Shared.Next(500, 2000),
                        ErrorMessage = null
                    },
                    ["salesforce"] = new SystemStatus
                    {
                        Status = "connected",
                        LastSync = DateTime.UtcNow.AddMinutes(-Random.Shared.Next(1, 60)),
                        RecordsSynced = Random.Shared.Next(300, 1500),
                        ErrorMessage = null
                    },
                    ["hubspot"] = new SystemStatus
                    {
                        Status = "connected",
                        LastSync = DateTime.UtcNow.AddMinutes(-Random.Shared.Next(1, 60)),
                        RecordsSynced = Random.Shared.Next(200, 1000),
                        ErrorMessage = null
                    }
                };

                var totalRecords = systems.Values.Sum(s => s.RecordsSynced);
                var connectedSystems = systems.Values.Count(s => s.Status == "connected");
                var successRate = (double)connectedSystems / systems.Count * 100;

                await Task.Delay(100); // Simulate processing time

                return new CRMStatusResponse
                {
                    Systems = systems,
                    OverallStatus = connectedSystems == systems.Count ? "healthy" : "degraded",
                    TotalRecordsSynced = totalRecords,
                    SyncSuccessRate = successRate,
                    Timestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting CRM status");
                throw;
            }
        }

        public async Task<SyncHistoryResponse> GetSyncHistoryAsync(DateTime? startDate, DateTime? endDate, int limit)
        {
            try
            {
                var filteredHistory = _syncHistory.AsEnumerable();

                if (startDate.HasValue)
                    filteredHistory = filteredHistory.Where(h => h.Timestamp >= startDate.Value);

                if (endDate.HasValue)
                    filteredHistory = filteredHistory.Where(h => h.Timestamp <= endDate.Value);

                var history = filteredHistory
                    .OrderByDescending(h => h.Timestamp)
                    .Take(limit)
                    .ToList();

                await Task.Delay(50); // Simulate processing time

                return new SyncHistoryResponse
                {
                    History = history,
                    TotalEntries = history.Count,
                    Timestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting sync history");
                throw;
            }
        }

        public async Task<ConnectionTestResponse> TestConnectionAsync(string crmSystem)
        {
            try
            {
                _logger.LogInformation("Testing connection to CRM system: {System}", crmSystem);

                // Simulate connection test
                await Task.Delay(Random.Shared.Next(500, 2000));

                var isConnected = Random.Shared.NextDouble() > 0.1; // 90% success rate

                return new ConnectionTestResponse
                {
                    CRMSystem = crmSystem,
                    IsConnected = isConnected,
                    Status = isConnected ? "connected" : "failed",
                    ErrorMessage = isConnected ? null : "Connection timeout or authentication failed",
                    Timestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error testing connection to {System}", crmSystem);
                
                return new ConnectionTestResponse
                {
                    CRMSystem = crmSystem,
                    IsConnected = false,
                    Status = "error",
                    ErrorMessage = ex.Message,
                    Timestamp = DateTime.UtcNow
                };
            }
        }

        private async Task<SystemSyncResult> SyncToSystem(string systemName, string dataType, bool forceSync)
        {
            try
            {
                // Simulate sync operation
                await Task.Delay(Random.Shared.Next(200, 1000));

                var success = Random.Shared.NextDouble() > 0.05; // 95% success rate
                var recordsSynced = success ? Random.Shared.Next(50, 500) : 0;

                return new SystemSyncResult
                {
                    SystemName = systemName,
                    Status = success ? "success" : "failed",
                    RecordsSynced = recordsSynced,
                    ErrorMessage = success ? null : "Sync operation failed due to network timeout",
                    Timestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                return new SystemSyncResult
                {
                    SystemName = systemName,
                    Status = "error",
                    RecordsSynced = 0,
                    ErrorMessage = ex.Message,
                    Timestamp = DateTime.UtcNow
                };
            }
        }
    }
}
