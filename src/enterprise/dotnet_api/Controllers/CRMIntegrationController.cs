using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using CustomerSegmentation.API.Services;
using CustomerSegmentation.API.Models;

namespace CustomerSegmentation.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class CRMIntegrationController : ControllerBase
    {
        private readonly ICRMIntegrationService _crmService;
        private readonly ILogger<CRMIntegrationController> _logger;

        public CRMIntegrationController(
            ICRMIntegrationService crmService,
            ILogger<CRMIntegrationController> logger)
        {
            _crmService = crmService;
            _logger = logger;
        }

        /// <summary>
        /// Sync customer segments to CRM systems
        /// </summary>
        [HttpPost("sync-segments")]
        public async Task<ActionResult<CRMSyncResponse>> SyncCustomerSegments(
            [FromBody] CRMSyncRequest request)
        {
            try
            {
                _logger.LogInformation("Syncing customer segments to CRM systems: {Systems}", 
                    string.Join(", ", request.Systems));
                
                var result = await _crmService.SyncCustomerSegmentsAsync(request);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error syncing customer segments");
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Trigger retention campaigns for high-risk customers
        /// </summary>
        [HttpPost("trigger-retention-campaigns")]
        public async Task<ActionResult<CampaignResponse>> TriggerRetentionCampaigns(
            [FromBody] RetentionCampaignRequest request)
        {
            try
            {
                _logger.LogInformation("Triggering retention campaigns for {Count} customers", 
                    request.CustomerIds.Count);
                
                var result = await _crmService.TriggerRetentionCampaignsAsync(request);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error triggering retention campaigns");
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Update lead scores in CRM systems
        /// </summary>
        [HttpPost("update-lead-scores")]
        public async Task<ActionResult<CRMSyncResponse>> UpdateLeadScores(
            [FromBody] LeadScoringRequest request)
        {
            try
            {
                _logger.LogInformation("Updating lead scores for {Count} leads", 
                    request.LeadScores.Count);
                
                var result = await _crmService.UpdateLeadScoresAsync(request);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating lead scores");
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Get CRM integration status
        /// </summary>
        [HttpGet("status")]
        public async Task<ActionResult<CRMStatusResponse>> GetCRMStatus()
        {
            try
            {
                _logger.LogInformation("Getting CRM integration status");
                
                var status = await _crmService.GetCRMStatusAsync();
                return Ok(status);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting CRM status");
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Get sync history
        /// </summary>
        [HttpGet("sync-history")]
        public async Task<ActionResult<SyncHistoryResponse>> GetSyncHistory(
            [FromQuery] DateTime? startDate = null,
            [FromQuery] DateTime? endDate = null,
            [FromQuery] int limit = 100)
        {
            try
            {
                _logger.LogInformation("Getting sync history from {StartDate} to {EndDate}", 
                    startDate, endDate);
                
                var history = await _crmService.GetSyncHistoryAsync(startDate, endDate, limit);
                return Ok(history);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting sync history");
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }

        /// <summary>
        /// Test CRM connection
        /// </summary>
        [HttpPost("test-connection")]
        public async Task<ActionResult<ConnectionTestResponse>> TestConnection(
            [FromBody] ConnectionTestRequest request)
        {
            try
            {
                _logger.LogInformation("Testing connection to CRM system: {System}", request.CRMSystem);
                
                var result = await _crmService.TestConnectionAsync(request.CRMSystem);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error testing CRM connection: {System}", request.CRMSystem);
                return StatusCode(500, new { error = "Internal server error", message = ex.Message });
            }
        }
    }
}
