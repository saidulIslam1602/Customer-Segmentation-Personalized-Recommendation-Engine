using System.Text;
using System.Text.Json;

namespace CustomerSegmentation.API.Services
{
    public class PythonMLService : IPythonMLService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<PythonMLService> _logger;
        private readonly JsonSerializerOptions _jsonOptions;

        public PythonMLService(IHttpClientFactory httpClientFactory, ILogger<PythonMLService> logger)
        {
            _httpClient = httpClientFactory.CreateClient("PythonAPI");
            _logger = logger;
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                PropertyNameCaseInsensitive = true
            };
        }

        public async Task<T> CallPythonServiceAsync<T>(string endpoint, object? data = null)
        {
            try
            {
                HttpResponseMessage response;

                if (data == null)
                {
                    response = await _httpClient.GetAsync(endpoint);
                }
                else
                {
                    var json = JsonSerializer.Serialize(data, _jsonOptions);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    response = await _httpClient.PostAsync(endpoint, content);
                }

                response.EnsureSuccessStatusCode();

                var responseContent = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<T>(responseContent, _jsonOptions);

                return result!;
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "HTTP error calling Python service endpoint: {Endpoint}", endpoint);
                throw new InvalidOperationException($"Failed to call Python ML service: {ex.Message}", ex);
            }
            catch (TaskCanceledException ex)
            {
                _logger.LogError(ex, "Timeout calling Python service endpoint: {Endpoint}", endpoint);
                throw new InvalidOperationException($"Python ML service call timed out: {ex.Message}", ex);
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "JSON deserialization error for endpoint: {Endpoint}", endpoint);
                throw new InvalidOperationException($"Failed to parse response from Python ML service: {ex.Message}", ex);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected error calling Python service endpoint: {Endpoint}", endpoint);
                throw;
            }
        }

        public async Task<bool> IsServiceHealthyAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("health");
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Python ML service health check failed");
                return false;
            }
        }
    }
}
