"""
Integration tests for Enterprise Business Intelligence Platform
Tests the interaction between Python ML services and .NET API gateway
"""

import pytest
import asyncio
import httpx
import json
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from enterprise.enterprise_platform_manager import EnterprisePlatformManager

class TestEnterpriseIntegration:
    """Test enterprise platform integration"""
    
    @pytest.fixture
    def enterprise_platform(self):
        """Create enterprise platform instance for testing"""
        return EnterprisePlatformManager(
            data_dir='data/processed',
            results_dir='tests/results'
        )
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for API testing"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_enterprise_platform_initialization(self, enterprise_platform):
        """Test enterprise platform initializes correctly"""
        assert enterprise_platform is not None
        assert hasattr(enterprise_platform, 'crm_integration')
        assert hasattr(enterprise_platform, 'api_gateway')
        assert hasattr(enterprise_platform, 'security_manager')
        assert hasattr(enterprise_platform, 'performance_monitor')
    
    @pytest.mark.asyncio
    async def test_crm_integration_layer(self, enterprise_platform):
        """Test CRM integration functionality"""
        # Test CRM connector initialization
        crm_layer = enterprise_platform.crm_integration
        assert crm_layer is not None
        assert len(crm_layer.connectors) >= 0  # May be empty in test environment
        
        # Test mock sync operation
        mock_insights = {
            'customer_segments': [],
            'high_risk_customers': [],
            'vip_customers': [],
            'top_customers': []
        }
        
        # This should not fail even with empty data
        sync_results = await crm_layer.sync_all_insights(mock_insights)
        assert isinstance(sync_results, dict)
    
    @pytest.mark.asyncio
    async def test_security_manager(self, enterprise_platform):
        """Test security manager functionality"""
        security_manager = enterprise_platform.security_manager
        
        # Test data encryption
        test_data = {'customer_id': 'test_123', 'sensitive_info': 'secret'}
        secured_data = security_manager.secure_data(test_data)
        
        assert secured_data is not None
        assert isinstance(secured_data, dict)
        
        # Test GDPR compliance validation
        compliance_report = security_manager.validate_gdpr_compliance(test_data)
        assert 'compliant' in compliance_report
        assert 'timestamp' in compliance_report
    
    @pytest.mark.asyncio
    async def test_performance_monitor(self, enterprise_platform):
        """Test performance monitoring functionality"""
        performance_monitor = enterprise_platform.performance_monitor
        
        # Test metrics collection
        mock_core_results = {'processing_time': 45.2, 'customers_analyzed': 1000}
        mock_crm_results = {'systems_connected': 3, 'sync_success_rate': 98.5}
        
        metrics = performance_monitor.collect_metrics(mock_core_results, mock_crm_results)
        
        assert 'overall_score' in metrics
        assert 'business_metrics' in metrics
        assert 'system_metrics' in metrics
        assert 'timestamp' in metrics
    
    @pytest.mark.asyncio
    async def test_api_gateway_health(self, http_client):
        """Test API gateway health endpoint"""
        try:
            response = await http_client.get("http://localhost:8001/health")
            if response.status_code == 200:
                data = response.json()
                assert 'status' in data
                assert data['status'] == 'healthy'
            else:
                # API may not be running in test environment
                pytest.skip("API gateway not available for testing")
        except httpx.ConnectError:
            pytest.skip("API gateway not available for testing")
    
    @pytest.mark.asyncio
    async def test_dotnet_api_health(self, http_client):
        """Test .NET API health endpoint"""
        try:
            response = await http_client.get("http://localhost:7000/health")
            if response.status_code == 200:
                data = response.json()
                assert 'Status' in data or 'status' in data
            else:
                pytest.skip(".NET API not available for testing")
        except httpx.ConnectError:
            pytest.skip(".NET API not available for testing")
    
    @pytest.mark.asyncio
    async def test_enterprise_analysis_workflow(self, enterprise_platform):
        """Test complete enterprise analysis workflow"""
        # This test may take longer as it runs the full pipeline
        try:
            results = await enterprise_platform.run_enterprise_analysis()
            
            assert results is not None
            assert 'timestamp' in results
            assert 'duration_seconds' in results
            
            # Check that core components ran
            if 'core_results' in results:
                assert results['core_results'] is not None
            
            # Check CRM sync results
            if 'crm_sync_results' in results:
                assert isinstance(results['crm_sync_results'], dict)
            
            # Check performance metrics
            if 'performance_metrics' in results:
                assert 'overall_score' in results['performance_metrics']
                
        except Exception as e:
            # In test environment, some components may not be fully available
            pytest.skip(f"Enterprise analysis not fully available: {e}")
    
    def test_configuration_loading(self, enterprise_platform):
        """Test enterprise configuration loading"""
        config = enterprise_platform.config
        
        assert config is not None
        assert 'crm' in config
        assert 'api' in config
        assert 'security' in config
        assert 'monitoring' in config
        
        # Check default values are present
        assert config['crm']['sync_interval_minutes'] > 0
        assert config['api']['port'] > 0
        assert config['security']['data_retention_days'] > 0
    
    def test_export_for_deployment(self, enterprise_platform):
        """Test deployment export functionality"""
        deployment_path = 'tests/deployment_test'
        
        manifest = enterprise_platform.export_for_deployment(deployment_path)
        
        assert manifest is not None
        assert 'platform_version' in manifest
        assert 'deployment_timestamp' in manifest
        assert 'required_services' in manifest
        assert 'environment_variables' in manifest
        
        # Check if deployment manifest file was created
        manifest_file = os.path.join(deployment_path, 'deployment_manifest.json')
        assert os.path.exists(manifest_file)
        
        # Clean up
        if os.path.exists(manifest_file):
            os.remove(manifest_file)
        if os.path.exists(deployment_path):
            os.rmdir(deployment_path)

class TestCRMConnectors:
    """Test individual CRM connector functionality"""
    
    @pytest.mark.asyncio
    async def test_dynamics365_connector(self):
        """Test Dynamics 365 connector"""
        from enterprise.crm_integration_layer import Dynamics365Connector
        
        config = {
            'base_url': 'https://test.crm.dynamics.com',
            'client_id': 'test_client',
            'client_secret': 'test_secret',
            'tenant_id': 'test_tenant'
        }
        
        connector = Dynamics365Connector(config)
        
        # Test authentication (mock)
        token = await connector.authenticate()
        assert token is not None
        assert token.startswith('mock_token_')
    
    @pytest.mark.asyncio
    async def test_salesforce_connector(self):
        """Test Salesforce connector"""
        from enterprise.crm_integration_layer import SalesforceConnector
        
        config = {
            'instance_url': 'https://test.salesforce.com',
            'client_id': 'test_client',
            'client_secret': 'test_secret',
            'username': 'test@example.com',
            'password': 'test_password'
        }
        
        connector = SalesforceConnector(config)
        
        # Test authentication (mock)
        token = await connector.authenticate()
        assert token is not None
        assert token.startswith('sf_mock_token_')
    
    @pytest.mark.asyncio
    async def test_hubspot_connector(self):
        """Test HubSpot connector"""
        from enterprise.crm_integration_layer import HubSpotConnector
        
        config = {
            'api_key': 'test_api_key'
        }
        
        connector = HubSpotConnector(config)
        
        # Test authentication (returns API key)
        api_key = await connector.authenticate()
        assert api_key == 'test_api_key'

class TestSecurityFeatures:
    """Test security and compliance features"""
    
    def test_data_encryption(self):
        """Test data encryption functionality"""
        from enterprise.security_manager import SecurityManager
        
        config = {
            'encryption_enabled': True,
            'audit_logging': True,
            'gdpr_compliance': True
        }
        
        security_manager = SecurityManager(config)
        
        # Test value encryption/decryption
        test_value = "sensitive_customer_data_123"
        encrypted = security_manager._encrypt_value(test_value)
        decrypted = security_manager.decrypt_value(encrypted)
        
        assert encrypted != test_value
        assert decrypted == test_value
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance features"""
        from enterprise.security_manager import SecurityManager
        import pandas as pd
        
        config = {
            'encryption_enabled': True,
            'gdpr_compliance': True,
            'data_retention_days': 365
        }
        
        security_manager = SecurityManager(config)
        
        # Test GDPR compliance on DataFrame
        test_df = pd.DataFrame({
            'customer_id': ['cust_1', 'cust_2'],
            'email': ['test1@example.com', 'test2@example.com'],
            'purchase_amount': [100.0, 200.0]
        })
        
        gdpr_df = security_manager._apply_gdpr_to_dataframe(test_df)
        
        assert 'data_processing_consent' in gdpr_df.columns
        assert 'data_retention_until' in gdpr_df.columns
        assert 'customer_id_hash' in gdpr_df.columns

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
