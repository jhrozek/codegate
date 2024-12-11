"""Tests for the server module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from codegate import __version__
from codegate.server import init_app
from codegate.providers.registry import ProviderRegistry
from codegate.pipeline.secrets.manager import SecretsManager


@pytest.fixture
def mock_secrets_manager():
    """Create a mock secrets manager."""
    return MagicMock(spec=SecretsManager)


@pytest.fixture
def mock_provider_registry():
    """Create a mock provider registry."""
    return MagicMock(spec=ProviderRegistry)


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI application."""
    app = init_app()
    return TestClient(app)


def test_app_initialization() -> None:
    """Test that the FastAPI application initializes correctly."""
    app = init_app()
    assert app is not None
    assert app.title == "CodeGate"
    assert app.version == __version__


def test_cors_middleware() -> None:
    """Test that CORS middleware is properly configured."""
    app = init_app()
    cors_middleware = next(
        m for m in app.middleware if isinstance(m, CORSMiddleware)
    )
    assert cors_middleware.options.allow_origins == ["*"]
    assert cors_middleware.options.allow_credentials is True
    assert cors_middleware.options.allow_methods == ["*"]
    assert cors_middleware.options.allow_headers == ["*"]


def test_health_check(test_client: TestClient) -> None:
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("codegate.server.ProviderRegistry")
@patch("codegate.server.SecretsManager")
def test_provider_registration(mock_secrets_mgr, mock_registry) -> None:
    """Test that all providers are registered correctly."""
    init_app()
    
    # Verify SecretsManager was initialized
    mock_secrets_mgr.assert_called_once()
    
    # Verify ProviderRegistry was initialized with the app
    mock_registry.assert_called_once()
    
    # Verify all providers were registered
    registry_instance = mock_registry.return_value
    assert registry_instance.add_provider.call_count == 5  # openai, anthropic, llamacpp, vllm, ollama
    
    # Verify specific providers were registered
    provider_names = [
        call.args[0] for call in registry_instance.add_provider.call_args_list
    ]
    assert "openai" in provider_names
    assert "anthropic" in provider_names
    assert "llamacpp" in provider_names
    assert "vllm" in provider_names
    assert "ollama" in provider_names


@patch("codegate.server.CodegateSignatures")
def test_signatures_initialization(mock_signatures) -> None:
    """Test that signatures are initialized correctly."""
    init_app()
    mock_signatures.initialize.assert_called_once_with("signatures.yaml")


def test_pipeline_initialization() -> None:
    """Test that pipelines are initialized correctly."""
    app = init_app()
    
    # Access the provider registry to check pipeline configuration
    registry = next(
        (route for route in app.routes if hasattr(route, "registry")),
        None
    )
    
    if registry:
        for provider in registry.registry.values():
            # Verify each provider has the required pipelines
            assert hasattr(provider, "pipeline_processor")
            assert hasattr(provider, "fim_pipeline_processor")
            assert hasattr(provider, "output_pipeline_processor")


def test_dashboard_routes() -> None:
    """Test that dashboard routes are included."""
    app = init_app()
    routes = [route.path for route in app.routes]
    
    # Verify dashboard endpoints are included
    dashboard_routes = [route for route in routes if route.startswith("/dashboard")]
    assert len(dashboard_routes) > 0


def test_system_routes() -> None:
    """Test that system routes are included."""
    app = init_app()
    routes = [route.path for route in app.routes]
    
    # Verify system endpoints are included
    assert "/health" in routes


@pytest.mark.asyncio
async def test_async_health_check() -> None:
    """Test the health check endpoint with async client."""
    app = init_app()
    
    async with TestClient(app) as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_error_handling(test_client: TestClient) -> None:
    """Test error handling for non-existent endpoints."""
    response = test_client.get("/nonexistent")
    assert response.status_code == 404
    
    # Test method not allowed
    response = test_client.post("/health")  # Health endpoint only allows GET
    assert response.status_code == 405
