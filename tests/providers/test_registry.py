from typing import Any, AsyncIterator, Dict

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from codegate.providers.base import BaseCompletionHandler, BaseProvider
from codegate.providers.registry import ProviderRegistry


class MockCompletionHandler(BaseCompletionHandler):
    async def complete(self, data: Dict, api_key: str) -> AsyncIterator[Any]:
        yield "test"

    def translate_request(self, data: Dict, api_key: str) -> Any:
        return data

    def translate_response(self, response: Any) -> Any:
        return response

    def translate_streaming_response(
        self,
        response: AsyncIterator[Any],
    ) -> AsyncIterator[Any]:
        return response

    def execute_completion(
        self,
        request: Any,
        stream: bool = False,
    ) -> Any:
        pass

    def create_streaming_response(
        self,
        stream: AsyncIterator[Any],
    ) -> StreamingResponse:
        return StreamingResponse(stream)


class MockProvider(BaseProvider):

    @property
    def provider_route_name(self) -> str:
        return 'mock_provider'

    def _setup_routes(self) -> None:
        @self.router.get(f"/{self.provider_route_name}/test")
        def test_route():
            return {"message": "test"}


@pytest.fixture
def mock_completion_handler():
    return MockCompletionHandler()


@pytest.fixture
def app():
    return FastAPI()


@pytest.fixture
def registry(app):
    return ProviderRegistry(app)


def test_add_provider(registry, mock_completion_handler):
    provider = MockProvider(mock_completion_handler)
    registry.add_provider("test", provider)

    assert "test" in registry.providers
    assert registry.providers["test"] == provider


def test_get_provider(registry, mock_completion_handler):
    provider = MockProvider(mock_completion_handler)
    registry.add_provider("test", provider)

    assert registry.get_provider("test") == provider
    assert registry.get_provider("nonexistent") is None


def test_provider_routes_added(app, registry, mock_completion_handler):
    provider = MockProvider(mock_completion_handler)
    registry.add_provider("test", provider)

    routes = [route for route in app.routes if route.path == "/mock_provider/test"]
    assert len(routes) == 1
