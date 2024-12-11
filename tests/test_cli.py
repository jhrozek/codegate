"""Tests for the CLI module."""

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator
import signal
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from click.testing import CliRunner
from uvicorn.config import Config as UvicornConfig

from codegate.cli import cli, UvicornServer
from codegate.codegate_logging import LogFormat, LogLevel
from codegate.config import DEFAULT_PROVIDER_URLS


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_logging(monkeypatch: Any) -> MagicMock:
    """Mock the logging function."""
    mock = MagicMock()
    monkeypatch.setattr("codegate.cli.structlog.get_logger", mock)
    return mock


@pytest.fixture
def mock_setup_logging(monkeypatch: Any) -> MagicMock:
    """Mock the setup_logging function."""
    mock = MagicMock()
    monkeypatch.setattr("codegate.cli.setup_logging", mock)
    return mock


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
port: 8989
host: localhost
log_level: DEBUG
log_format: JSON
certs_dir: "./test-certs"
ca_cert: "test-ca.crt"
ca_key: "test-ca.key"
server_cert: "test-server.crt"
server_key: "test-server.key"
"""
    )
    return config_file


@pytest.fixture
async def mock_uvicorn_server() -> AsyncGenerator[MagicMock, None]:
    """Create a mock Uvicorn server."""
    config = UvicornConfig(app=MagicMock(), host="localhost", port=8989)
    server = MagicMock()
    server.serve = AsyncMock()
    server.shutdown = AsyncMock()
    yield UvicornServer(config=config, server=server)


def test_cli_version(cli_runner: CliRunner) -> None:
    """Test CLI version command."""
    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_uvicorn_server_serve(mock_uvicorn_server: UvicornServer) -> None:
    """Test UvicornServer serve method."""
    # Start server in background task
    server_task = asyncio.create_task(mock_uvicorn_server.serve())
    
    # Wait for startup to complete
    await mock_uvicorn_server.wait_startup_complete()
    
    # Verify server started
    assert mock_uvicorn_server.server.serve.called
    
    # Cleanup
    await mock_uvicorn_server.cleanup()
    await server_task


@pytest.mark.asyncio
async def test_uvicorn_server_cleanup(mock_uvicorn_server: UvicornServer) -> None:
    """Test UvicornServer cleanup method."""
    # Start server
    server_task = asyncio.create_task(mock_uvicorn_server.serve())
    await mock_uvicorn_server.wait_startup_complete()
    
    # Trigger cleanup
    await mock_uvicorn_server.cleanup()
    
    # Verify shutdown was called
    assert mock_uvicorn_server.server.shutdown.called
    assert mock_uvicorn_server._shutdown_event.is_set()
    
    await server_task


@pytest.mark.asyncio
async def test_uvicorn_server_signal_handling(mock_uvicorn_server: UvicornServer) -> None:
    """Test signal handling in UvicornServer."""
    # Mock signal handlers
    with patch('asyncio.get_running_loop') as mock_loop:
        mock_loop_instance = MagicMock()
        mock_loop.return_value = mock_loop_instance
        
        # Start server
        server_task = asyncio.create_task(mock_uvicorn_server.serve())
        await mock_uvicorn_server.wait_startup_complete()
        
        # Simulate SIGTERM
        mock_loop_instance.add_signal_handler.assert_any_call(
            signal.SIGTERM,
            pytest.approx(type(lambda: None))  # Check if a callable was passed
        )
        
        # Simulate SIGINT
        mock_loop_instance.add_signal_handler.assert_any_call(
            signal.SIGINT,
            pytest.approx(type(lambda: None))  # Check if a callable was passed
        )
        
        await mock_uvicorn_server.cleanup()
        await server_task


def test_serve_default_options(
    cli_runner: CliRunner, mock_logging: Any, mock_setup_logging: Any
) -> None:
    """Test serve command with default options."""
    with patch("codegate.cli.run_servers") as mock_run:
        logger_instance = MagicMock()
        mock_logging.return_value = logger_instance
        result = cli_runner.invoke(cli, ["serve"])

        assert result.exit_code == 0
        mock_setup_logging.assert_called_once_with(LogLevel.INFO, LogFormat.JSON)
        mock_logging.assert_called_with("codegate")

        # validate only a subset of the expected extra arguments
        expected_extra = {
            "host": "localhost",
            "port": 8989,
            "log_level": "INFO",
            "log_format": "JSON",
            "prompts_loaded": 7,
            "provider_urls": DEFAULT_PROVIDER_URLS,
            "certs_dir": "./certs",  # Default certificate directory
        }

        # Retrieve the actual call arguments
        calls = [call[1]["extra"] for call in logger_instance.info.call_args_list]

        # Check if one of the calls matches the expected subset
        assert any(
            all(expected_extra[k] == actual_extra.get(k) for k in expected_extra)
            for actual_extra in calls
        )
        mock_run.assert_called_once()


def test_serve_custom_options(
    cli_runner: CliRunner, mock_logging: Any, mock_setup_logging: Any
) -> None:
    """Test serve command with custom options."""
    with patch("codegate.cli.run_servers") as mock_run:
        logger_instance = MagicMock()
        mock_logging.return_value = logger_instance
        result = cli_runner.invoke(
            cli,
            [
                "serve",
                "--port",
                "8989",
                "--host",
                "localhost",
                "--log-level",
                "DEBUG",
                "--log-format",
                "TEXT",
                "--certs-dir",
                "./custom-certs",
                "--ca-cert",
                "custom-ca.crt",
                "--ca-key",
                "custom-ca.key",
                "--server-cert",
                "custom-server.crt",
                "--server-key",
                "custom-server.key",
            ],
        )

        assert result.exit_code == 0
        mock_setup_logging.assert_called_once_with(LogLevel.DEBUG, LogFormat.TEXT)
        mock_logging.assert_called_with("codegate")

        # Retrieve the actual call arguments
        calls = [call[1]["extra"] for call in logger_instance.info.call_args_list]

        expected_extra = {
            "host": "localhost",
            "port": 8989,
            "log_level": "DEBUG",
            "log_format": "TEXT",
            "prompts_loaded": 7,  # Default prompts are loaded
            "provider_urls": DEFAULT_PROVIDER_URLS,
            "certs_dir": "./custom-certs",
        }

        # Check if one of the calls matches the expected subset
        assert any(
            all(expected_extra[k] == actual_extra.get(k) for k in expected_extra)
            for actual_extra in calls
        )
        mock_run.assert_called_once()


def test_serve_invalid_port(cli_runner: CliRunner) -> None:
    """Test serve command with invalid port."""
    result = cli_runner.invoke(cli, ["serve", "--port", "999999"])
    assert result.exit_code == 2
    assert "Port must be between 1 and 65535" in result.output


def test_serve_invalid_log_level(cli_runner: CliRunner) -> None:
    """Test serve command with invalid log level."""
    result = cli_runner.invoke(cli, ["serve", "--log-level", "INVALID"])
    assert result.exit_code == 2
    assert "Invalid value for '--log-level'" in result.output


def test_serve_with_config_file(
    cli_runner: CliRunner, mock_logging: Any, temp_config_file: Path, mock_setup_logging: Any
) -> None:
    """Test serve command with config file."""
    with patch("codegate.cli.run_servers") as mock_run:
        logger_instance = MagicMock()
        mock_logging.return_value = logger_instance
        result = cli_runner.invoke(cli, ["serve", "--config", str(temp_config_file)])

        assert result.exit_code == 0
        mock_setup_logging.assert_called_once_with(LogLevel.DEBUG, LogFormat.JSON)
        mock_logging.assert_called_with("codegate")

        # Retrieve the actual call arguments
        calls = [call[1]["extra"] for call in logger_instance.info.call_args_list]

        expected_extra = {
            "host": "localhost",
            "port": 8989,
            "log_level": "DEBUG",
            "log_format": "JSON",
            "prompts_loaded": 7,  # Default prompts are loaded
            "provider_urls": DEFAULT_PROVIDER_URLS,
            "certs_dir": "./test-certs",  # From config file
        }

        # Check if one of the calls matches the expected subset
        assert any(
            all(expected_extra[k] == actual_extra.get(k) for k in expected_extra)
            for actual_extra in calls
        )
        mock_run.assert_called_once()


def test_serve_with_nonexistent_config_file(cli_runner: CliRunner) -> None:
    """Test serve command with nonexistent config file."""
    result = cli_runner.invoke(cli, ["serve", "--config", "nonexistent.yaml"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_serve_priority_resolution(
    cli_runner: CliRunner,
    mock_logging: Any,
    temp_config_file: Path,
    env_vars: Any,
    mock_setup_logging: Any,
) -> None:
    """Test serve command respects configuration priority."""
    with patch("codegate.cli.run_servers") as mock_run:
        logger_instance = MagicMock()
        mock_logging.return_value = logger_instance
        result = cli_runner.invoke(
            cli,
            [
                "serve",
                "--config",
                str(temp_config_file),
                "--port",
                "8080",
                "--host",
                "example.com",
                "--log-level",
                "ERROR",
                "--log-format",
                "TEXT",
                "--certs-dir",
                "./cli-certs",
                "--ca-cert",
                "cli-ca.crt",
                "--ca-key",
                "cli-ca.key",
                "--server-cert",
                "cli-server.crt",
                "--server-key",
                "cli-server.key",
            ],
        )

        assert result.exit_code == 0
        mock_setup_logging.assert_called_once_with(LogLevel.ERROR, LogFormat.TEXT)
        mock_logging.assert_called_with("codegate")

        # Retrieve the actual call arguments
        calls = [call[1]["extra"] for call in logger_instance.info.call_args_list]

        expected_extra = {
            "host": "example.com",
            "port": 8080,
            "log_level": "ERROR",
            "log_format": "TEXT",
            "prompts_loaded": 7,  # Default prompts are loaded
            "provider_urls": DEFAULT_PROVIDER_URLS,
            "certs_dir": "./cli-certs",  # CLI args override config file
        }

        # Check if one of the calls matches the expected subset
        assert any(
            all(expected_extra[k] == actual_extra.get(k) for k in expected_extra)
            for actual_extra in calls
        )
        mock_run.assert_called_once()


def test_serve_certificate_options(
    cli_runner: CliRunner, mock_logging: Any, mock_setup_logging: Any
) -> None:
    """Test serve command with certificate options."""
    with patch("codegate.cli.run_servers") as mock_run:
        logger_instance = MagicMock()
        mock_logging.return_value = logger_instance
        result = cli_runner.invoke(
            cli,
            [
                "serve",
                "--certs-dir",
                "./custom-certs",
                "--ca-cert",
                "custom-ca.crt",
                "--ca-key",
                "custom-ca.key",
                "--server-cert",
                "custom-server.crt",
                "--server-key",
                "custom-server.key",
            ],
        )

        assert result.exit_code == 0
        mock_setup_logging.assert_called_once_with(LogLevel.INFO, LogFormat.JSON)
        mock_logging.assert_called_with("codegate")

        # Retrieve the actual call arguments
        calls = [call[1]["extra"] for call in logger_instance.info.call_args_list]

        expected_extra = {
            "host": "localhost",
            "port": 8989,
            "log_level": "INFO",
            "log_format": "JSON",
            "prompts_loaded": 6,
            "provider_urls": DEFAULT_PROVIDER_URLS,
            "certs_dir": "./custom-certs",
        }

        # Check if one of the calls matches the expected subset
        assert any(
            all(expected_extra[k] == actual_extra.get(k) for k in expected_extra)
            for actual_extra in calls
        )
        mock_run.assert_called_once()


def test_main_function() -> None:
    """Test main function."""
    with patch("codegate.cli.cli") as mock_cli:
        from codegate.cli import main

        main()
        mock_cli.assert_called_once()
