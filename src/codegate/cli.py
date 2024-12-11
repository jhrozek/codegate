"""Command-line interface for codegate."""

import sys
import asyncio
import signal
from pathlib import Path
from typing import Dict, Optional
from uvicorn.config import Config as UvicornConfig
from uvicorn.server import Server

import click
import structlog

from codegate.codegate_logging import LogFormat, LogLevel, setup_logging
from codegate.config import Config, ConfigurationError
from codegate.db.connection import init_db_sync
from codegate.server import init_app
from codegate.storage.utils import restore_storage_backup
from codegate.providers.copilot.provider import CopilotProvider
from codegate.ca.codegate_ca import CertificateAuthority

logger = structlog.get_logger("codegate")

class UvicornServer:
    def __init__(self, config: UvicornConfig, server: Server):
        self.server = server
        self.config = config
        self.port = config.port
        self.host = config.host
        self.log_level = config.log_level
        self.log_config = None
        self._startup_complete = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._should_exit = False



    async def serve(self) -> None:
        """Start the uvicorn server and handle shutdown gracefully."""
        logger.debug(f"Starting server on {self.host}:{self.port}")

        self.server = Server(config=self.config)
        self.server.force_exit = True

        try:
            self._startup_complete.set()
            await self.server.serve()
        except asyncio.CancelledError:
            logger.info("Server received cancellation")
        except Exception as e:
            logger.exception("Unexpected error occurred during server execution", exc_info=e)
        finally:
            await self.cleanup()

    async def wait_startup_complete(self) -> None:
        """Wait for the server to complete startup."""
        logger.debug("Waiting for server startup to complete")
        await self._startup_complete.wait()

    async def cleanup(self) -> None:
        """Cleanup server resources and ensure graceful shutdown."""
        logger.debug("Cleaning up server resources")
        if not self._should_exit:
            self._should_exit = True
            logger.debug("Initiating server shutdown")
            self._shutdown_event.set()

            if hasattr(self.server, 'shutdown'):
                logger.debug("Shutting down server")
                await self.server.shutdown()

            # Ensure all connections are closed
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            [task.cancel() for task in tasks]

            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug("Server shutdown complete")

def validate_port(ctx: click.Context, param: click.Parameter, value: int) -> int:
    logger.debug(f"Validating port number: {value}")
    """Validate the port number is in valid range."""
    if value is not None and not (1 <= value <= 65535):
        raise click.BadParameter("Port must be between 1 and 65535")
    return value


@click.group()
@click.version_option()
def cli() -> None:
    """Codegate - A configurable service gateway."""
    pass


@cli.command()
@click.option(
    "--prompts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Path to YAML prompts file (optional, shows default prompts if not provided)",
)
def show_prompts(prompts: Optional[Path]) -> None:
    """Display prompts from the specified file or default if no file specified."""
    try:
        cfg = Config.load(prompts_path=prompts)
        click.echo("Loaded prompts:")
        click.echo("-" * 40)
        for name, content in cfg.prompts.prompts.items():
            click.echo(f"\n{name}:")
            click.echo(f"{content}")
            click.echo("-" * 40)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--port",
    type=int,
    default=None,
    callback=validate_port,
    help="Port to listen on (default: 8989)",
)
@click.option(
    "--proxy-port",
    type=int,
    default=None,
    callback=validate_port,
    help="Proxy port to listen on (default: 8990)",
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="Host to bind to (default: localhost)",
)
@click.option(
    "--log-level",
    type=click.Choice([level.value for level in LogLevel]),
    default=None,
    help="Set the log level (default: INFO)",
)
@click.option(
    "--log-format",
    type=click.Choice([fmt.value for fmt in LogFormat], case_sensitive=False),
    default=None,
    help="Set the log format (default: JSON)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML config file",
)
@click.option(
    "--prompts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML prompts file",
)
@click.option(
    "--vllm-url",
    type=str,
    default=None,
    help="vLLM provider URL (default: http://localhost:8000/v1)",
)
@click.option(
    "--openai-url",
    type=str,
    default=None,
    help="OpenAI provider URL (default: https://api.openai.com/v1)",
)
@click.option(
    "--anthropic-url",
    type=str,
    default=None,
    help="Anthropic provider URL (default: https://api.anthropic.com/v1)",
)
@click.option(
    "--ollama-url",
    type=str,
    default=None,
    help="Ollama provider URL (default: http://localhost:11434/api)",
)
@click.option(
    "--model-base-path",
    type=str,
    default="./models",
    help="Path to the model base directory",
)
@click.option(
    "--embedding-model",
    type=str,
    default="all-minilm-L6-v2-q5_k_m.gguf",
    help="Name of the model to use for embeddings",
)
@click.option(
    "--certs-dir",
    type=str,
    default=None,
    help="Directory for certificate files (default: ./certs)",
)
@click.option(
    "--ca-cert",
    type=str,
    default=None,
    help="CA certificate file name (default: ca.crt)",
)
@click.option(
    "--ca-key",
    type=str,
    default=None,
    help="CA key file name (default: ca.key)",
)
@click.option(
    "--server-cert",
    type=str,
    default=None,
    help="Server certificate file name (default: server.crt)",
)
@click.option(
    "--server-key",
    type=str,
    default=None,
    help="Server key file name (default: server.key)",
)
def serve(
    port: Optional[int],
    proxy_port: Optional[int],
    host: Optional[str],
    log_level: Optional[str],
    log_format: Optional[str],
    config: Optional[Path],
    prompts: Optional[Path],
    vllm_url: Optional[str],
    openai_url: Optional[str],
    anthropic_url: Optional[str],
    ollama_url: Optional[str],
    model_base_path: Optional[str],
    embedding_model: Optional[str],
    certs_dir: Optional[str],
    ca_cert: Optional[str],
    ca_key: Optional[str],
    server_cert: Optional[str],
    server_key: Optional[str],
) -> None:
    """Start the codegate server."""
    try:
        # Create provider URLs dict from CLI options
        cli_provider_urls: Dict[str, str] = {}
        if vllm_url:
            cli_provider_urls["vllm"] = vllm_url
        if openai_url:
            cli_provider_urls["openai"] = openai_url
        if anthropic_url:
            cli_provider_urls["anthropic"] = anthropic_url
        if ollama_url:
            cli_provider_urls["ollama"] = ollama_url

        # Load configuration with priority resolution
        cfg = Config.load(
            config_path=config,
            prompts_path=prompts,
            cli_port=port,
            cli_proxy_port=proxy_port,
            cli_host=host,
            cli_log_level=log_level,
            cli_log_format=log_format,
            cli_provider_urls=cli_provider_urls,
            model_base_path=model_base_path,
            embedding_model=embedding_model,
            certs_dir=certs_dir,
            ca_cert=ca_cert,
            ca_key=ca_key,
            server_cert=server_cert,
            server_key=server_key,
        )

        init_db_sync()


        # Check certificates and create CA if necessary
        logger.info("Checking certificates and creating CA our created")
        ca = CertificateAuthority.get_instance()
        ca.ensure_certificates_exist()

        # Set up event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = init_app()

        # Run the server
        try:
            loop.run_until_complete(run_servers(cfg, app))
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            loop.close()

    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.exception("Unexpected error occurred")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


async def run_servers(cfg: Config, app) -> None:
    """Run the codegate server."""
    try:
        setup_logging(cfg.log_level, cfg.log_format)
        logger = structlog.get_logger("codegate")
        logger.info(
            "Starting server",
            extra={
                "host": cfg.host,
                "port": cfg.port,
                "proxy_port": cfg.proxy_port,
                "log_level": cfg.log_level.value,
                "log_format": cfg.log_format.value,
                "prompts_loaded": len(cfg.prompts.prompts),
                "provider_urls": cfg.provider_urls,
                "model_base_path": cfg.model_base_path,
                "embedding_model": cfg.embedding_model,
                "certs_dir": cfg.certs_dir,
            },
        )

        # init_db_sync()
        # app = init_app()

        # Create Uvicorn configuration
        uvicorn_config = UvicornConfig(
            app,
            host=cfg.host,
            port=cfg.port,
            log_level=cfg.log_level.value.lower(),
            log_config=None,  # Default logging configuration
        )

        server = UvicornServer(uvicorn_config, Server(config=uvicorn_config))

        # Initialize CopilotProvider and call run_proxy_server
        copilot_provider = CopilotProvider(cfg)
        # copilot_provider.run_proxy_server(cfg.proxy_port)


        tasks = [
            asyncio.create_task(server.serve()), # Uvicorn server
            asyncio.create_task(copilot_provider.run_proxy_server()) # Proxy server
        ]

        # Create and start Uvicorn server
        # server = UvicornServer(uvicorn_config, Server(config=uvicorn_config))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Server received cancellation")
        except Exception as e:
            logger.exception("Unexpected error occurred during server execution", exc_info=e)
        finally:
            await server.cleanup()
            # Cleanup
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                pass


        # Set up signal handlers
        # loop = asyncio.get_running_loop()
        # for sig in (signal.SIGTERM, signal.SIGINT):
        #     loop.add_signal_handler(sig, lambda: asyncio.create_task(server.cleanup()))

        # # Start server
        # await server.serve()

    except Exception as e:
        logger.exception("Error running servers")
        raise e

@cli.command()
@click.option(
    "--backup-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory path where the backup file is located.",
)
@click.option(
    "--backup-name",
    type=str,
    required=True,
    help="Name of the backup file to restore.",
)
def restore_backup(backup_path: Path, backup_name: str) -> None:
    """Restore the database from the specified backup."""
    try:
        restore_storage_backup(backup_path, backup_name)
        click.echo(f"Successfully restored the backup '{backup_name}' from {backup_path}.")
    except Exception as e:
        click.echo(f"Error restoring backup: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
