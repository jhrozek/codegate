import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from ..config.settings import settings

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("proxy_pilot")
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S%f'
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(
        log_dir / "proxy_pilot.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def serialize_for_logging(obj: Any) -> Any:
    """Serialize objects for logging, handling non-JSON serializable types"""
    if hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, (datetime, bytes)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_logging(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_logging(item) for item in obj]
    return obj

def log_request(method: str, path: str, status_code: int, client: Any) -> None:
    """Log HTTP request details"""
    logger = logging.getLogger("proxy_pilot")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "request",
        "method": method,
        "path": path,
        "status_code": status_code,
        "client": serialize_for_logging(client)
    }
    logger.info(f"Request: {json.dumps(log_data, indent=2)}")

def log_proxy_forward(target_url: str, method: str, status_code: int) -> None:
    """Log proxy forwarding details"""
    logger = logging.getLogger("proxy_pilot")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "proxy_forward",
        "target_url": target_url,
        "method": method,
        "status_code": status_code
    }
    logger.info(f"Proxy Forward: {json.dumps(log_data, indent=2)}")

def log_error(error_type: str, message: str, details: Dict = None) -> None:
    """Log error details"""
    logger = logging.getLogger("proxy_pilot")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "error",
        "error_type": error_type,
        "message": message,
        "details": serialize_for_logging(details or {})
    }
    logger.error(f"Error: {json.dumps(log_data, indent=2)}")

# Initialize logger
logger = setup_logging()


logger 