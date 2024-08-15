# tests/conftest.py

from typing import Any, Dict
from unittest.mock import Mock

import pytest
from _pytest.config import Config


@pytest.fixture
def mock_config() -> Mock:
    """Fixture for a mocked Config object with PyDiagno enabled."""
    config = Mock(spec=Config)
    config.getoption.return_value = True
    return config


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Provide a sample configuration for PyDiagno tests.

    This fixture returns a dictionary representing a complete PyDiagno configuration,
    including settings for LLM deployments, model abstraction, RAG, and Kubernetes.

    Returns:
        Dict[str, Any]: A dictionary containing a sample PyDiagno configuration.

    Example:
        def test_something(sample_config):
            assert sample_config['llm']['deployments'][0]['name'] == 'primary'
    """
    return {
        "llm": {
            "deployments": [
                {
                    "name": "primary",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                },
                {
                    "name": "local",
                    "provider": "local",
                    "model_path": "/path/to/local/model",
                },
                {
                    "name": "ssh_llm",
                    "provider": "ssh",
                    "model": "remote_model",
                    "ssh": {
                        "hostname": "remote_host",
                        "port": 22,
                        "username": "user",
                        "key_file": "/path/to/ssh/key",
                    },
                },
            ],
            "selection_strategy": "priority",
        },
        "model_abstraction": {
            "cache_size": 2048,
            "default_format": "onnx",
            "supported_formats": ["onnx", "guff", "ggml"],
            "model_configurations": [
                {
                    "name": "primary",
                    "format": "onnx",
                    "quantization": True,
                    "pruning": False,
                },
            ],
        },
        "rag": {
            "enabled": True,
            "database": {
                "type": "sqlite",
                "path": "test_rag.db",
            },
            "cache_size": 1024,
        },
        "kubernetes": {
            "enabled": True,
            "namespace": "test-namespace",
            "resources": {
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"cpu": "1", "memory": "2Gi"},
            },
            "auto_scaling": {
                "enabled": True,
                "min_replicas": 2,
                "max_replicas": 5,
            },
        },
        "log_processing": {
            "additional_log_paths": ["/path/to/additional/log"],
            "max_log_size": 1048576,
            "ignored_patterns": [".*ignore_this.*"],
        },
        "analysis": {
            "confidence_threshold": 0.8,
            "max_iterations": 5,
        },
        "reporting": {
            "format": "json",
            "output_path": "./pydiagno_reports",
        },
        "plugins": {
            "enabled": ["sample_plugin"],
            "auto_discovery": True,
        },
        "security": {
            "data_masking": {
                "enabled": True,
                "patterns": ["password=.*"],
            },
            "encryption": {
                "enabled": True,
                "algorithm": "AES-256",
            },
        },
        "dut": {
            "communication_protocol": "ssh",
            "connection": {
                "hostname": "test_dut",
                "port": 22,
                "username": "test_user",
                "key_file": "/path/to/key",
            },
            "timeout": 30,
            "max_retries": 3,
        },
        "event_bus": {
            "type": "rabbitmq",
            "connection": {
                "host": "localhost",
                "port": 5672,
                "username": "guest",
                "password": "guest",
            },
            "queue_size": 1000,
        },
        "resource_manager": {
            "cpu_limit": 80,
            "memory_limit": 8192,
            "storage_limit": 10240,
        },
        "monitoring": {
            "log_level": "INFO",
            "metrics": {
                "enabled": True,
                "push_gateway": "http://localhost:9091",
            },
            "tracing": {
                "enabled": False,
                "jaeger_endpoint": "http://localhost:14268/api/traces",
            },
        },
    }


@pytest.fixture
def mock_pydiagno_config(sample_config: Dict[str, Any]) -> Mock:
    """
    Fixture for a mocked PyDiagnoConfig object.

    This fixture creates a Mock object that mimics the structure of a PyDiagnoConfig
    instance, using the sample_config fixture as a basis for its attributes.

    Returns:
        Mock: A mocked PyDiagnoConfig object.

    Example:
        def test_something(mock_pydiagno_config):
            assert mock_pydiagno_config.llm.deployments[0].name == 'primary'
    """
    mock_config = Mock()

    def getattr_side_effect(name: str) -> Any:
        if name in sample_config:
            return Mock(**sample_config[name])
        raise AttributeError(
            f"'{type(mock_config).__name__}' object has no attribute '{name}'"
        )

    mock_config.configure_mock(__getattr__=getattr_side_effect)

    return mock_config
