# tests/conftest.py

from unittest.mock import Mock

import pytest
from _pytest.config import Config
from typing import Dict, Any


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
            "model_config": [
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
    }
