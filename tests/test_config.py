"""
This module contains test cases for the PyDiagno configuration system.

These tests cover various aspects of the configuration, including:
- LLM deployment configurations
- Model abstraction settings
- RAG (Retrieval-Augmented Generation) configuration
- Kubernetes deployment options
- Full configuration loading and validation
- Handling of invalid configurations
- Default configuration values

The tests use the `sample_config` fixture defined in conftest.py.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from pydantic import ValidationError

from pydiagno.config import (
    KubernetesConfig,
    LLMConfig,
    ModelAbstractionConfig,
    PyDiagnoConfig,
    RAGConfig,
    load_config,
)


def test_llm_config(sample_config: Dict[str, Any]) -> None:
    llm_config = LLMConfig(**sample_config["llm"])
    assert len(llm_config.deployments) == 3
    assert llm_config.deployments[0].name == "primary"
    assert llm_config.deployments[1].provider == "local"
    assert (
        llm_config.deployments[2].ssh
        and llm_config.deployments[2].ssh.hostname == "remote_host"
    )
    assert llm_config.selection_strategy == "priority"


def test_model_abstraction_config(sample_config: Dict[str, Any]) -> None:
    model_config = ModelAbstractionConfig(**sample_config["model_abstraction"])
    assert model_config.cache_size == 2048
    assert model_config.default_format == "onnx"
    assert "guff" in model_config.supported_formats
    assert len(model_config.model_configurations) == 1
    assert model_config.model_configurations[0].name == "primary"
    assert model_config.model_configurations[0].quantization is True


def test_rag_config(sample_config: Dict[str, Any]) -> None:
    rag_config = RAGConfig(**sample_config["rag"])
    assert rag_config.enabled is True
    assert rag_config.database.type == "sqlite"
    assert rag_config.cache_size == 1024


def test_kubernetes_config(sample_config: Dict[str, Any]) -> None:
    k8s_config = KubernetesConfig(**sample_config["kubernetes"])
    assert k8s_config.enabled is True
    assert k8s_config.namespace == "test-namespace"
    assert k8s_config.resources.requests["cpu"] == "500m"
    assert k8s_config.auto_scaling.min_replicas == 2


def test_load_config(tmp_path: Path, sample_config: Dict[str, Any]) -> None:
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)

    loaded_config = load_config(str(config_path))
    assert isinstance(loaded_config, PyDiagnoConfig)
    assert len(loaded_config.llm.deployments) == 3
    assert loaded_config.model_abstraction.cache_size == 2048
    assert loaded_config.rag.enabled is True
    assert loaded_config.kubernetes.namespace == "test-namespace"


def test_full_config(sample_config: Dict[str, Any]) -> None:
    config = PyDiagnoConfig.model_validate(sample_config)
    assert isinstance(config.llm, LLMConfig)
    assert isinstance(config.model_abstraction, ModelAbstractionConfig)
    assert isinstance(config.rag, RAGConfig)
    assert isinstance(config.kubernetes, KubernetesConfig)


def test_invalid_config() -> None:
    invalid_configs = [
        {
            "deployments": [
                {
                    "name": "invalid",
                    "provider": "unknown",
                }
            ]
        },
        {
            "deployments": [
                {
                    "name": "invalid",
                    "provider": "local",
                    "model_path": None,
                }
            ]
        },
        {
            "deployments": [
                {
                    "name": "invalid",
                    "provider": "ssh",
                    "ssh": None,
                }
            ]
        },
    ]
    for invalid_config in invalid_configs:
        with pytest.raises(ValidationError):
            LLMConfig(deployments=[invalid_config])


def test_default_config() -> None:
    default_config = PyDiagnoConfig()
    assert len(default_config.llm.deployments) == 0
    assert default_config.model_abstraction.cache_size == 2048
    assert default_config.rag.enabled is False
    assert default_config.kubernetes.enabled is False


def test_load_nonexistent_config() -> None:
    config = load_config("nonexistent_config.yaml")
    assert isinstance(config, PyDiagnoConfig)
    # Check that it's using default values
    assert len(config.llm.deployments) == 0
    assert config.model_abstraction.cache_size == 2048
