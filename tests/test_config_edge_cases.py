import pytest
from pydantic import ValidationError

from pydiagno.config import (
    LLMConfig,
    LLMDeployment,
    ModelAbstractionConfig,
    PyDiagnoConfig,
    RAGConfig,
)


def test_invalid_llm_provider() -> None:
    with pytest.raises(ValidationError) as excinfo:
        LLMDeployment(name="test", provider="invalid_provider")
    assert "Invalid provider 'invalid_provider'. Must be one of:" in str(excinfo.value)


def test_missing_model_path_for_local_provider() -> None:
    with pytest.raises(ValidationError) as excinfo:
        LLMConfig(deployments=[{"name": "test", "provider": "local"}])
    assert "model_path is required for local provider" in str(excinfo.value)


def test_missing_ssh_config_for_ssh_provider() -> None:
    with pytest.raises(ValidationError) as excinfo:
        LLMConfig(deployments=[{"name": "test", "provider": "ssh"}])
    assert "SSH configuration is required for SSH provider" in str(excinfo.value)


def test_invalid_model_format() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ModelAbstractionConfig(default_format="invalid_format")
    assert "Invalid model format. Must be one of: onnx, guff, ggml" in str(
        excinfo.value
    )


def test_negative_cache_size() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ModelAbstractionConfig(cache_size=-1)
    assert "Cache size must be non-negative" in str(excinfo.value)


def test_invalid_rag_database_type() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RAGConfig(database={"type": "invalid_type"})
    assert "Invalid database type. Must be either 'sqlite' or 'postgresql'" in str(
        excinfo.value
    )


def test_invalid_log_level() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PyDiagnoConfig(monitoring={"log_level": "INVALID"})
    assert "Invalid log level. Must be one of:" in str(excinfo.value)


def test_invalid_rate_limit_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PyDiagnoConfig(rate_limit={"requests_per_minute": -1})
    assert "Value must be non-negative" in str(excinfo.value)


def test_invalid_circuit_breaker_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PyDiagnoConfig(circuit_breaker={"failure_threshold": -1})
    assert "Value must be non-negative" in str(excinfo.value)


def test_invalid_kubernetes_resource_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PyDiagnoConfig(kubernetes={"resources": {"requests": {"cpu": "invalid"}}})
    assert "Invalid resource value" in str(excinfo.value)
