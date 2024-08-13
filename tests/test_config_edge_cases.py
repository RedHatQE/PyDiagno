import pytest
from pydantic import ValidationError

from pydiagno.config import (
    AnalysisConfig,
    KubernetesResourcesConfig,
    LLMConfig,
    LLMDeployment,
    ModelAbstractionConfig,
    PyDiagnoConfig,
    RAGConfig,
    ReportingConfig,
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


def test_invalid_confidence_threshold() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AnalysisConfig(confidence_threshold=2.0)
    assert "Input should be less than or equal to 1" in str(excinfo.value)


def test_negative_max_iterations() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AnalysisConfig(max_iterations=-1)
    assert "Input should be greater than or equal to 0" in str(excinfo.value)


def test_invalid_report_format() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ReportingConfig(format="invalid_format")
    assert "Input should be 'json', 'yaml' or 'text'" in str(excinfo.value)


def test_invalid_kubernetes_resource_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        KubernetesResourcesConfig(requests={"cpu": "invalid"})
    assert "Invalid resource value: invalid" in str(excinfo.value)


# ADDED: New test for combined configuration validation
def test_combined_config_validation() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PyDiagnoConfig(
            llm={
                "deployments": [
                    {
                        "name": "test",
                        "provider": "openai",
                        "model": "gpt-4",
                        "api_key": "test-key",
                    }
                ]
            },
            model_abstraction={
                "cache_size": -1,
                "default_format": "invalid",
            },
            monitoring={
                "log_level": "INVALID",
            },
            analysis={
                "confidence_threshold": 2.0,
                "max_iterations": -1,
            },
            rag={
                "database": {
                    "type": "invalid_type",
                }
            },
            reporting={
                "format": "invalid_format",
            },
            kubernetes={
                "resources": {
                    "requests": {"cpu": "invalid"},
                }
            },
        )

    error_str = str(excinfo.value)
    assert "Invalid model format" in error_str
    assert "Cache size must be non-negative" in error_str
    assert "Invalid log level" in error_str
    assert "Input should be less than or equal to 1" in error_str
    assert "Input should be greater than or equal to 0" in error_str
    assert "Invalid database type" in error_str
    assert "Input should be 'json', 'yaml' or 'text'" in error_str
    assert "Invalid resource value: invalid" in error_str


def test_invalid_rate_limit_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PyDiagnoConfig(rate_limit={"requests_per_minute": -1})
    assert "Value must be non-negative" in str(excinfo.value)


def test_invalid_circuit_breaker_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PyDiagnoConfig(circuit_breaker={"failure_threshold": -1})
    assert "Value must be non-negative" in str(excinfo.value)
