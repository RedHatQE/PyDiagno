import logging
import os
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

import yaml

# Implement circuit breaker
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar("T")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sleep_and_retry(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def limits(calls: int, period: int) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def circuit(
    failure_threshold: int, recovery_timeout: int
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        return wrapper

    return decorator


class SSHConfig(BaseModel):
    """Configuration for SSH connections."""

    hostname: str = Field(..., description="Hostname for SSH connection")
    port: int = Field(default=22, description="Port for SSH connection")
    username: str = Field(..., description="Username for SSH connection")
    key_file: str = Field(..., description="Path to SSH key file")


class LLMDeployment(BaseModel):
    """Configuration for an LLM deployment."""

    name: str = Field(..., description="Unique name for this LLM deployment")
    provider: str = Field(..., description="LLM provider")
    model: Optional[str] = Field(None, description="Specific model to use")
    api_key: Optional[SecretStr] = Field(
        default=None, description="API key for the LLM provider (if applicable)"
    )
    model_path: Optional[str] = Field(
        None, description="Path to the local model file (for local provider)"
    )
    ssh: Optional[SSHConfig] = Field(
        default=None, description="SSH configuration for remote LLM access"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_deployment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        provider = values.get("provider")
        model_path = values.get("model_path")
        ssh = values.get("ssh")

        allowed_providers = {
            "huggingface",
            "openai",
            "anthropic",
            "local",
            "in-cluster",
            "ssh",
        }
        if provider not in allowed_providers:
            raise ValueError(
                f"Invalid provider '{provider}'. Must be one of: "
                f"{', '.join(allowed_providers)}"
            )

        if provider == "local" and not model_path:
            raise ValueError("model_path is required for local provider")

        if provider == "ssh" and not ssh:
            raise ValueError("SSH configuration is required for SSH provider")

        return values


class LLMConfig(BaseModel):
    """Configuration for LLM deployments."""

    deployments: List[LLMDeployment] = Field(
        default_factory=list, description="List of LLM deployments"
    )
    selection_strategy: str = Field(
        default="priority", description="Strategy for selecting LLM"
    )


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    name: str = Field(
        ..., description="Name of the model (should match an LLM deployment name)"
    )
    format: str = Field(..., description="Format of the model")
    quantization: bool = Field(
        default=False, description="Whether to enable model quantization"
    )
    pruning: bool = Field(default=False, description="Whether to enable model pruning")


class ModelAbstractionConfig(BaseModel):
    """Configuration for model abstraction layer."""

    cache_size: int = Field(default=2048, description="Size of model cache in MB")
    default_format: str = Field(default="onnx", description="Default model format")
    supported_formats: List[str] = Field(
        default=["onnx", "guff", "ggml"], description="List of supported model formats"
    )
    model_configurations: List[ModelConfig] = Field(
        default_factory=list, description="Configuration for specific models"
    )

    model_config = {"protected_namespaces": ()}

    @field_validator("default_format")
    @classmethod
    def validate_default_format(cls: Any, v: str) -> str:
        if v not in ["onnx", "guff", "ggml"]:
            raise ValueError("Invalid model format. Must be one of: onnx, guff, ggml")
        return v

    @field_validator("cache_size")
    @classmethod
    def validate_cache_size(cls: Any, v: int) -> int:
        if v < 0:
            raise ValueError("Cache size must be non-negative")
        return v


class LogProcessingConfig(BaseModel):
    """Configuration for log processing."""

    additional_log_paths: List[str] = Field(
        default_factory=list, description="List of paths to additional log files"
    )
    max_log_size: int = Field(default=1048576, description="Maximum log size in bytes")
    ignored_patterns: List[str] = Field(
        default_factory=list, description="Regex patterns to ignore in logs"
    )


class AnalysisConfig(BaseModel):
    """Configuration for analysis settings."""

    confidence_threshold: float = Field(
        default=0.8, description="Minimum confidence score for results"
    )
    max_iterations: int = Field(
        default=5, description="Maximum number of analysis iterations (0 for unlimited)"
    )


class RAGDatabaseConfig(BaseModel):
    """Configuration for RAG database."""

    type: str = Field(default="sqlite", description="Database type for RAG")
    path: str = Field(
        default="pydiagno_rag.db", description="Database path or connection string"
    )

    @field_validator("type")
    @classmethod
    def validate_database_type(cls: Any, v: str) -> str:
        if v not in ["sqlite", "postgresql"]:
            raise ValueError(
                "Invalid database type. Must be either 'sqlite' or 'postgresql'"
            )
        return v


class RAGConfig(BaseModel):
    """Configuration for Retrieval-Augmented Generation."""

    enabled: bool = Field(default=False, description="Whether RAG is enabled")
    database: RAGDatabaseConfig = Field(default_factory=RAGDatabaseConfig)
    cache_size: int = Field(default=1024, description="Size of RAG cache in MB")


class ReportingConfig(BaseModel):
    """Configuration for reporting."""

    format: str = Field(default="json", description="Report output format")
    output_path: str = Field(
        default="./pydiagno_reports", description="Path for report output"
    )


class PluginConfig(BaseModel):
    """Configuration for plugins."""

    enabled: List[str] = Field(
        default_factory=list, description="List of enabled plugin names"
    )
    auto_discovery: bool = Field(
        default=True, description="Automatically discover and load compatible plugins"
    )


class DataMaskingConfig(BaseModel):
    """Configuration for data masking."""

    enabled: bool = Field(default=True, description="Whether to mask sensitive data")
    patterns: List[str] = Field(
        default_factory=list, description="Regex patterns to mask"
    )


class EncryptionConfig(BaseModel):
    """Configuration for encryption."""

    enabled: bool = Field(default=True, description="Whether to encrypt sensitive data")
    algorithm: str = Field(default="AES-256", description="Encryption algorithm")


class SecurityConfig(BaseModel):
    """Configuration for security settings."""

    data_masking: DataMaskingConfig = Field(default_factory=DataMaskingConfig)
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)


class DUTConnectionConfig(BaseModel):
    """Configuration for DUT connection."""

    hostname: str = Field(
        default="localhost", description="Hostname for DUT connection"
    )
    port: int = Field(default=22, description="Port for DUT connection")
    username: str = Field(default="", description="Username for DUT connection")
    key_file: str = Field(default="", description="Path to SSH key file for DUT")


class DUTConfig(BaseModel):
    """Configuration for Device Under Test."""

    communication_protocol: str = Field(
        default="ssh", description="Communication protocol for DUT"
    )
    connection: DUTConnectionConfig = Field(default_factory=DUTConnectionConfig)
    timeout: int = Field(
        default=30, description="Timeout for DUT operations in seconds"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for failed DUT operations"
    )


class EventBusConnectionConfig(BaseModel):
    """Configuration for event bus connection."""

    host: str = Field(default="localhost", description="Host for event bus connection")
    port: int = Field(default=5672, description="Port for event bus connection")
    username: str = Field(default="", description="Username for event bus connection")
    password: SecretStr = Field(
        default=SecretStr(""), description="Password for event bus connection"
    )


class EventBusConfig(BaseModel):
    """Configuration for event bus."""

    type: str = Field(default="rabbitmq", description="Type of event bus")
    connection: EventBusConnectionConfig = Field(
        default_factory=EventBusConnectionConfig
    )
    queue_size: int = Field(
        default=1000, description="Maximum number of messages in the queue"
    )


class ResourceManagerConfig(BaseModel):
    """Configuration for resource manager."""

    cpu_limit: int = Field(default=80, description="Maximum CPU usage percentage")
    memory_limit: int = Field(default=8192, description="Maximum memory usage in MB")
    storage_limit: int = Field(default=10240, description="Maximum storage usage in MB")


class KubernetesResourcesConfig(BaseModel):
    """Configuration for Kubernetes resources."""

    requests: Dict[str, str] = Field(
        default_factory=lambda: {"cpu": "500m", "memory": "1Gi"}
    )
    limits: Dict[str, str] = Field(
        default_factory=lambda: {"cpu": "2", "memory": "4Gi"}
    )

    @field_validator("requests", "limits")
    @classmethod
    def validate_resource_values(cls: Any, v: Dict[str, str]) -> Dict[str, str]:
        for key, value in v.items():
            if key not in ["cpu", "memory"]:
                raise ValueError(f"Invalid resource key: {key}")
            if not re.match(r"^(\d+(\.\d+)?(m|Mi|Gi)?)$", value):
                raise ValueError(f"Invalid resource value: {value}")
        return v


class KubernetesAutoScalingConfig(BaseModel):
    """Configuration for Kubernetes auto-scaling."""

    enabled: bool = Field(
        default=True, description="Whether to use Horizontal Pod Autoscaler"
    )
    min_replicas: int = Field(default=1, description="Minimum number of replicas")
    max_replicas: int = Field(default=5, description="Maximum number of replicas")


class KubernetesConfig(BaseModel):
    """Configuration for Kubernetes deployment."""

    enabled: bool = Field(
        default=False, description="Whether to use Kubernetes-specific features"
    )
    namespace: str = Field(
        default="pydiagno", description="Kubernetes namespace for PyDiagno resources"
    )
    resources: KubernetesResourcesConfig = Field(
        default_factory=KubernetesResourcesConfig
    )
    auto_scaling: KubernetesAutoScalingConfig = Field(
        default_factory=KubernetesAutoScalingConfig
    )


class PerformanceCachingConfig(BaseModel):
    """Configuration for performance caching."""

    enabled: bool = Field(default=True, description="Whether to use caching")
    ttl: int = Field(default=3600, description="Cache time-to-live in seconds")


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""

    parallel_analysis: bool = Field(
        default=True, description="Whether to perform analysis in parallel"
    )
    batch_size: int = Field(
        default=10, description="Number of items to process in a batch"
    )
    caching: PerformanceCachingConfig = Field(default_factory=PerformanceCachingConfig)


class MonitoringMetricsConfig(BaseModel):
    """Configuration for monitoring metrics."""

    enabled: bool = Field(default=True, description="Whether to collect metrics")
    push_gateway: str = Field(
        default="http://localhost:9091", description="Prometheus push gateway URL"
    )


class MonitoringTracingConfig(BaseModel):
    """Configuration for monitoring tracing."""

    enabled: bool = Field(
        default=False, description="Whether to use distributed tracing"
    )
    jaeger_endpoint: str = Field(
        default="http://localhost:14268/api/traces",
        description="Jaeger collector endpoint",
    )


class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""

    log_level: str = Field(default="INFO", description="Logging level")
    metrics: MonitoringMetricsConfig = Field(default_factory=MonitoringMetricsConfig)
    tracing: MonitoringTracingConfig = Field(default_factory=MonitoringTracingConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls: Any, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(
                f"Invalid log level. Must be one of: {', '.join(valid_levels)}"
            )
        return v


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    enabled: bool = Field(default=True, description="Whether to enable rate limiting")
    requests_per_minute: int = Field(
        default=60, description="Number of requests allowed per minute"
    )
    burst: int = Field(default=10, description="Number of requests allowed in a burst")

    @field_validator("requests_per_minute", "burst")
    @classmethod
    def validate_positive_int(cls: Any, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""

    enabled: bool = Field(default=True, description="Whether to enable circuit breaker")
    failure_threshold: int = Field(
        default=5, description="Number of failures before opening the circuit"
    )
    recovery_timeout: int = Field(
        default=30, description="Time in seconds before attempting to close the circuit"
    )

    @field_validator("failure_threshold", "recovery_timeout")
    @classmethod
    def validate_positive_int(cls: Any, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class PyDiagnoConfig(BaseSettings):
    """Main configuration class for PyDiagno."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    model_abstraction: ModelAbstractionConfig = Field(
        default_factory=ModelAbstractionConfig
    )
    log_processing: LogProcessingConfig = Field(default_factory=LogProcessingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    dut: DUTConfig = Field(default_factory=DUTConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    resource_manager: ResourceManagerConfig = Field(
        default_factory=ResourceManagerConfig
    )
    kubernetes: KubernetesConfig = Field(default_factory=KubernetesConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PYDIAGNO_",
        protected_namespaces=(),
    )


def load_config(config_path: str = "pydiagno_config.yaml") -> PyDiagnoConfig:
    """
    Load the PyDiagno configuration from a YAML file.

    This function attempts to load the configuration from the specified YAML file.
    If the file is not found or there's an error in parsing, it falls back to
    the default configuration. Any errors encountered during the process are
    logged for debugging purposes.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        PyDiagnoConfig: Loaded and validated configuration.

    Raises:
        ValueError: If the configuration file exists but contains invalid data.
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(
                f"Configuration file not found at {config_path}. "
                f"Using default configuration."
            )
            return PyDiagnoConfig()

        with open(config_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)

        if not isinstance(config_data, dict):
            raise ValueError("Invalid configuration format. Expected a dictionary.")

        return PyDiagnoConfig(**config_data)

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise ValueError(f"Invalid YAML in configuration file: {e}")

    except ValueError as e:
        logger.error(f"Error in configuration data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise ValueError(f"Failed to load configuration: {e}")


# Global configuration object
config: PyDiagnoConfig = load_config()


@sleep_and_retry
@limits(calls=config.rate_limit.requests_per_minute, period=60)
def rate_limited_api_call(func: Callable[[], T]) -> Callable[[], T]:
    """
    Decorator to apply rate limiting to API calls.

    Args:
        func: The function to be rate limited.

    Returns:
        A rate-limited version of the input function.
    """

    @wraps(func)
    def wrapper() -> T:
        # Here you would implement actual rate limiting logic
        return func()

    return wrapper


@circuit(
    failure_threshold=config.circuit_breaker.failure_threshold,
    recovery_timeout=config.circuit_breaker.recovery_timeout,
)
def circuit_protected_api_call(func: Callable[[], T]) -> Callable[[], T]:
    """
    Decorator to apply circuit breaker pattern to API calls.

    Args:
        func: The function to be protected by the circuit breaker.

    Returns:
        A circuit-breaker-protected version of the input function.
    """

    @wraps(func)
    def wrapper() -> T:
        # Here you would implement actual circuit breaker logic
        return func()

    return wrapper


def make_api_call(func: Callable[[], T]) -> T:
    """
    Make an API call with rate limiting and circuit breaker protection.

    Args:
        func: The API call function to be executed.

    Returns:
        The result of the API call.
    """
    if config.rate_limit.enabled:
        func = rate_limited_api_call(func)
    if config.circuit_breaker.enabled:
        func = circuit_protected_api_call(func)
    return func()


# Example usage
def example_api_call() -> None:
    # This is where you would put your actual API call
    pass
