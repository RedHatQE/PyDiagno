import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class SSHConfig(BaseModel):
    hostname: str = Field(..., description="Hostname for SSH connection")
    port: int = Field(default=22, description="Port for SSH connection")
    username: str = Field(..., description="Username for SSH connection")
    key_file: str = Field(..., description="Path to SSH key file")


class LLMDeployment(BaseModel):
    name: str = Field(..., description="Unique name for this LLM deployment")
    provider: str = Field(..., description="LLM provider (e.g., huggingface, openai, anthropic, local, in-cluster, ssh)")
    model: str = Field(..., description="Specific model to use")
    api_key: Optional[str] = Field(default="", description="API key for the LLM provider (if applicable)")
    model_path: Optional[str] = Field(default="", description="Path to the local model file (for local provider)")
    ssh: Optional[SSHConfig] = Field(default=None, description="SSH configuration for remote LLM access")


class LLMConfig(BaseModel):
    deployments: List[LLMDeployment] = Field(default_factory=list, description="List of LLM deployments")
    selection_strategy: str = Field(default="priority", description="Strategy for selecting LLM")


class ModelConfig(BaseModel):
    name: str = Field(..., description="Name of the model (should match an LLM deployment name)")
    format: str = Field(..., description="Format of the model")
    quantization: bool = Field(default=False, description="Whether to enable model quantization")
    pruning: bool = Field(default=False, description="Whether to enable model pruning")


class ModelAbstractionConfig(BaseModel):
    cache_size: int = Field(default=2048, description="Size of model cache in MB")
    default_format: str = Field(default="onnx", description="Default model format")
    supported_formats: List[str] = Field(default=["onnx", "guff", "ggml"], description="List of supported model formats")
    model_config: List[ModelConfig] = Field(default_factory=list, description="Configuration for specific models")


class LogProcessingConfig(BaseModel):
    additional_log_paths: List[str] = Field(default_factory=list, description="List of paths to additional log files")
    max_log_size: int = Field(default=1048576, description="Maximum log size in bytes")
    ignored_patterns: List[str] = Field(default_factory=list, description="Regex patterns to ignore in logs")


class AnalysisConfig(BaseModel):
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence score for results")
    max_iterations: int = Field(default=5, description="Maximum number of analysis iterations (0 for unlimited)")


class RAGDatabaseConfig(BaseModel):
    type: str = Field(default="sqlite", description="Database type for RAG")
    path: str = Field(default="pydiagno_rag.db", description="Database path or connection string")


class RAGConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether RAG is enabled")
    database: RAGDatabaseConfig = Field(default_factory=RAGDatabaseConfig)
    cache_size: int = Field(default=1024, description="Size of RAG cache in MB")


class ReportingConfig(BaseModel):
    format: str = Field(default="json", description="Report output format")
    output_path: str = Field(default="./pydiagno_reports", description="Path for report output")


class PluginConfig(BaseModel):
    enabled: List[str] = Field(default_factory=list, description="List of enabled plugin names")
    auto_discovery: bool = Field(default=True, description="Automatically discover and load compatible plugins")


class DataMaskingConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to mask sensitive data")
    patterns: List[str] = Field(default_factory=list, description="Regex patterns to mask")


class EncryptionConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to encrypt sensitive data")
    algorithm: str = Field(default="AES-256", description="Encryption algorithm")


class SecurityConfig(BaseModel):
    data_masking: DataMaskingConfig = Field(default_factory=DataMaskingConfig)
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)


class DUTConnectionConfig(BaseModel):
    hostname: str = Field(default="localhost", description="Hostname for DUT connection")
    port: int = Field(default=22, description="Port for DUT connection")
    username: str = Field(default="", description="Username for DUT connection")
    key_file: str = Field(default="", description="Path to SSH key file for DUT")


class DUTConfig(BaseModel):
    communication_protocol: str = Field(default="ssh", description="Communication protocol for DUT")
    connection: DUTConnectionConfig = Field(default_factory=DUTConnectionConfig)
    timeout: int = Field(default=30, description="Timeout for DUT operations in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed DUT operations")


class EventBusConnectionConfig(BaseModel):
    host: str = Field(default="localhost", description="Host for event bus connection")
    port: int = Field(default=5672, description="Port for event bus connection")
    username: str = Field(default="", description="Username for event bus connection")
    password: str = Field(default="", description="Password for event bus connection")


class EventBusConfig(BaseModel):
    type: str = Field(default="rabbitmq", description="Type of event bus")
    connection: EventBusConnectionConfig = Field(default_factory=EventBusConnectionConfig)
    queue_size: int = Field(default=1000, description="Maximum number of messages in the queue")


class ResourceManagerConfig(BaseModel):
    cpu_limit: int = Field(default=80, description="Maximum CPU usage percentage")
    memory_limit: int = Field(default=8192, description="Maximum memory usage in MB")
    storage_limit: int = Field(default=10240, description="Maximum storage usage in MB")


class KubernetesResourcesConfig(BaseModel):
    requests: Dict[str, str] = Field(default_factory=lambda: {"cpu": "500m", "memory": "1Gi"})
    limits: Dict[str, str] = Field(default_factory=lambda: {"cpu": "2", "memory": "4Gi"})


class KubernetesAutoScalingConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to use Horizontal Pod Autoscaler")
    min_replicas: int = Field(default=1, description="Minimum number of replicas")
    max_replicas: int = Field(default=5, description="Maximum number of replicas")


class KubernetesConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether to use Kubernetes-specific features")
    namespace: str = Field(default="pydiagno", description="Kubernetes namespace for PyDiagno resources")
    resources: KubernetesResourcesConfig = Field(default_factory=KubernetesResourcesConfig)
    auto_scaling: KubernetesAutoScalingConfig = Field(default_factory=KubernetesAutoScalingConfig)


class PerformanceCachingConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to use caching")
    ttl: int = Field(default=3600, description="Cache time-to-live in seconds")


class PerformanceConfig(BaseModel):
    parallel_analysis: bool = Field(default=True, description="Whether to perform analysis in parallel")
    batch_size: int = Field(default=10, description="Number of items to process in a batch")
    caching: PerformanceCachingConfig = Field(default_factory=PerformanceCachingConfig)


class MonitoringMetricsConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to collect metrics")
    push_gateway: str = Field(default="http://localhost:9091", description="Prometheus push gateway URL")


class MonitoringTracingConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether to use distributed tracing")
    jaeger_endpoint: str = Field(default="http://localhost:14268/api/traces", description="Jaeger collector endpoint")


class MonitoringConfig(BaseModel):
    log_level: str = Field(default="INFO", description="Logging level")
    metrics: MonitoringMetricsConfig = Field(default_factory=MonitoringMetricsConfig)
    tracing: MonitoringTracingConfig = Field(default_factory=MonitoringTracingConfig)


class PyDiagnoConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    model_abstraction: ModelAbstractionConfig = Field(default_factory=ModelAbstractionConfig)
    log_processing: LogProcessingConfig = Field(default_factory=LogProcessingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    dut: DUTConfig = Field(default_factory=DUTConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    resource_manager: ResourceManagerConfig = Field(default_factory=ResourceManagerConfig)
    kubernetes: KubernetesConfig = Field(default_factory=KubernetesConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_config(config_path: str = "pydiagno_config.yaml") -> PyDiagnoConfig:
    """
    Load the PyDiagno configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        PyDiagnoConfig: Loaded and validated configuration.
    """
    if not os.path.exists(config_path):
        return PyDiagnoConfig()

    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)

    return PyDiagnoConfig(**config_data)


# Global configuration object
config = load_config()