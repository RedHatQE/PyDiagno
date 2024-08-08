# PyDiagno

PyDiagno is a pytest plugin designed to enhance the efficiency of automated testing processes by leveraging the capabilities of Large Language Models (LLMs) for intelligent test failure analysis.

## Features

- Dynamic LLM Integration: Configurable Large Language Model integration for adaptable analysis.
- Automated TFA Process: Leveraging LLMs for root cause analysis of test failures.
- Basic RAG Feature: Enhanced diagnostics using historical test data.
- User Control and Customization: Extensive configuration for personalized tool operation.
- Framework and Environment Adaptability: Support for pytest with future expansions planned.

## Installation

You can install PyDiagno using pip:

```bash
pip install pydiagno
```

## Quick Start

1. Install PyDiagno in your project.
2. Create a `pydiagno.yaml` configuration file in your project root.
3. Run your pytest suite as usual. PyDiagno will automatically analyze test failures.

## Configuration

PyDiagno uses a YAML configuration file for customization. By default, it looks for `pydiagno_config.yaml` in the project root directory. Sensitive data should be stored in a `.env` file.
For a full list of configuration options, please refer to the `config.py` file in the `src/pydiagno` directory.


### Key Configuration Sections:

1. **LLM Configuration**: Define multiple LLM deployments (local, remote, SSH-based).
2. **Model Abstraction**: Manage model loading, caching, and format support.
3. **Log Processing**: Configure additional log handling beyond pytest.
4. **Analysis Settings**: Set confidence thresholds and iteration limits.
5. **RAG Configuration**: Enable and configure Retrieval-Augmented Generation.
6. **Reporting Options**: Customize analysis report format and location.
7. **Plugin Management**: Enable and discover plugins.
8. **Security Settings**: Configure data masking and encryption.
9. **DUT Configuration**: Set up Device Under Test communication.
10. **Event Bus Settings**: Configure inter-component communication.
11. **Resource Management**: Set resource usage limits.
12. **Kubernetes Options**: Configure deployment in Kubernetes environments.
13. **Performance Tuning**: Optimize parallel processing and caching.
14. **Monitoring and Logging**: Set up observability and debugging options.

### Example Configuration:

```yaml
llm:
  deployments:
    - name: "primary"
      provider: "openai"
      model: "gpt-3.5-turbo"
      api_key: "your-api-key-here"
    - name: "local"
      provider: "local"
      model_path: "/path/to/local/model"

model_abstraction:
  cache_size: 2048
  default_format: "onnx"
  model_config:
    - name: "primary"
      format: "onnx"
      quantization: true

analysis:
  confidence_threshold: 0.8
  max_iterations: 5

rag:
  enabled: true
  database:
    type: "sqlite"
    path: "pydiagno_rag.db"

reporting:
  format: "json"
  output_path: "./pydiagno_reports"

# Additional sections can be configured as needed
```

For a full list of configuration options and their default values, please refer to the `config.py` file in the `src/pydiagno` directory.

## Project Structure

- `src/pydiagno/`: Contains the main plugin code
  - `plugin.py`: Implements the pytest hooks for PyDiagno
- `tests/`: Contains the test files
- `pyproject.toml`: Defines the project dependencies and configuration

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## Development

To set up the development environment:

1. Clone the repository
2. Install Poetry: `pip install poetry`
3. Install dependencies: `poetry install`
4. Activate the virtual environment: `poetry shell`
5. Run tests: `pytest`

We use pre-commit hooks to ensure code quality. Install them with:

```bash
pre-commit install
```

## License

PyDiagno is released under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.