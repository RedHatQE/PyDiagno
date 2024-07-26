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

PyDiagno uses a YAML configuration file. Here's a basic example:

```yaml
llm:
  model: "huggingface/model-name"
  api_key: "your-api-key"

rag:
  enabled: true
  data_path: "./historical_data"

report:
  format: "json"
  output_path: "./pydiagno_reports"
```

For more detailed configuration options, see our [configuration documentation](docs/configuration.md).

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