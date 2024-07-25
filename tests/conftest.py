# tests/conftest.py

from unittest.mock import Mock

import pytest
from _pytest.config import Config


@pytest.fixture
def mock_config() -> Mock:
    """Fixture for a mocked Config object with PyDiagno enabled."""
    config = Mock(spec=Config)
    config.getoption.return_value = True
    return config
