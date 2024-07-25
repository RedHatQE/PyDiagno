# tests/test_plugin.py

from unittest.mock import Mock

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter

from pydiagno import plugin


def test_pytest_addoption() -> None:
    """Test that pytest_addoption adds the PyDiagno option correctly."""
    parser = Mock(spec=Parser)
    plugin.pytest_addoption(parser)
    parser.getgroup.assert_called_once_with("pydiagno")
    parser.getgroup.return_value.addoption.assert_called_once()


def test_pytest_configure() -> None:
    """Test that pytest_configure adds the PyDiagno marker correctly."""
    config = Mock(spec=Config)
    plugin.pytest_configure(config)
    config.addinivalue_line.assert_called_once_with(
        "markers", "pydiagno: mark test for PyDiagno analysis"
    )
    assert config.getoption.called


@pytest.mark.parametrize("pydiagno_enabled", [True, False])
def test_pytest_runtest_makereport(pydiagno_enabled: bool) -> None:
    """Test pytest_runtest_makereport functionality."""
    item = Mock(spec=Item)
    call = Mock(spec=pytest.CallInfo)

    config = Mock(spec=Config)
    config.getoption.return_value = pydiagno_enabled
    item.config = config

    marker = Mock() if pydiagno_enabled else None
    item.get_closest_marker.return_value = marker

    report = Mock(spec=TestReport)
    report.when = "call"

    outcome = Mock()
    outcome.get_result.return_value = report

    generator = plugin.pytest_runtest_makereport(item, call)
    next(generator)
    result = generator.send(outcome)

    assert result == report
    if pydiagno_enabled:
        item.get_closest_marker.assert_called_once_with("pydiagno")


def test_pytest_terminal_summary() -> None:
    """Test that pytest_terminal_summary writes the PyDiagno summary when enabled."""
    terminalreporter = Mock(spec=TerminalReporter)
    config = Mock(spec=Config)
    config.getoption.return_value = True

    plugin.pytest_terminal_summary(terminalreporter, 0, config)

    terminalreporter.write_sep.assert_called_once()
    terminalreporter.write_line.assert_called_once()


def test_plugin_integration(mock_config: Mock) -> None:
    """Test that all plugin hooks are registered correctly."""
    pm = pytest.PytestPluginManager()
    pm.import_plugin(plugin)

    assert pm.has_plugin("pydiagno")
    hook_callers = pm.get_hookcallers(plugin)
    assert hook_callers is not None

    hook_names = [h.name for h in hook_callers]
    assert "pytest_addoption" in hook_names
    assert "pytest_configure" in hook_names
    assert "pytest_runtest_makereport" in hook_names
    assert "pytest_terminal_summary" in hook_names
