# tests/test_plugin.py
import time
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
    """
    Test the pytest_runtest_makereport hook functionality.

    This test verifies that the pytest_runtest_makereport hook correctly handles
    test reports with and without PyDiagno enabled. It checks the following:
    1. The hook correctly processes the outcome when it's available.
    2. The hook can create a report when the outcome is None.
    3. The PyDiagno marker is correctly identified and processed.
    4. The resulting TestReport object has the correct attributes.

    Args:
        pydiagno_enabled (bool): Whether PyDiagno is enabled for the test.

    The test creates mock objects for Item, CallInfo, and Config, simulating
    a pytest environment. It then runs the hook twice: once with a mock outcome
    and once with None as the outcome.
    """
    item = Mock(spec=Item)
    item.keywords = {"test_name": 1, "pydiagno": pydiagno_enabled}
    item._report_sections = []
    item.user_properties = []
    call = Mock(spec=pytest.CallInfo)
    call.when = "call"
    call.start = time.time()
    call.stop = call.start + 0.1
    call.duration = call.stop - call.start
    call.excinfo = None

    config = Mock(spec=Config)
    config.getoption.return_value = pydiagno_enabled
    item.config = config

    marker = Mock() if pydiagno_enabled else None
    item.get_closest_marker.return_value = marker

    report = Mock(spec=TestReport)
    report.when = "call"

    outcome = Mock()
    outcome.get_result.return_value = report

    hook = plugin.pytest_runtest_makereport(item, call)
    next(hook)  # Start the generator

    try:
        # Send the outcome and get the final result
        final_result = hook.send(outcome)
    except StopIteration as exc:
        final_result = exc.value

    assert isinstance(final_result, TestReport)
    assert final_result == report

    item.get_closest_marker.assert_called_once_with("pydiagno")

    # Test the case where outcome.get_result() doesn't exist
    # Reset the mock to clear previous calls
    item.get_closest_marker.reset_mock()
    hook = plugin.pytest_runtest_makereport(item, call)
    next(hook)
    try:
        final_result = hook.send(None)
    except StopIteration as exc:
        final_result = exc.value

    assert isinstance(final_result, TestReport)
    assert final_result != report
    assert final_result.when == "call"
    assert abs(final_result.duration - call.duration) < 0.001
    assert abs(final_result.start - call.start) < 0.001
    assert abs(final_result.stop - call.stop) < 0.001

    # Check that keywords are correctly set,
    # accounting for potential modification of pydiagno value
    assert final_result.keywords["test_name"] == item.keywords["test_name"]
    assert "pydiagno" in final_result.keywords
    # Check that user_properties are correctly set
    assert final_result.user_properties == item.user_properties

    # The marker should be checked regardless of pydiagno_enabled
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
    pm.import_plugin("pydiagno.plugin")

    assert pm.has_plugin("pydiagno.plugin")
    hook_callers = pm.get_hookcallers(plugin)
    assert hook_callers is not None

    hook_names = [h.name for h in hook_callers]
    assert "pytest_addoption" in hook_names
    assert "pytest_configure" in hook_names
    assert "pytest_runtest_makereport" in hook_names
    assert "pytest_terminal_summary" in hook_names
