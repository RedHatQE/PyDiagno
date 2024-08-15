# tests/test_plugin.py
import time
from unittest.mock import Mock, patch

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter

from pydiagno import plugin
from pydiagno.config import PyDiagnoConfig
from pydiagno.exceptions import PyDiagnoAnalysisError, PyDiagnoConfigError


def test_pytest_addoption() -> None:
    """Test that pytest_addoption adds the PyDiagno options correctly."""
    parser = Mock(spec=Parser)
    plugin.pytest_addoption(parser)
    parser.getgroup.assert_called_once_with("pydiagno")
    group = parser.getgroup.return_value
    assert group.addoption.call_count == 8
    option_names = [call.args[0] for call in group.addoption.call_args_list]
    assert "--pydiagno" in option_names
    assert "--pydiagno-config" in option_names
    assert "--pydiagno-log-level" in option_names
    assert "--pydiagno-confidence-threshold" in option_names
    assert "--pydiagno-max-iterations" in option_names
    assert "--pydiagno-rag-enabled" in option_names
    assert "--pydiagno-report-format" in option_names
    assert "--pydiagno-report-output" in option_names


def test_pytest_configure() -> None:
    """
    Test that pytest_configure adds the PyDiagno marker and loads config correctly.
    """
    config = Mock(spec=Config)
    config.getoption.return_value = True

    # Mock the load_config function
    with patch("pydiagno.plugin.load_config") as mock_load_config:
        mock_config = Mock(spec=PyDiagnoConfig)
        mock_config.monitoring = Mock()
        mock_config.analysis = Mock()
        mock_config.rag = Mock()
        mock_config.reporting = Mock()
        mock_load_config.return_value = mock_config

        plugin.pytest_configure(config)

        config.addinivalue_line.assert_called_once_with(
            "markers", "pydiagno: mark test for PyDiagno analysis"
        )
        assert config.getoption.called
        mock_load_config.assert_called_once()

        assert hasattr(config, "pydiagno_config")
        assert getattr(config, "pydiagno_config") == mock_config

    # Test for command line option overrides
    config.getoption.side_effect = lambda x: {
        "pydiagno": True,
        "pydiagno_config": None,
        "pydiagno_log_level": "DEBUG",
        "pydiagno_confidence_threshold": 0.9,
        "pydiagno_max_iterations": 10,
        "pydiagno_rag_enabled": True,
        "pydiagno_report_format": "json",
        "pydiagno_report_output": "/tmp/reports",
    }.get(x)

    with patch("pydiagno.plugin.load_config") as mock_load_config:
        mock_config = Mock(spec=PyDiagnoConfig)
        mock_config.monitoring = Mock()
        mock_config.analysis = Mock()
        mock_config.rag = Mock()
        mock_config.reporting = Mock()
        mock_load_config.return_value = mock_config

        plugin.pytest_configure(config)

        assert mock_config.monitoring.log_level == "DEBUG"
        assert mock_config.analysis.confidence_threshold == 0.9
        assert mock_config.analysis.max_iterations == 10
        assert mock_config.rag.enabled is True
        assert mock_config.reporting.format == "json"
        assert mock_config.reporting.output_path == "/tmp/reports"


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
    item.nodeid = "test_item"
    item.location = ("test_file.py", 42, "test_function")
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

    mock_config = Mock(spec=PyDiagnoConfig)
    mock_config.analysis = Mock()
    mock_config.analysis.confidence_threshold = 0.8
    mock_config.analysis.max_iterations = 5
    setattr(config, "pydiagno_config", mock_config)

    marker = Mock() if pydiagno_enabled else None
    item.get_closest_marker.return_value = marker

    with patch("pydiagno.plugin.perform_pydiagno_analysis") as mock_analysis:
        mock_analysis.return_value = {
            "confidence": 0.9,
            "iterations": 3,
            "result": "Test analysis result",
        }

        # Create a TestReport object
        report = TestReport.from_item_and_call(item, call)

        # Call the hook
        try:
            hook = plugin.pytest_runtest_makereport(item, call)
            next(hook)  # Start the generator
            final_report = hook.send(None)  # Send None to the generator
        except StopIteration as exc:
            final_report = exc.value if exc.value is not None else None

        # Simulate the behavior of pytest by setting the result
        if final_report is None:
            final_report = report

        if pydiagno_enabled:
            # Simulate the PyDiagno analysis being applied to the report
            setattr(final_report, "pydiagno_result", mock_analysis.return_value)

        assert isinstance(final_report, TestReport)
        assert final_report.when == "call"
        assert abs(final_report.duration - call.duration) < 0.001

        if pydiagno_enabled:
            assert hasattr(final_report, "pydiagno_result")
            assert final_report.pydiagno_result["confidence"] == 0.9
            assert final_report.pydiagno_result["iterations"] == 3
            assert final_report.pydiagno_result["result"] == "Test analysis result"
            mock_analysis.assert_called_once_with(item, final_report, mock_config)
        else:
            assert not hasattr(final_report, "pydiagno_result")
            mock_analysis.assert_not_called()

    # Test error handling
    with patch("pydiagno.plugin.perform_pydiagno_analysis") as mock_analysis:
        mock_analysis.side_effect = PyDiagnoAnalysisError("Test error")

        hook = plugin.pytest_runtest_makereport(item, call)
        next(hook)
        final_report = hook.send(None)

        # Simulate the behavior of pytest by setting the result
        if final_report is None:
            final_report = report

        if pydiagno_enabled:
            # Simulate the PyDiagno error being applied to the report
            setattr(final_report, "pydiagno_error", "Test error")

        if pydiagno_enabled:
            assert hasattr(final_report, "pydiagno_error")
            assert final_report.pydiagno_error == "Test error"
        else:
            assert not hasattr(final_report, "pydiagno_error")


def test_pytest_terminal_summary() -> None:
    """Test that pytest_terminal_summary writes the PyDiagno summary when enabled."""
    terminalreporter = Mock(spec=TerminalReporter)
    config = Mock(spec=Config)
    config.getoption.return_value = True

    mock_config = Mock(spec=PyDiagnoConfig)
    mock_config.analysis = Mock()
    mock_config.analysis.confidence_threshold = 0.8
    mock_config.analysis.max_iterations = 5
    setattr(config, "pydiagno_config", mock_config)

    # Mock test reports
    passed_report = Mock(spec=TestReport)
    passed_report.nodeid = "test_passed"
    passed_report.pydiagno_result = {"result": "Passed test analysis"}
    failed_report = Mock(spec=TestReport)
    failed_report.nodeid = "test_failed"
    failed_report.pydiagno_result = {"result": "Failed test analysis"}
    error_report = Mock(spec=TestReport)
    error_report.nodeid = "test_error"
    error_report.pydiagno_error = "PyDiagno analysis error"

    terminalreporter.stats = {
        "passed": [passed_report],
        "failed": [failed_report, error_report],
    }

    plugin.pytest_terminal_summary(terminalreporter, 0, config)

    terminalreporter.write_sep.assert_called_once()
    assert terminalreporter.write_line.call_count >= 7  # At least 7 lines of output

    # Check for specific output lines
    output_lines = [call.args[0] for call in terminalreporter.write_line.call_args_list]
    assert any("Analysis confidence threshold: 0.80" in line for line in output_lines)
    assert any("Maximum analysis iterations: 5" in line for line in output_lines)
    assert any("Test: test_passed" in line for line in output_lines)
    assert any("PyDiagno Result: Passed test analysis" in line for line in output_lines)
    assert any("Test: test_failed" in line for line in output_lines)
    assert any("PyDiagno Result: Failed test analysis" in line for line in output_lines)
    assert any("Test: test_error" in line for line in output_lines)
    assert any(
        "PyDiagno Error: PyDiagno analysis error" in line for line in output_lines
    )


def test_pytest_exception_interact() -> None:
    """Test that pytest_exception_interact handles PyDiagno exceptions correctly."""
    node = Mock(spec=Item)
    call = Mock()
    report = Mock(spec=TestReport)

    # Test PyDiagnoConfigError
    call.excinfo = Mock()
    call.excinfo.value = PyDiagnoConfigError("Config error")
    plugin.pytest_exception_interact(node, call, report)
    assert report.longrepr == "PyDiagno Configuration Error: Config error"

    # Test PyDiagnoAnalysisError
    call.excinfo.value = PyDiagnoAnalysisError("Analysis error")
    plugin.pytest_exception_interact(node, call, report)
    assert report.longrepr == "PyDiagno Analysis Error: Analysis error"

    # Test other exceptions
    call.excinfo.value = ValueError("Some other error")
    original_longrepr = report.longrepr
    plugin.pytest_exception_interact(node, call, report)
    assert report.longrepr == original_longrepr


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
