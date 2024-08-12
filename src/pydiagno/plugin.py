from typing import Generator, Optional

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter

from pydiagno.config import PyDiagnoConfig, load_config

def pytest_addoption(parser: Parser) -> None:
    """Add PyDiagno-specific command line options to pytest."""
    group = parser.getgroup("pydiagno")
    group.addoption(
        "--pydiagno",
        action="store_true",
        dest="pydiagno",
        default=False,
        help="Enable PyDiagno analysis",
    )
    group.addoption(
        "--pydiagno-config",
        action="store",
        dest="pydiagno_config",
        default=None,
        help="Path to custom PyDiagno configuration file",
    )
    group.addoption(
        "--pydiagno-log-level",
        action="store",
        dest="pydiagno_log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set the log level for PyDiagno",
    )
    group.addoption(
        "--pydiagno-confidence-threshold",
        action="store",
        dest="pydiagno_confidence_threshold",
        type=float,
        default=None,
        help="Set the confidence threshold for PyDiagno analysis",
    )
    group.addoption(
        "--pydiagno-max-iterations",
        action="store",
        dest="pydiagno_max_iterations",
        type=int,
        default=None,
        help="Set the maximum number of iterations for PyDiagno analysis",
    )
    group.addoption(
        "--pydiagno-rag-enabled",
        action="store_true",
        dest="pydiagno_rag_enabled",
        default=None,
        help="Enable Retrieval-Augmented Generation (RAG) feature",
    )
    group.addoption(
        "--pydiagno-report-format",
        action="store",
        dest="pydiagno_report_format",
        choices=["json", "yaml", "text"],
        default=None,
        help="Set the format for PyDiagno reports",
    )
    group.addoption(
        "--pydiagno-report-output",
        action="store",
        dest="pydiagno_report_output",
        default=None,
        help="Set the output path for PyDiagno reports",
    )


def pytest_configure(config: Config) -> None:
    """Configure PyDiagno plugin."""
    config.addinivalue_line("markers", "pydiagno: mark test for PyDiagno analysis")
    if config.getoption("pydiagno"):
        config_path = config.getoption("pydiagno_config")
        pydiagno_config = load_config(config_path) if config_path else load_config()

        # Override configuration with command line options
        if config.getoption("pydiagno_log_level"):
            pydiagno_config.monitoring.log_level = config.getoption(
                "pydiagno_log_level")
        if config.getoption("pydiagno_confidence_threshold") is not None:
            pydiagno_config.analysis.confidence_threshold = config.getoption(
                "pydiagno_confidence_threshold")
        if config.getoption("pydiagno_max_iterations") is not None:
            pydiagno_config.analysis.max_iterations = config.getoption(
                "pydiagno_max_iterations")
        if config.getoption("pydiagno_rag_enabled") is not None:
            pydiagno_config.rag.enabled = config.getoption("pydiagno_rag_enabled")
        if config.getoption("pydiagno_report_format"):
            pydiagno_config.reporting.format = config.getoption(
                "pydiagno_report_format")
        if config.getoption("pydiagno_report_output"):
            pydiagno_config.reporting.output_path = config.getoption(
                "pydiagno_report_output")

        config.pydiagno_config = pydiagno_config
        # TODO: Initialize PyDiagno here if needed
    return None


def perform_pydiagno_analysis(item: Item, report: TestReport, config: PyDiagnoConfig) \
        -> Optional[dict]:
    """
    Perform PyDiagno analysis on a test item.

    Args:
        item: Test item being analyzed.
        report: Test report for the item.
        config: PyDiagno configuration.

    Returns:
        Optional[dict]: Analysis result or None if analysis couldn't be performed.
    """
    # TODO: Implement actual PyDiagno analysis logic here
    # This is a placeholder implementation
    analysis_result = {
        "confidence": config.analysis.confidence_threshold,
        "iterations": config.analysis.max_iterations,
        "result": "Placeholder analysis result"
    }
    return analysis_result


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(
    item: Item, call: pytest.CallInfo[None]
) -> Generator[None, None, TestReport]:
    """
    Extend test reports with PyDiagno analysis results.

    Args:
        item: Test item being executed.
        call: Result of test execution.

    Yields:
        None
    """
    outcome = yield
    if outcome is not None and hasattr(outcome, "get_result"):
        report = outcome.get_result()
    else:
        report = TestReport.from_item_and_call(item, call)

    # Check if PyDiagno is enabled globally or for this specific test
    pydiagno_enabled = item.config.getoption("pydiagno") or item.get_closest_marker(
        "pydiagno")

    if isinstance(report, TestReport) and report.when == "call"  and pydiagno_enabled:
        if pydiagno_enabled:
            # Retrieve PyDiagno configuration
            pydiagno_config = getattr(item.config, 'pydiagno_config', None)

            if pydiagno_config:
                # Perform PyDiagno analysis
                analysis_result = perform_pydiagno_analysis(item, report,
                                                            pydiagno_config)

                # Attach analysis result to the report
                report.pydiagno_result = analysis_result
            else:
                # Log a warning if PyDiagno is enabled but configuration is missing
                item.warn(pytest.PytestWarning(
                    "PyDiagno is enabled, but configuration is missing."))

    return report


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: int, config: Config
) -> None:
    """
    Add PyDiagno analysis summary to pytest output.

    Args:
        terminalreporter: Terminal reporter object.
        exitstatus: Exit status of pytest run.
        config: Pytest configuration object.
    """
    if config.getoption("pydiagno"):
        # TODO: Here we'll add the summary of PyDiagno analysis in the future
        pydiagno_config = getattr(config, 'pydiagno_config', None)
        if pydiagno_config:
            terminalreporter.write_sep("-", "PyDiagno Analysis Summary")
            terminalreporter.write_line(
                f"Analysis confidence threshold: "
                f"{config.analysis.confidence_threshold:.2f}")
            terminalreporter.write_line(
                f"Maximum analysis iterations: "
                f"{config.analysis.max_iterations:d}")

            for report in terminalreporter.stats.get('passed',
                                                     []) + terminalreporter.stats.get(
                    'failed', []):
                if hasattr(report, 'pydiagno_result'):
                    terminalreporter.write_line(f"Test: {report.nodeid}")
                    terminalreporter.write_line(
                        f"PyDiagno Result: {report.pydiagno_result['result']}")
                    terminalreporter.write_line("")
        else:
            terminalreporter.write_line("PyDiagno configuration not found.")
        # TODO: Add more detailed summary based on the new configuration options
