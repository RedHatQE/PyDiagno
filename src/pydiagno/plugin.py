from typing import Generator

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter


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


def pytest_configure(config: Config) -> None:
    """Configure PyDiagno plugin."""
    config.addinivalue_line("markers", "pydiagno: mark test for PyDiagno analysis")
    if config.getoption("pydiagno"):
        # TODO: Initialize PyDiagno here if needed
        pass
    return None


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

    if isinstance(report, TestReport) and report.when == "call":
        marker = item.get_closest_marker("pydiagno")
        if marker:
            # TODO: Add logic for PyDiagno analysis here in the future
            pass

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
        terminalreporter.write_sep("-", "PyDiagno Analysis Summary")
        terminalreporter.write_line("PyDiagno analysis summary will be added here.")
