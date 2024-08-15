import pytest

from pydiagno.exceptions import (
    PyDiagnoAnalysisError,
    PyDiagnoConfigError,
    PyDiagnoError,
    PyDiagnoNotImplementedError,
    PyDiagnoRuntimeError,
    PyDiagnoValueError,
)


def test_pydiagno_error() -> None:
    """Test that PyDiagnoError can be raised with a message."""
    with pytest.raises(PyDiagnoError) as excinfo:
        raise PyDiagnoError("Test error message")
    assert str(excinfo.value) == "Test error message"


def test_pydiagno_config_error() -> None:
    """Test that PyDiagnoConfigError can be raised with a message."""
    with pytest.raises(PyDiagnoConfigError) as excinfo:
        raise PyDiagnoConfigError("Configuration error")
    assert str(excinfo.value) == "Configuration error"
    assert isinstance(excinfo.value, PyDiagnoError)


def test_pydiagno_analysis_error() -> None:
    """Test that PyDiagnoAnalysisError can be raised with a message."""
    with pytest.raises(PyDiagnoAnalysisError) as excinfo:
        raise PyDiagnoAnalysisError("Analysis failed")
    assert str(excinfo.value) == "Analysis failed"
    assert isinstance(excinfo.value, PyDiagnoError)


def test_pydiagno_runtime_error() -> None:
    """Test that PyDiagnoRuntimeError can be raised with a message."""
    with pytest.raises(PyDiagnoRuntimeError) as excinfo:
        raise PyDiagnoRuntimeError("Runtime error occurred")
    assert str(excinfo.value) == "Runtime error occurred"
    assert isinstance(excinfo.value, PyDiagnoError)


def test_pydiagno_value_error() -> None:
    """Test that PyDiagnoValueError can be raised with a message."""
    with pytest.raises(PyDiagnoValueError) as excinfo:
        raise PyDiagnoValueError("Invalid value")
    assert str(excinfo.value) == "Invalid value"
    assert isinstance(excinfo.value, PyDiagnoError)


def test_pydiagno_not_implemented_error() -> None:
    """Test that PyDiagnoNotImplementedError can be raised with a message."""
    with pytest.raises(PyDiagnoNotImplementedError) as excinfo:
        raise PyDiagnoNotImplementedError("Feature not implemented")
    assert str(excinfo.value) == "Feature not implemented"
    assert isinstance(excinfo.value, PyDiagnoError)


def test_exception_hierarchy() -> None:
    """Test that all PyDiagno exceptions inherit from PyDiagnoError."""
    assert issubclass(PyDiagnoConfigError, PyDiagnoError)
    assert issubclass(PyDiagnoAnalysisError, PyDiagnoError)
    assert issubclass(PyDiagnoRuntimeError, PyDiagnoError)
    assert issubclass(PyDiagnoValueError, PyDiagnoError)
    assert issubclass(PyDiagnoNotImplementedError, PyDiagnoError)
