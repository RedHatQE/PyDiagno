class PyDiagnoError(Exception):
    """Base exception class for PyDiagno."""


class PyDiagnoConfigError(PyDiagnoError):
    """Raised when there's an error in PyDiagno configuration."""


class PyDiagnoAnalysisError(PyDiagnoError):
    """Raised when there's an error during PyDiagno analysis."""


class PyDiagnoRuntimeError(PyDiagnoError):
    """Raised when there's a runtime error in PyDiagno."""


class PyDiagnoValueError(PyDiagnoError):
    """Raised when there's an invalid value in PyDiagno operations."""


class PyDiagnoNotImplementedError(PyDiagnoError):
    """Raised when a requested feature is not implemented."""
