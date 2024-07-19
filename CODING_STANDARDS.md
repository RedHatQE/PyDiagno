# PyDiagno Coding Standards

## Table of Contents
1. [Introduction](#introduction)
2. [Python Version](#python-version)
3. [Code Formatting](#code-formatting)
4. [Imports](#imports)
5. [Naming Conventions](#naming-conventions)
6. [Type Annotations](#type-annotations)
7. [Docstrings](#docstrings)
8. [Comments](#comments)
9. [Exception Handling](#exception-handling)
10. [Testing](#testing)
11. [Version Control](#version-control)

## Introduction

This document outlines the coding standards for the PyDiagno project. All contributors are expected to adhere to these guidelines to maintain consistency and readability across the codebase.

## Python Version

PyDiagno uses Python 3.9 or higher. Ensure your code is compatible with this version.

## Code Formatting

- Use [Black](https://github.com/psf/black) for automatic code formatting.
- Set maximum line length to 88 characters.
- Use 4 spaces for indentation (no tabs).

## Imports

- Sort imports using [isort](https://pycqa.github.io/isort/).
- Group imports in the following order:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
- Use absolute imports when possible.

## Naming Conventions

- Use `snake_case` for function and variable names.
- Use `PascalCase` for class names.
- Use `UPPER_CASE` for constants.
- Prefix private attributes and methods with a single underscore.

## Type Annotations

- Use type annotations for function arguments and return values.
- Use `Optional[Type]` for arguments that can be None.
- For complex types, use type aliases to improve readability.

Example:
```python
from typing import List, Optional

def process_data(data: List[str], limit: Optional[int] = None) -> int:
    # Function implementation
```

## Docstrings

Use Google Style docstrings for all public modules, functions, classes, and methods.

Example:

```python
def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetches data from the specified URL.

    Args:
        url (str): The URL to fetch data from.
        timeout (int, optional): The timeout in seconds. Defaults to 30.

    Returns:
        dict: The fetched data as a dictionary.

    Raises:
        RequestError: If the request fails.
    """
    # Function implementation
```

## Comments

- Use comments sparingly, preferring self-explanatory code.
- Write comments in complete sentences.
- Use inline comments only for complex logic that isn't immediately clear.

## Exception Handling

- Use specific exception types instead of bare `except` clauses.
- Use context managers (`with` statements) for resource management.

## Testing

- Write unit tests for all new functionality.
- Use pytest as the testing framework.
- Aim for at least 80% code coverage.

## Version Control

- Use Git for version control.
- Write clear, concise commit messages.
- Use feature branches for new features or bug fixes.
- Submit changes via pull requests for review.


