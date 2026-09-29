#!/usr/bin/env python3
"""
ML Factory CLI entry point (``ensemble-pipeline`` console script).

Delegates to ``src.cli`` (``python -m src.cli --help``).
"""

from src.cli import main

if __name__ == "__main__":
    main()
