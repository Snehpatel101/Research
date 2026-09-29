"""
Allow running the CLI as a module: python -m src.cli

This enables the CLI to be run with:
    python -m src.cli --help
    python -m src.cli run -d data.parquet --symbol MES
    python -m src.cli models
"""

from src.cli import main

if __name__ == "__main__":
    main()
