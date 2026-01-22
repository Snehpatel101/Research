"""
Allow running the CLI as a module: python -m src.cli

This enables the CLI to be run with:
    python -m src.cli --help
    python -m src.cli run --symbol MES
    python -m src.cli train model --model xgboost
"""

from src.cli import main

if __name__ == "__main__":
    main()
