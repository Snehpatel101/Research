"""
CLI Commands Package.

This package contains all CLI subcommands organized by domain:
- pipeline: run, data, status, resume
- train: train, train-ensemble
- evaluate: cv, walk-forward, cpcv-pbo
"""

from .evaluate import evaluate_app
from .pipeline import pipeline_app
from .train import train_app

__all__ = ["pipeline_app", "train_app", "evaluate_app"]
