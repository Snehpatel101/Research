"""
Alert connectors for production monitoring.
"""

from .slack import SlackAlertConnector, SlackConfig

__all__ = [
    "SlackAlertConnector",
    "SlackConfig",
]
