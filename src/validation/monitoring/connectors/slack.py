"""
Slack alert connector for drift and performance alerts.

Sends formatted alerts to Slack channels when critical model issues are detected.
Requires slack-sdk package: pip install slack-sdk

Example:
    import os
    from src.validation.monitoring.connectors import SlackAlertConnector, SlackConfig

    config = SlackConfig(
        token=os.getenv("SLACK_BOT_TOKEN"),
        channel="#model-alerts"
    )
    connector = SlackAlertConnector(config)
    connector.send_drift_alert("rsi_14", "HIGH", 0.35, 0.20)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SlackConfig:
    """
    Configuration for Slack alert connector.

    Attributes:
        token: Slack bot token (xoxb-...)
        channel: Slack channel to post alerts (#model-alerts)
        username: Bot display name in Slack
        icon_emoji: Bot icon emoji
    """

    token: str
    channel: str = "#model-alerts"
    username: str = "ML Factory Monitor"
    icon_emoji: str = ":robot_face:"


class SlackAlertConnector:
    """
    Send alerts to Slack channel.

    This connector formats and sends model monitoring alerts to Slack using
    the Slack Web API. It provides rich formatting with severity indicators
    and structured field layouts.

    Example:
        config = SlackConfig(token=os.getenv("SLACK_BOT_TOKEN"))
        connector = SlackAlertConnector(config)

        # Send drift alert
        connector.send_drift_alert(
            feature_name="volatility_20",
            severity="HIGH",
            metric_value=0.35,
            threshold=0.20
        )

        # Send performance alert
        connector.send_alert(
            alert_type="accuracy_degradation",
            severity="CRITICAL",
            message="Model accuracy dropped from 0.65 to 0.52"
        )
    """

    def __init__(self, config: SlackConfig) -> None:
        """
        Initialize Slack connector.

        Args:
            config: SlackConfig with token and channel settings

        Raises:
            ImportError: If slack-sdk is not installed
        """
        self.config = config

        try:
            from slack_sdk import WebClient

            self.client = WebClient(token=config.token)
        except ImportError as e:
            raise ImportError("slack-sdk not installed. Install with: pip install slack-sdk") from e

    def send_drift_alert(
        self,
        feature_name: str,
        severity: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        """
        Send drift detection alert to Slack.

        Args:
            feature_name: Name of feature with drift
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            metric_value: Drift metric value (PSI or KS statistic)
            threshold: Configured threshold
        """
        severity_emoji = {
            "LOW": ":large_yellow_circle:",
            "MEDIUM": ":large_orange_circle:",
            "HIGH": ":red_circle:",
            "CRITICAL": ":rotating_light:",
        }

        message = {
            "text": f"{severity_emoji.get(severity, ':warning:')} *DRIFT ALERT*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Feature drift detected: {feature_name}*",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                        {"type": "mrkdwn", "text": f"*PSI Metric:*\n{metric_value:.4f}"},
                        {"type": "mrkdwn", "text": f"*Threshold:*\n{threshold:.4f}"},
                    ],
                },
            ],
        }

        self._send_message(message)

    def send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
    ) -> None:
        """
        Send generic alert to Slack.

        Args:
            alert_type: Type of alert (model_freshness, accuracy_degradation, etc.)
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            message: Alert message
        """
        severity_emoji = {
            "LOW": ":large_yellow_circle:",
            "MEDIUM": ":large_orange_circle:",
            "HIGH": ":red_circle:",
            "CRITICAL": ":rotating_light:",
        }

        formatted_message = {
            "text": f"{severity_emoji.get(severity, ':warning:')} *{alert_type.upper()}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{alert_type.replace('_', ' ').title()}*\n{message}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Severity: *{severity}*",
                        }
                    ],
                },
            ],
        }

        self._send_message(formatted_message)

    def _send_message(self, message: dict[str, Any]) -> None:
        """
        Internal method to send message to Slack.

        Args:
            message: Formatted message dict with text and blocks
        """
        try:
            self.client.chat_postMessage(
                channel=self.config.channel,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji,
                **message,
            )
            logger.info(f"Sent alert to Slack channel {self.config.channel}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SlackAlertConnector",
    "SlackConfig",
]
