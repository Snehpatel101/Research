#!/usr/bin/env python3
"""
Model Serving CLI.

Start an HTTP server for real-time model inference.

Usage:
    # Start server with single model
    python scripts/serve_model.py --bundle ./bundles/xgb_h20

    # Start on specific port
    python scripts/serve_model.py --bundle ./bundles/xgb_h20 --port 8080

    # Ensemble serving
    python scripts/serve_model.py \
        --bundles ./bundles/xgb_h20,./bundles/lgbm_h20 \
        --port 8080

API Endpoints:
    GET  /health           - Health check
    GET  /info             - Model information
    POST /predict          - Make predictions
    POST /predict_ensemble - Ensemble predictions (if multiple models)
    GET  /metrics          - Server metrics

Example Request:
    curl -X POST http://localhost:8080/predict \
        -H "Content-Type: application/json" \
        -d '{"features": [[0.1, 0.2, 0.3, ...]]}'
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ModelServer, ServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start HTTP server for model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python scripts/serve_model.py --bundle ./bundles/xgb_h20

  # Ensemble
  python scripts/serve_model.py --bundles ./bundles/xgb_h20,./bundles/lgbm_h20

  # Custom port
  python scripts/serve_model.py --bundle ./bundles/xgb_h20 --port 9000

API Example:
  curl -X POST http://localhost:8080/predict \\
      -H "Content-Type: application/json" \\
      -d '{"features": [[0.1, 0.2, ...]]}'
        """,
    )

    # Model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--bundle",
        type=Path,
        help="Path to single model bundle",
    )
    model_group.add_argument(
        "--bundles",
        type=str,
        help="Comma-separated paths to multiple bundles",
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    # Options
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1000,
        help="Maximum batch size for requests (default: 1000)",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable /metrics endpoint",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load bundles
    if args.bundle:
        bundle_paths = [args.bundle]
    else:
        bundle_paths = [Path(p.strip()) for p in args.bundles.split(",")]

    for path in bundle_paths:
        if not path.exists():
            logger.error(f"Bundle not found: {path}")
            return 1

    # Create server config
    config = ServerConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        max_batch_size=args.max_batch_size,
        enable_metrics=not args.no_metrics,
    )

    # Create server
    logger.info(f"Loading {len(bundle_paths)} bundle(s)...")

    try:
        if len(bundle_paths) == 1:
            server = ModelServer.from_bundle(bundle_paths[0], config)
        else:
            server = ModelServer.from_bundles(bundle_paths, config)
    except ImportError as e:
        logger.error(str(e))
        logger.error("Please install Flask: pip install flask")
        return 1

    # Log model info
    for info in server.pipeline.get_model_info():
        logger.info(f"  Model: {info['name']} (H{info['horizon']}, {info['features']} features)")

    # Print API info
    print("\n" + "=" * 60)
    print("MODEL SERVER")
    print("=" * 60)
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Models: {server.pipeline.model_names}")
    print(f"Horizon: {server.pipeline.horizon}")
    print("\nEndpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /info             - Model information")
    print("  POST /predict          - Make predictions")
    if len(bundle_paths) > 1:
        print("  POST /predict_ensemble - Ensemble predictions")
    if not args.no_metrics:
        print("  GET  /metrics          - Server metrics")
    print("=" * 60 + "\n")

    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
