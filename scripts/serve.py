"""CLI entry point for launching the FastAPI server."""

from __future__ import annotations

import argparse

import uvicorn

from fata_cognita.config import load_config


def main() -> None:
    """Start the FastAPI server."""
    parser = argparse.ArgumentParser(description="Serve the Fata Cognita API")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    uvicorn.run(
        "fata_cognita.api.main:create_app",
        factory=True,
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
