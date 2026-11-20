"""Run an isolated loopback Web preview without reading project credentials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi.staticfiles import StaticFiles

from ecocompanion.api import create_app
from ecocompanion.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--local-model", default=None)
    parser.add_argument("--reuse-output", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    web_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
    if not (web_dist / "index.html").is_file():
        parser.error("build web/dist before starting the preview")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.reuse_output)
    configuration: dict[str, Any] = {
        "_env_file": None,
        "LITELLM_BASE_URL": "http://127.0.0.1:11434/v1",
        "LITELLM_API_KEY": "local-preview" if args.local_model else None,
        "ECO_DEFAULT_MODEL": args.local_model,
        "ECO_DEMO_FALLBACK_MODEL": None,
        "ECO_ARTIFACTS_DIR": output,
        "ECO_DATABASE_URL": f"sqlite+aiosqlite:///{(output / 'preview.db').as_posix()}",
        "ECO_PROVIDER_EXTRA_BODY": None,
        "ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION": None,
        "ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION": None,
        "ECO_MODEL_PRICING_VERSION": None,
        "ECO_RESEARCH_MODE": True,
        "max_same_model_retries": 0,
    }
    settings = Settings(**configuration)
    app = create_app(settings=settings)
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="preview")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
