"""Run the local API and production UI on a loopback interface."""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "yt_clipper.webapi.app:app",
        host="127.0.0.1",
        port=int(os.getenv("CLIPPER_WEB_PORT", "8787")),
        reload=False,
    )
