"""Local web API adapter for the ClipForge UI."""

from dotenv import load_dotenv

load_dotenv()

from .app import create_app

__all__ = ["create_app"]
