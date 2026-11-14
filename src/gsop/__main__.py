"""``python -m gsop`` is the same as the ``gsop`` command."""

from gsop.cli import app

if __name__ == "__main__":  # spawn-safe: workers re-import this module without running the CLI
    app()
