"""A small desktop studio for collecting keyword spotting audio."""

__version__ = "0.1.0"


def main() -> None:
    from .app import main as run

    run()
