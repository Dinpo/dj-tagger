"""Kept for older instructions: same as `djtagger verify <folder>`."""

import sys

from djtagger.cli import app

if __name__ == "__main__":
    sys.argv = ["djtagger", "verify", *sys.argv[1:]]
    app()
