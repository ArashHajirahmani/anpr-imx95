"""Allow `python -m anpr`."""

import sys

from anpr.cli import main

if __name__ == "__main__":
    sys.exit(main())
