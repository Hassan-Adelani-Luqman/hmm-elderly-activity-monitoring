"""Redact the app-generated device id from Sensor Logger Metadata.csv files.

The repository is public, so the per-install UUID is replaced with "redacted"
before raw recordings are committed. All other metadata (device model, OS
version, configured sample rates) is kept because the report needs it.

Usage:
    python scripts/redact_metadata.py data/raw/pilot
"""

import csv
import sys
from pathlib import Path

FIELD = "device id"
REDACTED = "redacted"


def redact_file(path):
    """Redact the device id in one Metadata.csv. Returns True if the file changed."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if FIELD not in fieldnames or all(r[FIELD] == REDACTED for r in rows):
        return False
    for r in rows:
        r[FIELD] = REDACTED
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return True


def main(roots):
    for root in roots:
        for path in sorted(Path(root).rglob("Metadata.csv")):
            status = "redacted" if redact_file(path) else "unchanged"
            print(f"{status:9s} {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
