#!/bin/bash
# Download SPX 0DTE trade data for GEX backtesting
#
# Downloads 513 daily parquet files (2024-01-02 to 2026-02-19)
# from the data server and extracts them into data/
#
# Usage:
#   ./download_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
URL="http://45.55.51.49/data/gex-spx-0dte-trades.tar.gz"
ARCHIVE="/tmp/gex-spx-0dte-trades.tar.gz"

# Check if data already exists
EXISTING=$(ls "$DATA_DIR"/trades_*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [ "$EXISTING" -gt 0 ]; then
    echo "Found $EXISTING existing parquet files in data/"
    read -p "Re-download and overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipped."
        exit 0
    fi
fi

echo "Downloading SPX 0DTE trade data (2.3 GB)..."
echo "Source: $URL"
echo ""

# Download with progress
if command -v curl &> /dev/null; then
    curl -L --progress-bar -o "$ARCHIVE" "$URL"
elif command -v wget &> /dev/null; then
    wget --show-progress -O "$ARCHIVE" "$URL"
else
    echo "Error: curl or wget required"
    exit 1
fi

echo ""
echo "Extracting to data/..."
mkdir -p "$DATA_DIR"
tar xzf "$ARCHIVE" -C "$DATA_DIR"
rm -f "$ARCHIVE"

# Count files
TOTAL=$(ls "$DATA_DIR"/trades_*.parquet 2>/dev/null | wc -l | tr -d ' ')
FIRST=$(ls "$DATA_DIR"/trades_*.parquet | sort | head -1 | xargs basename | sed 's/trades_//;s/.parquet//')
LAST=$(ls "$DATA_DIR"/trades_*.parquet | sort | tail -1 | xargs basename | sed 's/trades_//;s/.parquet//')

echo ""
echo "Done! $TOTAL trading days ($FIRST to $LAST)"
echo "Verify: ls data/trades_*.parquet | wc -l"
