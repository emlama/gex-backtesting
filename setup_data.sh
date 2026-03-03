#!/bin/bash
# Setup script for GEX Backtesting data
#
# This script creates symlinks from the Hermes repo's data directory
# to this project's data/ directory.
#
# Usage:
#   ./setup_data.sh /path/to/hermes/backtesting/research/gci_meta_analysis/data
#
# Or if you have the Hermes repo checked out at the default location:
#   ./setup_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

# Default source path (adjust to your Hermes repo location)
DEFAULT_SOURCE="$HOME/Documents/GitHub/Hermes/backtesting/research/gci_meta_analysis/data"

SOURCE="${1:-$DEFAULT_SOURCE}"

if [ ! -d "$SOURCE" ]; then
    echo "Source directory not found: $SOURCE"
    echo ""
    echo "Usage: $0 /path/to/parquet/directory"
    echo ""
    echo "The directory should contain files like:"
    echo "  trades_2024-01-02.parquet"
    echo "  trades_2024-01-03.parquet"
    echo "  ..."
    exit 1
fi

# Count source files
SOURCE_COUNT=$(ls "$SOURCE"/trades_*.parquet 2>/dev/null | wc -l | tr -d ' ')
echo "Source: $SOURCE ($SOURCE_COUNT parquet files)"

# Create symlinks
mkdir -p "$DATA_DIR"

LINKED=0
for f in "$SOURCE"/trades_*.parquet; do
    fname=$(basename "$f")
    if [ ! -e "$DATA_DIR/$fname" ]; then
        ln -s "$f" "$DATA_DIR/$fname"
        LINKED=$((LINKED + 1))
    fi
done

# Final count
TOTAL=$(ls "$DATA_DIR"/trades_*.parquet 2>/dev/null | wc -l | tr -d ' ')
echo "Linked $LINKED new files ($TOTAL total in data/)"
echo ""
echo "Verify with: ls data/trades_*.parquet | head -5"
