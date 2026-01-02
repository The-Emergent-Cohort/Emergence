#!/bin/bash
#
# Cron wrapper for Kaikki updates
#
# Add to crontab:
#   0 3 * * 1 /path/to/tokenizers/scripts/cron_update.sh
#
# This runs weekly on Monday at 3am, matching Kaikki's update schedule.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$BASE_DIR/venv"
LOG_DIR="$BASE_DIR/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Log file with timestamp
LOG_FILE="$LOG_DIR/cron_update_$(date +%Y%m%d_%H%M%S).log"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found at $VENV_DIR" >> "$LOG_FILE"
    exit 1
fi

# Run update with cleanup
echo "Starting Kaikki update at $(date)" >> "$LOG_FILE"
python "$SCRIPT_DIR/update_kaikki.py" --cleanup >> "$LOG_FILE" 2>&1
exit_code=$?

echo "Update completed at $(date) with exit code $exit_code" >> "$LOG_FILE"

# Cleanup old logs (keep 30 days)
find "$LOG_DIR" -name "cron_update_*.log" -mtime +30 -delete 2>/dev/null || true

# Deactivate
deactivate

exit $exit_code
