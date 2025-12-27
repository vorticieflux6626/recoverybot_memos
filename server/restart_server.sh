#!/bin/bash
# Restart memOS Server
# Usage: ./restart_server.sh [--debug]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 Restarting memOS Server..."
echo ""

# Stop server
./stop_server.sh

# Brief pause
sleep 1

# Start server with any passed arguments
./start_server.sh "$@"
