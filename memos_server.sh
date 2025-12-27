#!/bin/bash

# memOS Server Management Script
# Controls the memOS server for Recovery Bot

# Configuration
SERVER_DIR="/home/sparkone/sdd/Recovery_Bot/memOS/server"
VENV_PATH="$SERVER_DIR/venv"
PID_FILE="$SERVER_DIR/memos_server.pid"
LOG_FILE="$SERVER_DIR/server.log"
HOST="0.0.0.0"
PORT="8001"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
        echo "Please create it first with: python -m venv $VENV_PATH"
        exit 1
    fi
}

# Get server PID if running
get_pid() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "$PID"
        else
            rm -f "$PID_FILE"
            echo ""
        fi
    else
        # Check if server is running without PID file
        PID=$(ps aux | grep "uvicorn main:app" | grep -v grep | awk '{print $2}' | head -n 1)
        echo "$PID"
    fi
}

# Start the server
start_server() {
    echo -e "${GREEN}Starting memOS Server...${NC}"
    
    check_venv
    
    # Check if already running
    PID=$(get_pid)
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}Server is already running with PID: $PID${NC}"
        return 0
    fi
    
    # Change to server directory
    cd "$SERVER_DIR" || exit 1
    
    # Activate virtual environment and start server
    source "$VENV_PATH/bin/activate"
    
    # Start server in background
    nohup python -m uvicorn main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level info \
        > "$LOG_FILE" 2>&1 &
    
    # Save PID
    echo $! > "$PID_FILE"
    
    # Wait a moment to check if server started successfully
    sleep 2
    
    PID=$(get_pid)
    if [ -n "$PID" ]; then
        echo -e "${GREEN}✓ memOS Server started successfully${NC}"
        echo -e "  PID: $PID"
        echo -e "  URL: http://$HOST:$PORT"
        echo -e "  Logs: tail -f $LOG_FILE"
    else
        echo -e "${RED}✗ Failed to start memOS Server${NC}"
        echo "Check logs at: $LOG_FILE"
        exit 1
    fi
}

# Stop the server
stop_server() {
    echo -e "${YELLOW}Stopping memOS Server...${NC}"
    
    PID=$(get_pid)
    if [ -z "$PID" ]; then
        echo -e "${YELLOW}Server is not running${NC}"
        return 0
    fi
    
    # Send SIGTERM for graceful shutdown
    kill -TERM "$PID" 2>/dev/null
    
    # Wait for process to stop (max 10 seconds)
    COUNTER=0
    while [ $COUNTER -lt 10 ]; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 1
        COUNTER=$((COUNTER + 1))
    done
    
    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Force stopping server...${NC}"
        kill -9 "$PID" 2>/dev/null
    fi
    
    # Remove PID file
    rm -f "$PID_FILE"
    
    echo -e "${GREEN}✓ memOS Server stopped${NC}"
}

# Restart the server
restart_server() {
    echo -e "${YELLOW}Restarting memOS Server...${NC}"
    stop_server
    sleep 2
    start_server
}

# Show server status
status_server() {
    PID=$(get_pid)
    
    if [ -n "$PID" ]; then
        echo -e "${GREEN}✓ memOS Server is running${NC}"
        echo -e "  PID: $PID"
        echo -e "  URL: http://$HOST:$PORT"
        
        # Check health endpoint
        if command -v curl &> /dev/null; then
            echo -e "\n${YELLOW}Health Check:${NC}"
            HEALTH=$(curl -s "http://localhost:$PORT/health" 2>/dev/null)
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Server is healthy${NC}"
                if command -v jq &> /dev/null; then
                    echo "$HEALTH" | jq -r '.services | to_entries[] | "  \(.key): \(.value.status)"'
                fi
            else
                echo -e "${RED}✗ Health check failed${NC}"
            fi
        fi
        
        # Show recent logs
        echo -e "\n${YELLOW}Recent logs:${NC}"
        if [ -f "$LOG_FILE" ]; then
            tail -n 5 "$LOG_FILE"
        fi
    else
        echo -e "${RED}✗ memOS Server is not running${NC}"
    fi
}

# Show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}Following memOS Server logs (Ctrl+C to exit)...${NC}"
        tail -f "$LOG_FILE"
    else
        echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        exit 1
    fi
}

# Main script logic
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "memOS Server Management Script"
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the memOS server"
        echo "  stop    - Stop the memOS server"
        echo "  restart - Restart the memOS server"
        echo "  status  - Show server status and health"
        echo "  logs    - Follow server logs"
        echo ""
        echo "Configuration:"
        echo "  Server Directory: $SERVER_DIR"
        echo "  Host: $HOST"
        echo "  Port: $PORT"
        exit 1
        ;;
esac

exit 0