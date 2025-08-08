#!/bin/bash

# Multi-Timeframe Crypto Signal Bot Management Script
# Version: 2.0.0

BOT_SCRIPT="final_bot.py"
PORT=5004
LOG_FILE="multi_timeframe_bot.log"
PID_FILE="bot.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Multi-Timeframe Crypto Signal Bot Manager${NC}"
echo "=================================================="

case "$1" in
    start)
        echo -e "${GREEN}Starting Multi-Timeframe Crypto Signal Bot...${NC}"
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${YELLOW}Bot is already running (PID: $PID)${NC}"
                exit 1
            else
                echo -e "${YELLOW}Removing stale PID file${NC}"
                rm -f "$PID_FILE"
            fi
        fi
        
        # Start the bot in background
        nohup python3 "$BOT_SCRIPT" > "$LOG_FILE" 2>&1 &
        BOT_PID=$!
        echo $BOT_PID > "$PID_FILE"
        
        echo -e "${GREEN}Bot started successfully (PID: $BOT_PID)${NC}"
        echo -e "${BLUE}Log file: $LOG_FILE${NC}"
        echo -e "${BLUE}Dashboard: http://localhost:$PORT${NC}"
        echo -e "${BLUE}API Health: http://localhost:$PORT/health${NC}"
        
        # Wait a moment and check if it's running
        sleep 3
        if ps -p $BOT_PID > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Bot is running and healthy${NC}"
        else
            echo -e "${RED}❌ Bot failed to start. Check logs: $LOG_FILE${NC}"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;
        
    stop)
        echo -e "${YELLOW}Stopping Multi-Timeframe Crypto Signal Bot...${NC}"
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                kill $PID
                echo -e "${GREEN}Bot stopped (PID: $PID)${NC}"
            else
                echo -e "${YELLOW}Bot was not running${NC}"
            fi
            rm -f "$PID_FILE"
        else
            echo -e "${YELLOW}No PID file found${NC}"
        fi
        ;;
        
    restart)
        echo -e "${BLUE}Restarting Multi-Timeframe Crypto Signal Bot...${NC}"
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo -e "${BLUE}Multi-Timeframe Crypto Signal Bot Status${NC}"
        echo "=============================================="
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${GREEN}✅ Bot is running (PID: $PID)${NC}"
                
                # Check API health
                if command -v curl > /dev/null 2>&1; then
                    echo -e "${BLUE}Checking API health...${NC}"
                    HEALTH_RESPONSE=$(curl -s "http://localhost:$PORT/health" 2>/dev/null)
                    if [ $? -eq 0 ]; then
                        echo -e "${GREEN}✅ API is responding${NC}"
                        echo -e "${BLUE}Dashboard: http://localhost:$PORT${NC}"
                    else
                        echo -e "${RED}❌ API is not responding${NC}"
                    fi
                fi
                
                # Show recent logs
                if [ -f "$LOG_FILE" ]; then
                    echo -e "${BLUE}Recent logs:${NC}"
                    tail -n 10 "$LOG_FILE"
                fi
            else
                echo -e "${RED}❌ Bot is not running (stale PID file)${NC}"
                rm -f "$PID_FILE"
            fi
        else
            echo -e "${RED}❌ Bot is not running${NC}"
        fi
        ;;
        
    logs)
        echo -e "${BLUE}Showing Multi-Timeframe Bot logs:${NC}"
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        fi
        ;;
        
    test-all)
        echo -e "${BLUE}Running comprehensive test suite...${NC}"
        python3 master_test.py
        ;;
        
    test-signal)
        echo -e "${BLUE}Generating test signal...${NC}"
        if command -v curl > /dev/null 2>&1; then
            curl -X POST "http://localhost:$PORT/test_signal" -H "Content-Type: application/json"
            echo ""
        else
            echo -e "${RED}curl not available. Please install curl or manually test via browser.${NC}"
        fi
        ;;
        
    *)
        echo -e "${BLUE}Usage: $0 {start|stop|restart|status|logs|test-all|test-signal}${NC}"
        echo ""
        echo -e "${BLUE}Commands:${NC}"
        echo -e "  ${GREEN}start${NC}      - Start the multi-timeframe bot"
        echo -e "  ${GREEN}stop${NC}       - Stop the bot"
        echo -e "  ${GREEN}restart${NC}    - Restart the bot"
        echo -e "  ${GREEN}status${NC}     - Show bot status and health"
        echo -e "  ${GREEN}logs${NC}       - Show real-time logs"
        echo -e "  ${GREEN}test-all${NC}   - Run comprehensive test suite"
        echo -e "  ${GREEN}test-signal${NC} - Generate a test signal"
        echo ""
        echo -e "${BLUE}Multi-Timeframe Features:${NC}"
        echo -e "  • ${YELLOW}5-minute${NC} scanning loop"
        echo -e "  • ${YELLOW}15-minute${NC} scanning loop" 
        echo -e "  • ${YELLOW}1-hour${NC} scanning loop"
        echo -e "  • ${YELLOW}Stage 1${NC}: Top-30% liquidity filter"
        echo -e "  • ${YELLOW}Stage 2${NC}: Dual-indicator confluence"
        echo -e "  • ${YELLOW}Real-time${NC} Discord notifications"
        echo -e "  • ${YELLOW}Live dashboard${NC} at http://localhost:$PORT"
        exit 1
        ;;
esac 