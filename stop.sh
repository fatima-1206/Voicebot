#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Stopping Voicebot Application...${NC}"

# Stop Celery worker
if [ -f "celery.pid" ]; then
    CELERY_PID=$(cat celery.pid)
    if kill -0 $CELERY_PID 2>/dev/null; then
        echo -e "${YELLOW}🔄 Stopping Celery worker (PID: $CELERY_PID)...${NC}"
        kill $CELERY_PID
        sleep 2
        # Force kill if still running
        if kill -0 $CELERY_PID 2>/dev/null; then
            kill -9 $CELERY_PID
        fi
    fi
    rm -f celery.pid
    echo -e "${GREEN}✅ Celery worker stopped${NC}"
else
    echo -e "${YELLOW}⚠️ No Celery PID file found${NC}"
fi

# Stop FastAPI server
if [ -f "fastapi.pid" ]; then
    FASTAPI_PID=$(cat fastapi.pid)
    if kill -0 $FASTAPI_PID 2>/dev/null; then
        echo -e "${YELLOW}🔄 Stopping FastAPI server (PID: $FASTAPI_PID)...${NC}"
        kill $FASTAPI_PID
        sleep 2
        # Force kill if still running
        if kill -0 $FASTAPI_PID 2>/dev/null; then
            kill -9 $FASTAPI_PID
        fi
    fi
    rm -f fastapi.pid
    echo -e "${GREEN}✅ FastAPI server stopped${NC}"
else
    echo -e "${YELLOW}⚠️ No FastAPI PID file found${NC}"
fi

# Clean up any remaining processes
pkill -f "celery.*voicebot" 2>/dev/null
pkill -f "uvicorn.*backend.main" 2>/dev/null

echo -e "${GREEN}✅ Application stopped successfully${NC}"