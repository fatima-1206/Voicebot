#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 Voicebot Application Status${NC}"
echo ""

# Check Redis
echo -e "${BLUE}🔍 Redis Status:${NC}"
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis is not running${NC}"
fi

# Check Celery worker
echo -e "${BLUE}🔍 Celery Worker Status:${NC}"
if [ -f "celery.pid" ]; then
    CELERY_PID=$(cat celery.pid)
    if kill -0 $CELERY_PID 2>/dev/null; then
        echo -e "${GREEN}✅ Celery worker is running (PID: $CELERY_PID)${NC}"
    else
        echo -e "${RED}❌ Celery worker is not running (stale PID file)${NC}"
        rm -f celery.pid
    fi
else
    echo -e "${YELLOW}⚠️ No Celery PID file found${NC}"
fi

# Check FastAPI server
echo -e "${BLUE}🔍 FastAPI Server Status:${NC}"
if [ -f "fastapi.pid" ]; then
    FASTAPI_PID=$(cat fastapi.pid)
    if kill -0 $FASTAPI_PID 2>/dev/null; then
        echo -e "${GREEN}✅ FastAPI server is running (PID: $FASTAPI_PID)${NC}"
        # Test HTTP endpoint
        if curl -s http://localhost:8000/ > /dev/null; then
            echo -e "${GREEN}✅ HTTP endpoint is responding${NC}"
        else
            echo -e "${YELLOW}⚠️ HTTP endpoint is not responding${NC}"
        fi
    else
        echo -e "${RED}❌ FastAPI server is not running (stale PID file)${NC}"
        rm -f fastapi.pid
    fi
else
    echo -e "${YELLOW}⚠️ No FastAPI PID file found${NC}"
fi

echo ""
echo -e "${BLUE}📍 URLs:${NC}"
echo -e "${BLUE}   - API: http://localhost:8000${NC}"
echo -e "${BLUE}   - Docs: http://localhost:8000/docs${NC}"