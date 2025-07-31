#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
FASTAPI_PORT=8000
FASTAPI_HOST="0.0.0.0"
CELERY_WORKERS=1
DEV_MODE=true


# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT       FastAPI port (default: 8000)"
    echo "  -h, --host HOST       FastAPI host (default: 0.0.0.0)"
    echo "  -w, --workers NUM     Number of Celery workers (default: 1)"
    echo "  -d, --dev             Development mode (auto-reload)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    Start with default settings"
    echo "  $0 -p 8080 -w 2      Start on port 8080 with 2 workers"
    echo "  $0 --dev             Start in development mode"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            FASTAPI_PORT="$2"
            shift 2
            ;;
        -h|--host)
            FASTAPI_HOST="$2"
            shift 2
            ;;
        -w|--workers)
            CELERY_WORKERS="$2"
            shift 2
            ;;
        -d|--dev)
            DEV_MODE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 Starting Voicebot Application${NC}"
echo -e "${BLUE}📊 Configuration:${NC}"
echo -e "${BLUE}   - FastAPI: ${FASTAPI_HOST}:${FASTAPI_PORT}${NC}"
echo -e "${BLUE}   - Celery workers: ${CELERY_WORKERS}${NC}"
echo -e "${BLUE}   - Development mode: ${DEV_MODE}${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check virtual environment
    if [ ! -d "venv" ]; then
        echo -e "${RED}❌ Virtual environment not found${NC}"
        echo "Please create one first: python -m venv venv"
        exit 1
    fi
    
    # Check .env file
    if [ ! -f ".env" ]; then
        echo -e "${RED}❌ .env file not found${NC}"
        echo "Please create .env file from sample.env"
        exit 1
    fi
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        echo -e "${RED}❌ Redis not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Start Redis if not running
start_redis() {
    echo -e "${YELLOW}🔍 Checking Redis...${NC}"
    if ! redis-cli ping > /dev/null 2>&1; then
        echo -e "${YELLOW}🔄 Starting Redis...${NC}"
        if command -v systemctl &> /dev/null; then
            sudo systemctl start redis-server
        else
            redis-server --daemonize yes
        fi
        sleep 2
        
        if ! redis-cli ping > /dev/null 2>&1; then
            echo -e "${RED}❌ Failed to start Redis${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}✅ Redis is running${NC}"
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down...${NC}"
    
    if [ -f "celery.pid" ]; then
        CELERY_PID=$(cat celery.pid)
        if kill -0 $CELERY_PID 2>/dev/null; then
            echo -e "${YELLOW}🔄 Stopping Celery worker (PID: $CELERY_PID)...${NC}"
            kill $CELERY_PID
        fi
        rm -f celery.pid
    fi
    
    if [ -f "fastapi.pid" ]; then
        FASTAPI_PID=$(cat fastapi.pid)
        if kill -0 $FASTAPI_PID 2>/dev/null; then
            echo -e "${YELLOW}🔄 Stopping FastAPI server (PID: $FASTAPI_PID)...${NC}"
            kill $FASTAPI_PID
        fi
        rm -f fastapi.pid
    fi
    
    echo -e "${GREEN}✅ Shutdown complete${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run checks
check_prerequisites

# Activate virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source venv/bin/activate

# download nltk 'punkt' tokenizer
echo -e "${YELLOW}⬇️ Downloading NLTK 'punkt' tokenizer...${NC}"
python -c "import nltk; nltk.download('punkt')"


# Start Redis
start_redis

# Start Celery worker
# echo -e "${YELLOW}🔄 Starting Celery worker(s)...${NC}"
# celery -A backend.workers.celery_worker.celery_app worker \
#     --loglevel=info \
#     --concurrency=$CELERY_WORKERS \
#     --hostname=voicebot@%h &
# CELERY_PID=$!
# to prevent the issues caused be forking
echo -e "${YELLOW}🔄 Starting Celery worker(s)...${NC}"
celery -A backend.workers.celery_worker.celery_app worker \
    --loglevel=info \
    --concurrency=$CELERY_WORKERS \
    --pool=threads \
    --hostname=voicebot@%h &
CELERY_PID=$!

# Wait for Celery to start
sleep 3

# Start FastAPI server
echo -e "${YELLOW}🌐 Starting FastAPI server...${NC}"
if [ "$DEV_MODE" = true ]; then
    python -m uvicorn backend.main:app \
        --reload \
        --host $FASTAPI_HOST \
        --port $FASTAPI_PORT \
        --ws-ping-interval 30 \
        --ws-ping-timeout 30 \
        --timeout-keep-alive 300 &
else
    python -m uvicorn backend.main:app \
        --host $FASTAPI_HOST \
        --port $FASTAPI_PORT \
        --ws-ping-interval 30 \
        --ws-ping-timeout 30 \
        --timeout-keep-alive 300 &
fi
FASTAPI_PID=$!

mkdir -p tmp

# Save PIDs
echo "$CELERY_PID" > tmp/celery.pid
echo "$FASTAPI_PID" > tmp/fastapi.pid

echo ""
echo -e "${GREEN}✅ Application started successfully!${NC}"
echo -e "${BLUE}📍 FastAPI server: http://${FASTAPI_HOST}:${FASTAPI_PORT}${NC}"
echo -e "${BLUE}📍 API docs: http://${FASTAPI_HOST}:${FASTAPI_PORT}/docs${NC}"
echo -e "${BLUE}📍 Celery worker PID: $CELERY_PID${NC}"
echo -e "${BLUE}📍 FastAPI server PID: $FASTAPI_PID${NC}"
echo ""
echo -e "${YELLOW}💡 Press Ctrl+C to stop the application${NC}"

# Wait for processes
wait