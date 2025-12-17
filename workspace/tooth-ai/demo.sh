#!/bin/bash
# One-click demo script for Tooth-AI POC
# Starts Docker container, waits for services, and opens browser

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="tooth-ai-demo"
API_PORT=8000
UI_PORT=8501

echo "=========================================="
echo "Tooth-AI POC Demo Script"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container..."
    docker stop "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true
fi

# Build Docker image if it doesn't exist
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^tooth-ai:poc$"; then
    echo "Building Docker image (this may take a few minutes)..."
    cd "${SCRIPT_DIR}"
    docker build -t tooth-ai:poc . > docker_build.log 2>&1
    echo "✓ Docker image built"
else
    echo "✓ Docker image already exists"
fi

# Start container
echo ""
echo "Starting Docker container..."
cd "${SCRIPT_DIR}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p ${API_PORT}:8000 \
    -p ${UI_PORT}:8501 \
    -v "${SCRIPT_DIR}/models:/app/models:ro" \
    -v "${SCRIPT_DIR}/tmp:/app/tmp" \
    -v "${SCRIPT_DIR}/logs:/app/logs" \
    -v "${SCRIPT_DIR}/feedback:/app/feedback" \
    tooth-ai:poc

echo "✓ Container started: ${CONTAINER_NAME}"

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
MAX_WAIT=60
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${API_PORT}/ping > /dev/null 2>&1; then
        echo "✓ API is ready"
        break
    fi
    
    if [ $WAIT_COUNT -eq 0 ]; then
        echo -n "  Waiting"
    fi
    echo -n "."
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo ""
    echo "⚠ Warning: Services may not be fully ready. Continuing anyway..."
else
    echo ""
fi

# Test API
echo ""
echo "Testing API endpoint..."
if curl -s http://localhost:${API_PORT}/ping > /dev/null; then
    echo "✓ API is responding"
    
    # Try a sample prediction if test image exists
    TEST_IMAGE="${SCRIPT_DIR}/tests/sample_image.png"
    if [ -f "${TEST_IMAGE}" ]; then
        echo "  Running sample prediction..."
        curl -X POST http://localhost:${API_PORT}/predict \
             -F "file=@${TEST_IMAGE}" \
             -F "return_visualization=false" \
             -o "${SCRIPT_DIR}/tmp/smoke_pred.json" 2>/dev/null || true
        
        if [ -f "${SCRIPT_DIR}/tmp/smoke_pred.json" ]; then
            echo "  ✓ Sample prediction completed"
        fi
    fi
else
    echo "⚠ API may not be fully ready"
fi

# Open browser
echo ""
echo "=========================================="
echo "POC System Running!"
echo "=========================================="
echo ""
echo "FastAPI Server: http://localhost:${API_PORT}"
echo "Streamlit UI:   http://localhost:${UI_PORT}"
echo ""
echo "API Endpoints:"
echo "  GET  http://localhost:${API_PORT}/ping"
echo "  POST http://localhost:${API_PORT}/predict"
echo "  GET  http://localhost:${API_PORT}/version"
echo "  GET  http://localhost:${API_PORT}/model_info"
echo ""

# Try to open browser (platform-specific)
if command -v xdg-open > /dev/null; then
    # Linux
    xdg-open "http://localhost:${UI_PORT}" > /dev/null 2>&1 &
elif command -v open > /dev/null; then
    # macOS
    open "http://localhost:${UI_PORT}" > /dev/null 2>&1 &
elif command -v start > /dev/null; then
    # Windows (Git Bash)
    start "http://localhost:${UI_PORT}" > /dev/null 2>&1 &
fi

echo "To stop the demo, run:"
echo "  docker stop ${CONTAINER_NAME}"
echo ""
echo "To view logs, run:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""



