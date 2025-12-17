# Dockerfile for Tooth-AI POC Production System
# Base: NVIDIA PyTorch container with CUDA support

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY api/ /app/api/
COPY ui/ /app/ui/
COPY inference/ /app/inference/
COPY models/ /app/models/
COPY export/ /app/export/
COPY cli/ /app/cli/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install API requirements
RUN if [ -f /app/api/requirements.txt ]; then \
    pip install --no-cache-dir -r /app/api/requirements.txt; \
    fi

# Install UI requirements (Streamlit)
RUN pip install --no-cache-dir streamlit

# Install additional dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pycocotools \
    timm \
    'git+https://github.com/facebookresearch/detectron2.git' \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn

# Create necessary directories
RUN mkdir -p /app/tmp /app/logs /app/feedback

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Tooth-AI services..."\n\
\n\
# Start FastAPI server in background\n\
cd /app\n\
uvicorn api.server:app --host 0.0.0.0 --port 8000 &\n\
API_PID=$!\n\
\n\
# Start Streamlit UI in background\n\
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &\n\
STREAMLIT_PID=$!\n\
\n\
echo "FastAPI running on http://0.0.0.0:8000"\n\
echo "Streamlit UI running on http://0.0.0.0:8501"\n\
echo "Services started. PIDs: API=$API_PID, Streamlit=$STREAMLIT_PID"\n\
\n\
# Wait for both processes\n\
wait $API_PID $STREAMLIT_PID\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]



