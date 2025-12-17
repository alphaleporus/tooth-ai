#!/bin/bash
# Quick launch script for Tooth-AI UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Tooth-AI UI Launcher"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing..."
    echo ""
    
    # Try user installation first
    if pip3 install --user streamlit 2>/dev/null; then
        echo "✓ Streamlit installed (user)"
    else
        echo "⚠ Installation failed. Trying with virtual environment..."
        echo ""
        
        # Create venv if it doesn't exist
        if [ ! -d ".venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv .venv
        fi
        
        # Activate and install
        source .venv/bin/activate
        pip install --quiet streamlit
        echo "✓ Streamlit installed (venv)"
    fi
else
    echo "✓ Streamlit already installed"
fi

echo ""
echo "=========================================="
echo "Launching UI..."
echo "=========================================="
echo ""
echo "The UI will open in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "Note: Models are not available, so inference will show an error."
echo "The UI interface will still be fully functional for preview."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Launch streamlit
if [ -n "$VIRTUAL_ENV" ]; then
    streamlit run ui/app.py --server.port 8501
else
    python3 -m streamlit run ui/app.py --server.port 8501
fi



