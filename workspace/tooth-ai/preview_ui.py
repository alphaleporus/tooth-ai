#!/usr/bin/env python3
"""
Simple UI preview script that can run without full dependencies.
Shows the UI structure and handles missing models gracefully.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__} available")
    except ImportError:
        missing.append("streamlit")
        print("✗ Streamlit not installed")
    
    try:
        import numpy
        print("✓ NumPy available")
    except ImportError:
        missing.append("numpy")
        print("✗ NumPy not installed")
    
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        missing.append("opencv-python")
        print("✗ OpenCV not installed")
    
    try:
        from PIL import Image
        print("✓ Pillow available")
    except ImportError:
        missing.append("Pillow")
        print("✗ Pillow not installed")
    
    return missing


def launch_ui():
    """Launch Streamlit UI."""
    ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'app.py')
    
    if not os.path.exists(ui_path):
        print(f"✗ UI file not found: {ui_path}")
        return False
    
    print(f"\n{'='*60}")
    print("Launching Tooth-AI UI")
    print(f"{'='*60}")
    print(f"\nUI File: {ui_path}")
    print("\nStarting Streamlit server...")
    print("The UI will open in your browser at: http://localhost:8501")
    print("\nNote: Models are not available, so inference will show an error.")
    print("The UI interface will still be fully functional for preview.")
    print("\nPress Ctrl+C to stop the server.")
    print(f"{'='*60}\n")
    
    # Try to launch
    try:
        import subprocess
        import sys
        
        # Use streamlit command
        cmd = [sys.executable, "-m", "streamlit", "run", ui_path, "--server.port", "8501"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
        return True
    except Exception as e:
        print(f"\n✗ Error launching UI: {e}")
        print("\nAlternative: Install streamlit and run manually:")
        print("  pip install streamlit")
        print("  streamlit run ui/app.py")
        return False


if __name__ == "__main__":
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("\nTo install:")
        print("  pip install --user " + " ".join(missing))
        print("\nOr install all requirements:")
        print("  pip install --user -r requirements_full.txt")
        print("\nAttempting to launch anyway...\n")
    else:
        print("\n✓ All dependencies available\n")
    
    launch_ui()



