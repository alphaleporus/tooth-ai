"""
Streamlit UI for clinician review of tooth detection results.
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import base64
from pathlib import Path
from PIL import Image
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import ToothDetectionEngine, load_engine
from inference.visualize import visualize_predictions, save_visualization


# Page config
st.set_page_config(
    page_title="Tooth-AI POC - Clinician Review",
    page_icon="🦷",
    layout="wide"
)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None


@st.cache_resource
def load_inference_engine():
    """Load inference engine (cached)."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    try:
        engine = load_engine(model_dir)
        return engine, None
    except Exception as e:
        return None, str(e)


def main():
    st.title("🦷 Tooth-AI POC - Clinician Review Interface")
    st.markdown("---")
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("Model Configuration")
        
        model_dir = st.text_input(
            "Model Directory",
            value=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
            help="Path to directory containing models"
        )
        
        if st.button("Load Models"):
            with st.spinner("Loading models..."):
                try:
                    engine = load_engine(model_dir)
                    st.session_state.engine = engine
                    st.success("Models loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading models: {e}")
        
        if st.session_state.engine:
            st.success("✓ Models ready")
        else:
            st.warning("Models not loaded")
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Load models using the button above
        2. Upload an OPG image
        3. Click 'Run Inference'
        4. Review results and provide feedback
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Image")
        
        uploaded_file = st.file_uploader(
            "Upload OPG Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an orthopantomogram (OPG) image"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert to BGR for OpenCV
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
                else:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            st.session_state.uploaded_image = image_np
            
            # Display image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Save temporarily
            temp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tmp', 'uploaded_image.png')
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            cv2.imwrite(temp_path, image_np)
            
            # Run inference button
            if st.button("🔍 Run Inference", type="primary", use_container_width=True):
                if st.session_state.engine is None:
                    st.error("Please load models first!")
                else:
                    with st.spinner("Running inference..."):
                        try:
                            results = st.session_state.engine.predict(temp_path, return_visualization=True)
                            st.session_state.results = results
                            st.success(f"Detected {results['num_detections']} teeth")
                        except Exception as e:
                            st.error(f"Error during inference: {e}")
    
    with col2:
        st.header("Results")
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Display visualization
            if 'visualization' in results:
                vis_base64 = results['visualization']
                vis_bytes = base64.b64decode(vis_base64)
                st.image(vis_bytes, caption="Detection Results", use_container_width=True)
            
            # Display statistics
            st.subheader("Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Teeth Detected", results['num_detections'])
            
            with col_stat2:
                st.metric("Mask R-CNN Used", results['metadata']['maskrcnn_used'])
            
            with col_stat3:
                st.metric("EfficientNet Used", results['metadata']['effnet_used'])
            
            # Display teeth list
            st.subheader("Detected Teeth")
            
            # Create DataFrame-like view
            teeth_data = []
            for tooth in results['teeth']:
                teeth_data.append({
                    'FDI': tooth['fdi'],
                    'Confidence': f"{tooth['score']:.3f}",
                    'Method': tooth['method_used'],
                    'Corrected': 'Yes' if tooth['correction_applied'] else 'No'
                })
            
            st.dataframe(teeth_data, use_container_width=True, hide_index=True)
            
            # View options
            st.subheader("View Options")
            show_json = st.checkbox("Show JSON Output", value=False)
            
            if show_json:
                st.json(results)
            
            # Export options
            st.subheader("Export Results")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if st.button("📥 Download JSON", use_container_width=True):
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download",
                        data=json_str,
                        file_name="tooth_detection_results.json",
                        mime="application/json"
                    )
            
            with col_exp2:
                if st.button("🖼️ Download Visualization", use_container_width=True):
                    if 'visualization' in results:
                        st.download_button(
                            label="Download",
                            data=base64.b64decode(results['visualization']),
                            file_name="tooth_detection_visualization.png",
                            mime="image/png"
                        )
            
            # Feedback section
            st.markdown("---")
            st.subheader("Clinician Feedback")
            
            feedback_type = st.selectbox(
                "Feedback Type",
                ["No Issues", "Low Confidence", "Wrong FDI", "Missing Tooth", "Other"]
            )
            
            feedback_text = st.text_area("Additional Notes", height=100)
            
            if st.button("💾 Save Feedback", use_container_width=True):
                feedback_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feedback')
                os.makedirs(feedback_dir, exist_ok=True)
                
                feedback_data = {
                    'image_filename': uploaded_file.name if uploaded_file else 'unknown',
                    'feedback_type': feedback_type,
                    'feedback_text': feedback_text,
                    'results': results,
                    'timestamp': str(Path().cwd())
                }
                
                feedback_path = os.path.join(feedback_dir, f"feedback_{uploaded_file.name}.json")
                with open(feedback_path, 'w') as f:
                    json.dump(feedback_data, f, indent=2)
                
                st.success("Feedback saved!")
        else:
            st.info("Upload an image and run inference to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Tooth-AI POC** - Phase 5 Unified Pipeline")
    st.markdown("Dataset: Niha Adnan et al., Data in Brief, 2024")


if __name__ == '__main__':
    main()



