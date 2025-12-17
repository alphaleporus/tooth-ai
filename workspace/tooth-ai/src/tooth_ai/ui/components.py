"""
Reusable UI components for Streamlit app.
"""

import streamlit as st
import json
from typing import Dict, List


def display_tooth_table(teeth: List[Dict]):
    """Display teeth in a formatted table."""
    import pandas as pd
    
    data = []
    for tooth in teeth:
        data.append({
            'FDI': tooth['fdi'],
            'Confidence': f"{tooth['score']:.3f}",
            'Method': tooth['method_used'],
            'Corrected': '✓' if tooth.get('correction_applied', False) else '✗',
            'Centroid': f"({tooth['centroid'][0]:.1f}, {tooth['centroid'][1]:.1f})"
        })
    
    df = pd.DataFrame(data)
    return df


def display_statistics(results: Dict):
    """Display statistics cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Teeth", results['num_detections'])
    
    with col2:
        st.metric("Mask R-CNN", results['metadata']['maskrcnn_used'])
    
    with col3:
        st.metric("EfficientNet", results['metadata']['effnet_used'])
    
    with col4:
        st.metric("Corrections", results['metadata']['corrections_applied'])

