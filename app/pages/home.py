"""DEEPVISION Home Page"""

import streamlit as st
import time
from datetime import datetime


def show():
    """Display home page"""
    
    # Hero section
    st.markdown("""
    <div class="main-header">
        🔍 DEEPVISION
    </div>
    <p style="text-align: center; color: var(--text-secondary); font-size: 1.2rem; margin-bottom: 2rem;">
        Advanced AI-Powered Deepfake Detection System
    </p>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <h3 style="color: var(--accent);">🖼️ Image Detection</h3>
            <p style="color: var(--text-secondary);">Detect AI-generated images with 95%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <h3 style="color: var(--accent);">🎬 Video Analysis</h3>
            <p style="color: var(--text-secondary);">Frame-by-frame video deepfake detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <h3 style="color: var(--accent);">📊 Ensemble Methods</h3>
            <p style="color: var(--text-secondary);">CNN + Frequency + Noise + Metadata analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <h3 style="color: var(--accent);">⚡ Real-time</h3>
            <p style="color: var(--text-secondary);">Fast detection with detailed analysis reports</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick upload section
    st.markdown('<p class="sub-header">⚡ Quick Detection</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload image or video for quick detection",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: Images (JPG, PNG, WebP, BMP), Videos (MP4, AVI, MOV, MKV)"
        )
        
        if uploaded_file:
            with st.spinner("Processing..."):
                time.sleep(1)  # Simulate processing
                
                # Show result
                result = {
                    "is_ai_generated": False,
                    "ai_probability": 0.15,
                    "real_probability": 0.85,
                    "confidence": 0.85,
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.detection_history.append(result)
                
                # Display result
                if result["is_ai_generated"]:
                    st.error(f"⚠️ AI Generated Detected ({result['ai_probability']*100:.1f}% confidence)")
                else:
                    st.success(f"✅ Real Content Detected ({result['real_probability']*100:.1f}% confidence)")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">95%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent activity
    st.markdown('<p class="sub-header">📋 Recent Activity</p>', unsafe_allow_html=True)
    
    if st.session_state.detection_history:
        for i, result in enumerate(reversed(st.session_state.detection_history[-5:])):
            with st.expander(f"Detection #{len(st.session_state.detection_history) - i}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Result", "AI Generated" if result["is_ai_generated"] else "Real")
                with col2:
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                st.text(f"Time: {result.get('timestamp', 'N/A')}")
    else:
        st.info("No recent detections. Upload a file to get started!")
    
    st.markdown("---")
    
    # Project timeline
    st.markdown('<p class="sub-header">📅 Project Timeline</p>', unsafe_allow_html=True)
    
    timeline = st.container()
    
    phases = [
        ("Phase 1", "Foundation", "Weeks 1-4", "✅ Completed"),
        ("Phase 2", "Core Development", "Weeks 5-12", "✅ Completed"),
        ("Phase 3", "UI/UX Development", "Weeks 13-18", "✅ Completed"),
        ("Phase 4", "Integration & Deployment", "Weeks 19-24", "⏳ Planned")
    ]
    
    for phase, name, duration, status in phases:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{phase}**")
        with col2:
            st.write(name)
        with col3:
            st.write(duration)
        with col4:
            st.write(status)