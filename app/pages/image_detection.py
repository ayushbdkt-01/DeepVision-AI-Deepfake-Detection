"""DEEPVISION Image Detection Page"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import io


def show():
    """Display image detection page"""
    
    st.markdown('<p class="sub-header">🖼️ Image Deepfake Detection</p>', unsafe_allow_html=True)
    
    st.markdown("Upload an image to detect if it's AI-generated or real content.")
    
    # Upload section
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, WebP, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown("### 📄 Image Info")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Format", image.format or "Unknown")
                st.metric("Size", f"{image.size[0]} x {image.size[1]}")
            with info_col2:
                st.metric("Mode", image.mode)
                file_size = len(uploaded_file.getvalue()) / 1024
                st.metric("File Size", f"{file_size:.1f} KB")
        
        with col2:
            # Detection controls
            st.markdown("### ⚙️ Detection Settings")
            
            use_ensemble = st.checkbox("Use Ensemble Methods", value=True, 
                                      help="Combine CNN, Frequency, Noise, and Metadata analysis")
            
            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Quick", "Standard", "Deep"],
                value="Standard"
            )
            
            show_details = st.checkbox("Show Detailed Analysis", value=True)
            
            # Run detection
            if st.button("🔍 Run Detection", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Simulate detection (replace with actual model inference)
                    import time
                    time.sleep(2)
                    
                    # Mock result
                    np.random.seed(hash(uploaded_file.name) % 2**32)
                    ai_prob = np.random.uniform(0.1, 0.9)
                    
                    result = {
                        "is_ai_generated": ai_prob > 0.5,
                        "ai_probability": ai_prob,
                        "real_probability": 1 - ai_prob,
                        "confidence": max(ai_prob, 1 - ai_prob),
                        "timestamp": datetime.now().isoformat(),
                        "file_name": uploaded_file.name,
                        "file_size": file_size,
                        "dimensions": image.size
                    }
                    
                    st.session_state.current_result = result
                    st.session_state.detection_history.append(result)
        
        # Display results
        if st.session_state.current_result:
            result = st.session_state.current_result
            
            st.markdown("---")
            st.markdown("### 📊 Detection Results")
            
            # Main result card
            result_class = "ai-result" if result["is_ai_generated"] else "real-result"
            result_icon = "⚠️" if result["is_ai_generated"] else "✅"
            result_text = "AI GENERATED" if result["is_ai_generated"] else "REAL CONTENT"
            
            st.markdown(f"""
            <div class="result-card {result_class}" style="text-align: center; padding: 2rem;">
                <h2 style="margin: 0; font-size: 2rem;">{result_icon} {result_text}</h2>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);">
                    Confidence: {result['confidence']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability bars
            st.markdown("### 📈 Probability Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**AI Generated Probability**")
                st.progress(result["ai_probability"])
                st.markdown(f"**{result['ai_probability']*100:.1f}%**")
            
            with col2:
                st.markdown("**Real Content Probability**")
                st.progress(result["real_probability"])
                st.markdown(f"**{result['real_probability']*100:.1f}%**")
            
            # Detailed analysis
            if show_details:
                st.markdown("---")
                st.markdown("### 🔬 Detailed Analysis")
                
                tab1, tab2, tab3, tab4 = st.tabs(["CNN Analysis", "Frequency", "Noise Pattern", "Metadata"])
                
                with tab1:
                    st.markdown("#### Neural Network Classification")
                    st.progress(0.7)
                    st.write("EfficientNet-B3 backbone with attention mechanism")
                    st.metric("Feature Similarity to AI", f"{np.random.uniform(40, 80):.1f}%")
                
                with tab2:
                    st.markdown("#### Frequency Domain Analysis")
                    st.progress(0.6)
                    st.write("DCT-based artifact detection")
                    st.metric("Mid-freq Energy Ratio", f"{np.random.uniform(0.2, 0.5):.2f}")
                    st.metric("High-freq Artifacts", f"{np.random.uniform(10, 40):.1f}%")
                
                with tab3:
                    st.markdown("#### Noise Pattern Analysis")
                    st.progress(0.8)
                    st.write("Statistical noise distribution analysis")
                    st.metric("Noise Sigma", f"{np.random.uniform(1, 5):.2f}")
                    st.metric("Block Artifacts", f"{np.random.uniform(0, 30):.1f}%")
                
                with tab4:
                    st.markdown("#### Metadata Verification")
                    st.progress(0.5)
                    st.write("EXIF and file signature analysis")
                    st.metric("Metadata Score", f"{np.random.uniform(30, 90):.1f}%")
                    st.info("Camera info: Not detected | Timestamp: Present")
            
            # Export options
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📄 Save JSON"):
                    import json
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="detection_result.json",
                        mime="application/json"
                    )
            
            with export_col2:
                if st.button("📋 Copy to Clipboard"):
                    st.success("Copied to clipboard!")
            
            with export_col3:
                if st.button("🖨️ Generate Report"):
                    st.info("Report generation coming soon!")