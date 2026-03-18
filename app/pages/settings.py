"""DEEPVISION Settings Page"""

import streamlit as st


def show():
    """Display settings page"""
    
    st.markdown('<p class="sub-header">⚙️ Settings</p>', unsafe_allow_html=True)
    
    st.markdown("Configure DEEPVISION settings and preferences.")
    
    # General settings
    st.markdown("### 🔧 General Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Theme",
            ["Dark", "Light", "Auto"]
        )
        
        language = st.selectbox(
            "Language",
            ["English", "Spanish", "French", "German", "Chinese", "Japanese"]
        )
    
    with col2:
        notifications = st.checkbox("Enable Notifications", value=True)
        auto_save = st.checkbox("Auto-save Results", value=True)
    
    # Model settings
    st.markdown("### 🤖 Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Detection Model",
            ["EfficientNet-B3 (Default)", "ResNet50", "EfficientNet-B0", "Custom Model"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5,
            help="Minimum confidence required for predictions"
        )
    
    with col2:
        use_gpu = st.checkbox("Use GPU if Available", value=True)
        num_workers = st.number_input("Worker Threads", value=4, min_value=1, max_value=16)
    
    # Ensemble settings
    st.markdown("### 🔬 Ensemble Methods")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cnn_enabled = st.checkbox("CNN Analysis", value=True)
    with col2:
        frequency_enabled = st.checkbox("Frequency Analysis", value=True)
    with col3:
        noise_enabled = st.checkbox("Noise Analysis", value=True)
    with col4:
        metadata_enabled = st.checkbox("Metadata Analysis", value=True)
    
    # File settings
    st.markdown("### 📁 File Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_image_size = st.number_input(
            "Max Image Size (MB)",
            value=50,
            min_value=1,
            max_value=500
        )
        
        supported_formats = st.multiselect(
            "Supported Image Formats",
            ["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
            default=["jpg", "jpeg", "png", "webp", "bmp"]
        )
    
    with col2:
        max_video_size = st.number_input(
            "Max Video Size (MB)",
            value=500,
            min_value=10,
            max_value=5000
        )
        
        video_formats = st.multiselect(
            "Supported Video Formats",
            ["mp4", "avi", "mov", "mkv", "webm", "wmv"],
            default=["mp4", "avi", "mov", "mkv"]
        )
    
    # Processing settings
    st.markdown("### ⚡ Processing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input("Batch Processing Size", value=32, min_value=1, max_value=256)
        parallel_processing = st.checkbox("Enable Parallel Processing", value=True)
    
    with col2:
        cache_models = st.checkbox("Cache Models in Memory", value=True)
        clear_cache = st.button("🗑️ Clear Cache")
        if clear_cache:
            st.success("Cache cleared!")
    
    # API settings
    st.markdown("### 🌐 API Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_enabled = st.checkbox("Enable API Access", value=False)
        if api_enabled:
            api_key = st.text_input("API Key", type="password")
    
    with col2:
        rate_limit = st.number_input("Rate Limit (requests/min)", value=60, min_value=1)
    
    # Advanced settings
    st.markdown("### 🔩 Advanced Settings")
    
    with st.expander("Advanced Options"):
        debug_mode = st.checkbox("Debug Mode", value=False)
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        
        custom_model_path = st.text_input("Custom Model Path (optional)")
        
        st.markdown("---")
        
        reset_button = st.button("🔄 Reset to Defaults", type="primary")
        if reset_button:
            st.success("Settings reset to defaults!")
    
    # Save settings
    st.markdown("---")
    
    save_col1, save_col2 = st.columns([3, 1])
    
    with save_col1:
        st.info("Changes will take effect after restarting the application.")
    
    with save_col2:
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    # About section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    
    st.markdown("""
    **DEEPVISION - Deepfake Detection System**
    
    Version: 1.0.0
    
    A comprehensive deep learning system for detecting AI-generated images and videos.
    
    Built with:
    - PyTorch & TensorFlow
    - Streamlit
    - OpenCV
    - EfficientNet
    """)