"""DEEPVISION - Deepfake Detection System

Main Streamlit application entry point.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="DEEPVISION - Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': '''
        ## DEEPVISION - Deepfake Detection System
        Advanced AI-powered detection system for identifying AI-generated images and videos.
        
        **Version:** 1.0.0
        **Phase:** Major Project (6 months)
        '''
    }
)


# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #0D1B2A;
        --secondary: #1B263B;
        --accent: #00F5D4;
        --warning: #FF6B6B;
        --success: #06D6A0;
        --background: #0A0F1A;
        --text-primary: #E0E1DD;
        --text-secondary: #778DA9;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--background) 0%, var(--primary) 100%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    .stMarkdown, .stText, p, span, div {
        font-family: 'Inter', sans-serif !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, var(--accent), #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: var(--accent) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, #00D4AA 100%);
        color: var(--primary) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 245, 212, 0.3) !important;
    }
    
    .upload-zone {
        border: 2px dashed var(--accent) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        text-align: center !important;
        background: rgba(0, 245, 212, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    .upload-zone:hover {
        background: rgba(0, 245, 212, 0.1) !important;
        border-color: var(--accent) !important;
    }
    
    .result-card {
        background: rgba(27, 38, 59, 0.8) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 245, 212, 0.2) !important;
    }
    
    .metric-card {
        background: var(--secondary) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        text-align: center !important;
    }
    
    .metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
    }
    
    .metric-label {
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
    }
    
    .sidebar .stRadio > div {
        background: var(--secondary) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), #00D4AA) !important;
    }
    
    .stAlert {
        background: var(--secondary) !important;
        border-radius: 10px !important;
    }
    
    .confidence-bar {
        height: 30px !important;
        border-radius: 15px !important;
        background: var(--secondary) !important;
        overflow: hidden !important;
    }
    
    .confidence-fill {
        height: 100% !important;
        border-radius: 15px !important;
        transition: width 0.5s ease !important;
    }
    
    .ai-result {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 107, 107, 0.1)) !important;
        border: 2px solid var(--warning) !important;
    }
    
    .real-result {
        background: linear-gradient(135deg, rgba(6, 214, 160, 0.2), rgba(6, 214, 160, 0.1)) !important;
        border: 2px solid var(--success) !important;
    }
    
    .tab-content {
        padding: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--secondary) !important;
        border-bottom: 2px solid var(--accent) !important;
    }
    
    div[data-testid="stSidebar"] {
        background: var(--primary) !important;
    }
    
    .stDataFrame {
        background: var(--secondary) !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None


def main():
    """Main application entry point"""
    local_css()
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="font-size: 1.8rem; margin: 0; color: var(--accent);">🔍 DEEPVISION</h1>
            <p style="color: var(--text-secondary); font-size: 0.8rem;">Deepfake Detection System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        pages = {
            "🏠 Home": "home",
            "🖼️ Image Detection": "image_detection",
            "🎬 Video Detection": "video_detection",
            "📁 Batch Processing": "batch_processing",
            "⚙️ Training Studio": "training_studio",
            "📊 Analytics": "analytics",
            "⚙️ Settings": "settings"
        }
        
        selected_page = st.radio(
            "Navigation",
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Processed", len(st.session_state.detection_history))
        with col2:
            ai_count = sum(1 for r in st.session_state.detection_history if r.get('is_ai_generated'))
            st.metric("AI Detected", ai_count)
        
        st.markdown("---")
        
        # System info
        st.markdown("### 💻 System Info")
        st.info("Model: EfficientNet-B3")
        st.info("Device: CPU")
        st.info("Status: Ready")
    
    # Route to selected page
    page = pages[selected_page]
    
    if page == "home":
        from app.pages import home
        home.show()
    elif page == "image_detection":
        from app.pages import image_detection
        image_detection.show()
    elif page == "video_detection":
        from app.pages import video_detection
        video_detection.show()
    elif page == "batch_processing":
        from app.pages import batch_processing
        batch_processing.show()
    elif page == "training_studio":
        from app.pages import training_studio
        training_studio.show()
    elif page == "analytics":
        from app.pages import analytics
        analytics.show()
    elif page == "settings":
        from app.pages import settings
        settings.show()


if __name__ == "__main__":
    main()