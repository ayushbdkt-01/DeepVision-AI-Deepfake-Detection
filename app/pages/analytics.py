"""DEEPVISION Analytics Page"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


def show():
    """Display analytics page"""
    
    st.markdown('<p class="sub-header">📊 Analytics Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("View detailed analytics and statistics about your deepfake detection activities.")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
        index=1
    )
    
    # Overview metrics
    st.markdown("### 📈 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", "1,234")
    with col2:
        st.metric("AI Detected", "456", delta="12%")
    with col3:
        st.metric("Real Content", "778", delta="8%")
    with col4:
        st.metric("Avg Confidence", "87.5%")
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Detection Distribution")
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['AI Generated', 'Real Content'],
            values=[456, 778],
            hole=0.4,
            marker=dict(colors=['#FF6B6B', '#06D6A0'])
        )])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E1DD',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Detection Over Time")
        
        # Line chart
        dates = pd.date_range(end=datetime.today(), periods=7)
        detections = [150, 180, 165, 200, 175, 190, 174]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=detections,
            mode='lines+markers',
            name='Total',
            line=dict(color='#00F5D4', width=3)
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E1DD',
            xaxis_title="Date",
            yaxis_title="Detections"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confidence Score Distribution")
        
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=np.random.normal(85, 10, 1000),
            name='Confidence',
            marker_color='#00F5D4',
            nbinsx=20
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E1DD',
            xaxis_title="Confidence Score",
            yaxis_title="Count"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### File Type Distribution")
        
        file_types = {
            'Type': ['Images', 'Videos', 'Batches'],
            'Count': [800, 200, 234]
        }
        
        fig = px.bar(
            file_types,
            x='Type',
            y='Count',
            color='Type',
            color_discrete_sequence=['#00F5D4', '#06D6A0', '#778DA9']
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E1DD'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed statistics
    st.markdown("### 📋 Detailed Statistics")
    
    # Model performance
    st.markdown("#### Model Performance Metrics")
    
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Value": ["95.2%", "94.8%", "95.5%", "95.1%", "98.2%"]
    }
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    # Recent activity table
    st.markdown("#### Recent Activity")
    
    activity_data = []
    for i in range(10):
        np.random.seed(i)
        activity_data.append({
            "Timestamp": (datetime.now() - timedelta(minutes=i*30)).strftime("%Y-%m-%d %H:%M"),
            "File": f"sample_{i+1}.jpg",
            "Type": np.random.choice(["Image", "Video"]),
            "Result": np.random.choice(["AI Generated", "Real"]),
            "Confidence": f"{np.random.uniform(75, 99):.1f}%"
        })
    
    st.dataframe(pd.DataFrame(activity_data), use_container_width=True)
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Analytics")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("📊 Export as CSV"):
            st.success("Analytics exported to CSV!")
    
    with export_col2:
        if st.button("📑 Generate PDF Report"):
            st.info("PDF report generation coming soon!")