"""
Streamlit Frontend for Network Intrusion Detection System
Interactive web interface to test predictions and visualize results
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List

# Page configuration
st.set_page_config(
    page_title="IDS Prediction System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Initialize session state
if "api_available" not in st.session_state:
    st.session_state.api_available = False
if "last_predictions" not in st.session_state:
    st.session_state.last_predictions = []


def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_api_info():
    """Fetch API information"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.json()
    except:
        return None


def predict_single(features: List[float]):
    """Make a single prediction via API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def predict_batch(samples: List[List[float]]):
    """Make batch predictions via API"""
    try:
        response = requests.post(
            f"{API_URL}/predict-batch",
            json={"samples": samples},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Batch prediction failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None





def display_prediction_result(result):
    """Display a single prediction result"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Classification", result['prediction'])
    with col2:
        confidence_pct = result['confidence'] * 100
        st.metric("Confidence", f"{confidence_pct:.1f}%")
    with col3:
        st.metric("Model Used", result['model_used'])


def display_batch_results(result):
    """Display batch prediction results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", result['total_samples'])
    with col2:
        st.metric("Benign Traffic", result['benign_count'])
    with col3:
        st.metric("Attack Traffic", result['attack_count'])
    
    # Pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        data = {
            'Type': ['Benign', 'Attack'],
            'Count': [result['benign_count'], result['attack_count']]
        }
        fig = px.pie(data, values='Count', names='Type', 
                     title='Traffic Classification Distribution',
                     color_discrete_map={'Benign': '#2ecc71', 'Attack': '#e74c3c'})
        st.plotly_chart(fig, width='stretch')
    
    # Attack distribution
    with col2:
        predictions = result['predictions']
        attack_types = [p['prediction'] for p in predictions]
        attack_df = pd.DataFrame({'Attack Type': attack_types})
        attack_counts = attack_df['Attack Type'].value_counts()
        
        if len(attack_counts) > 0:
            fig = px.bar(x=attack_counts.index, y=attack_counts.values,
                        title='Attack Type Distribution',
                        labels={'x': 'Attack Type', 'y': 'Count'},
                        color=attack_counts.index,
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, width='stretch')
    
    # Detailed predictions table
    st.subheader("Detailed Predictions")
    predictions_df = pd.DataFrame(result['predictions'])
    st.dataframe(predictions_df, width='stretch')


# Main app
def main():
    st.title("🔒 Network Intrusion Detection System")
    st.markdown("ML-powered classification of network traffic as benign or malicious attacks")
    
    # Check API status
    api_available = check_api_health()
    
    if api_available:
        st.success("✓ API Backend Connected")
    else:
        st.error("✗ API Backend Unavailable - Make sure FastAPI is running on http://localhost:8000")
        st.info("Start the API with: `python3 -m uvicorn app.app:app --reload`")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Single Prediction", "Batch Prediction", "About"]
    )
    
    # HOME PAGE
    if page == "Home":
        st.header("Welcome to IDS Prediction System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 System Overview")
            st.markdown("""
            This system classifies network traffic into 6 categories:
            
            - **Benign** - Normal network traffic
            - **DoS** - Denial of Service attacks
            - **DDoS** - Distributed Denial of Service attacks
            - **Probe** - Reconnaissance attacks
            - **R2L** - Remote to Local attacks
            - **U2R** - User to Root attacks
            
            **Models Used:**
            - Logistic Regression
            - Random Forest
            - Support Vector Machine (SVM)
            - Naive Bayes
            """)
        
        with col2:
            st.subheader("🧠 Model Performance")
            st.info("""
            The system uses high-performance ML models trained on 
            network traffic datasets with 87 features per sample.
            
            **Best Model:** Random Forest (highest accuracy)
            
            **Input:** 87 network traffic features
            **Output:** Attack classification + confidence score
            """)
        
        # Features info
        st.subheader("📋 Expected Features")
        st.write("The system expects 82 network features including:")
        st.write("- Protocol info (TCP/UDP flags, lengths)")
        st.write("- Flow statistics (duration, bytes, packets)")
        st.write("- Window sizes and urgent flags")
        st.write("- And many more network characteristics...")
    
    # SINGLE PREDICTION PAGE
    elif page == "Single Prediction":
        st.header("Make a Single Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Method")
            input_method = st.radio(
                "Choose input method:",
                ["Manual Entry", "Demo Sample"]
            )
        
        features = None
        
        if input_method == "Manual Entry":
            st.subheader("Enter Network Features")
            
            # Create 82 input fields
            feature_values = []
            cols = st.columns(5)
            
            for i in range(82):
                col_idx = i % 5
                with cols[col_idx]:
                    value = st.number_input(
                        f"Feature {i+1}",
                        value=0.0,
                        step=0.1,
                        key=f"feature_{i}"
                    )
                    feature_values.append(value)
            
            features = feature_values
            
            if st.button("🔍 Predict"):
                with st.spinner("Making prediction..."):
                    result = predict_single(features)
                    if result:
                        st.success("Prediction Complete!")
                        display_prediction_result(result)
                        st.session_state.last_predictions.append(result)
        
        else:  # Demo Sample
            st.subheader("Demo Sample")
            st.info("Using a sample with mostly zeros (all features normalized)")
            
            # Create demo sample
            demo_sample = [0.0] * 82
            demo_sample[0] = 1.5  # Add some variation
            demo_sample[1] = 2.3
            demo_sample[2] = 0.8
            
            st.write("Sample features:")
            st.code(str(demo_sample[:10]) + " ... (82 features total)")
            
            if st.button("🔍 Predict on Demo Sample"):
                with st.spinner("Making prediction..."):
                    result = predict_single(demo_sample)
                    if result:
                        st.success("Prediction Complete!")
                        display_prediction_result(result)
                        st.session_state.last_predictions.append(result)
    
    # BATCH PREDICTION PAGE
    elif page == "Batch Prediction":
        st.header("Batch Prediction")
        
        st.subheader("Generate Test Samples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.number_input(
                "Number of samples to generate",
                min_value=1,
                max_value=100,
                value=10
            )
        
        with col2:
            sample_type = st.selectbox(
                "Sample type:",
                ["Random (varying features)", "Zeros", "Ones"]
            )
        
        if st.button("📊 Generate and Predict"):
            with st.spinner("Generating samples and making predictions..."):
                # Generate samples
                if sample_type == "Random (varying features)":
                    samples = np.random.randn(num_samples, 82).tolist()
                elif sample_type == "Zeros":
                    samples = np.zeros((num_samples, 82)).tolist()
                else:  # Ones
                    samples = np.ones((num_samples, 82)).tolist()
                
                # Make predictions
                result = predict_batch(samples)
                if result:
                    st.success(f"Predicted on {num_samples} samples!")
                    display_batch_results(result)
    
    # ABOUT PAGE
    elif page == "About":
        st.header("About This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Objective")
            st.write("""
            The Network Intrusion Detection System (IDS) is a machine learning 
            application designed to classify network traffic and identify 
            malicious activities in real-time.
            
            This system uses multiple ML algorithms to achieve high accuracy 
            in distinguishing between benign and malicious network traffic.
            """)
        
        with col2:
            st.subheader("🛠️ Technology Stack")
            st.write("""
            - **Backend:** FastAPI
            - **Frontend:** Streamlit
            - **ML Framework:** scikit-learn
            - **Data Processing:** pandas, numpy
            - **Visualization:** Plotly
            """)
        
        st.subheader("📚 Machine Learning Models")
        st.markdown("""
        | Model | Type | Complexity |
        |-------|------|-----------|
        | Logistic Regression | Linear | Low |
        | Random Forest | Ensemble | Medium |
        | SVM (RBF) | Kernel-based | High |
        | Naive Bayes | Probabilistic | Low |
        """)
        
        st.subheader("📊 Dataset Information")
        st.write("""
        - **Features:** 82 network traffic characteristics
        - **Classes:** 6 (Benign, DoS, DDoS, Probe, R2L, U2R)
        - **Preprocessing:** StandardScaler normalization
        - **Train/Test Split:** 80/20 with stratification
        """)
        
        st.subheader("📞 Support")
        st.write("""
        For issues or questions, please refer to the project documentation
        or contact the development team.
        """)


if __name__ == "__main__":
    main()
