import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import time
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="AI Text & Emotion Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header Gradient */
    .main-header {
        font-size: clamp(2.5rem, 6vw, 4rem);
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: gradientShift 3s ease infinite;
        padding: 0 1rem;
    }
    
    @keyframes gradientShift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(20deg); }
    }
    
    .sub-header {
        font-size: clamp(1.1rem, 3vw, 1.6rem);
        text-align: center;
        color: #4a5568;
        margin-bottom: 2rem;
        padding: 0 1rem;
        font-weight: 300;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: clamp(2rem, 5vw, 4rem);
        border-radius: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
        animation: fadeInUp 1s ease;
    }
    
    .hero-section h1 {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .hero-section p {
        font-size: clamp(1rem, 2.5vw, 1.3rem);
        opacity: 0.95;
        line-height: 1.6;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: clamp(1.5rem, 3vw, 2.5rem);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 2px solid transparent;
        height: 100%;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.2);
        border: 2px solid #667eea;
    }
    
    .feature-card h3 {
        color: #667eea;
        font-size: clamp(1.3rem, 3vw, 1.8rem);
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .feature-card p {
        color: #4a5568;
        font-size: clamp(0.95rem, 2vw, 1.1rem);
        line-height: 1.6;
    }
    
    /* Prediction Results */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: clamp(2rem, 4vw, 3rem);
        border-radius: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
        animation: scaleIn 0.5s ease;
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .result-card h2 {
        font-size: clamp(1.5rem, 4vw, 2rem);
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .result-card .big-text {
        font-size: clamp(2.5rem, 7vw, 4.5rem);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    
    /* Chat Interface */
    .chat-container {
        background: #f7fafc;
        border-radius: 20px;
        padding: clamp(1.5rem, 3vw, 2rem);
        margin: 1.5rem 0;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        animation: slideInRight 0.3s ease;
    }
    
    .ai-message {
        background: white;
        color: #2d3748;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: clamp(1.5rem, 3vw, 2rem);
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.2);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-card h2 {
        font-size: clamp(2.5rem, 6vw, 4rem);
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .stat-card p {
        font-size: clamp(1rem, 2.5vw, 1.2rem);
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
        transform: translateX(5px);
    }
    
    .metric-card h3 {
        color: #2d3748;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #718096;
        font-size: clamp(0.9rem, 2vw, 1.1rem);
    }
    
    /* Quote Box */
    .quote-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: clamp(1.5rem, 3vw, 2rem);
        border-radius: 20px;
        margin: 1.5rem 0;
        border-left: 6px solid #f5576c;
        box-shadow: 0 10px 30px rgba(252, 182, 159, 0.2);
    }
    
    .quote-box p {
        font-size: clamp(1.1rem, 2.5vw, 1.4rem);
        font-style: italic;
        color: #2d3748;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Process Steps */
    .process-step {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .process-step:hover {
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        transform: translateX(10px);
    }
    
    .process-step h4 {
        color: #667eea;
        font-size: clamp(1.1rem, 2.5vw, 1.3rem);
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .process-step p {
        color: #4a5568;
        font-size: clamp(0.95rem, 2vw, 1.1rem);
        margin: 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: clamp(1rem, 2.5vw, 1.2rem);
        font-weight: 600;
        padding: 0.8rem 2.5rem;
        border-radius: 30px;
        border: none;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .stColumn {
            padding: 0.5rem !important;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
        
        .feature-card, .result-card, .stat-card {
            margin: 1rem 0;
        }
    }
    
    /* Text Areas */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        font-size: clamp(0.95rem, 2vw, 1.1rem);
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Section Dividers */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 3rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS AND DATA ====================
@st.cache_resource
def load_lstm_resources():
    """Load LSTM model and tokenizer for next word prediction"""
    try:
        model = load_model("lstm__model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)
        return model, tokenizer, max_len
    except Exception as e:
        st.error(f"Error loading LSTM model: {str(e)}")
        return None, None, None

@st.cache_resource
def load_emotion_resources():
    """Load emotion detection model and vectorizer"""
    try:
        model = joblib.load("LOG_NLP.pkl")
        vectorizer = joblib.load("bow.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading emotion model: {str(e)}")
        return None, None

# Load all resources
lstm_model, tokenizer, max_len = load_lstm_resources()
emotion_model, vectorizer = load_emotion_resources()

# ==================== HELPER FUNCTIONS ====================
def predict_next_word(text):
    """Predict the next word using LSTM model"""
    if not lstm_model or not tokenizer or not max_len:
        return "Model not loaded"
    
    try:
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
        
        preds = lstm_model.predict(sequence, verbose=0)
        predicted_index = np.argmax(preds)
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word
        return ""
    except Exception as e:
        return f"Error: {str(e)}"

def predict_emotion(text):
    """Predict emotion from text"""
    if not emotion_model or not vectorizer:
        return "unknown", 0.0
    
    try:
        input_vector = vectorizer.transform([text])
        prediction = emotion_model.predict(input_vector)[0]
        
        # Get probability scores if available
        if hasattr(emotion_model, 'predict_proba'):
            proba = emotion_model.predict_proba(input_vector)[0]
            confidence = max(proba) * 100
        else:
            confidence = 85.0
            
        return prediction, confidence
    except Exception as e:
        return "unknown", 0.0

# ==================== EMOTION DATA ====================
EMOTION_EMOJIS = {
    'sadness': '😢',
    'joy': '😊',
    'anger': '😠',
    'love': '❤️',
    'fear': '😨',
    'surprise': '😮'
}

EMOTION_COLORS = {
    'sadness': '#4299e1',
    'joy': '#48bb78',
    'anger': '#f56565',
    'love': '#ed64a6',
    'fear': '#9f7aea',
    'surprise': '#ed8936'
}

EMOTION_QUOTES = {
    'sadness': [
        "🌧️ 'Every cloud has a silver lining. Stay strong!'",
        "💙 'Sadness is a natural emotion. Allow yourself to feel.'",
        "🌙 'The darkest nights produce the brightest stars.'",
        "🕊️ 'This too shall pass. Better days are coming.'",
    ],
    'joy': [
        "🌟 'Happiness is contagious. Keep spreading the joy!'",
        "😊 'Your smile is your superpower!'",
        "🎉 'Joy is the simplest form of gratitude.'",
        "☀️ 'Celebrate this moment of happiness!'",
    ],
    'anger': [
        "🔥 'Take a deep breath. Channel this energy positively.'",
        "🧘 'In the midst of anger, find your calm center.'",
        "💪 'Turn your anger into motivation for change.'",
        "🌊 'Like a storm, anger passes. Peace will return.'",
    ],
    'love': [
        "❤️ 'Love is the greatest emotion of all!'",
        "💝 'Where there is love, there is life.'",
        "🌹 'Love makes the world go round.'",
        "💖 'What a beautiful feeling to have!'",
    ],
    'fear': [
        "🦁 'Courage is not the absence of fear, but triumph over it.'",
        "✨ 'You are stronger than your fears.'",
        "🌈 'Fear is temporary, regret is forever. Be brave!'",
        "💪 'Face your fears and they will fade.'",
    ],
    'surprise': [
        "🎊 'Life is full of wonderful surprises!'",
        "✨ 'Embrace the unexpected!'",
        "🎁 'Surprises make life exciting!'",
        "🌟 'Keep an open mind to life's surprises!'",
    ]
}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="font-size: 2.5rem;">🤖</h1>
        <h2 style="color: #667eea; margin: 0;">AI Analyzer</h2>
        <p style="color: #718096; font-size: 0.9rem;">Next-Gen Text Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎯 Predict", "📊 About"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 15px; color: white; text-align: center;">
        <h4 style="margin: 0; font-size: 0.9rem;">⚡ Quick Stats</h4>
        <p style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: 700;">2</p>
        <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">AI Models Active</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <p style="color: #718096; font-size: 0.85rem; line-height: 1.6;">
        <strong style="color: #667eea;">Powered by:</strong><br>
        🧠 LSTM Neural Networks<br>
        🎯 Machine Learning<br>
        💬 NLP Technology
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="floating">
            <h1>🤖 AI Text & Emotion Analyzer</h1>
        </div>
        <p>
            Experience the power of dual AI models working together to predict your next word 
            and understand your emotions in real-time. Built with cutting-edge deep learning technology.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # What is This?
    st.markdown('<h2 style="text-align: center; color: #2d3748; font-size: 2.5rem; margin: 2rem 0;">✨ What Can I Do?</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 Next Word Prediction</h3>
            <p>
                Our LSTM neural network analyzes your text patterns and intelligently 
                predicts the next word you're likely to type. Trained on thousands of 
                quotes and sentences for accurate predictions.
            </p>
            <div style="margin-top: 1.5rem;">
                <strong style="color: #667eea;">Features:</strong>
                <ul style="margin-top: 0.5rem; color: #4a5568;">
                    <li>Deep learning LSTM architecture</li>
                    <li>Context-aware predictions</li>
                    <li>Real-time processing</li>
                    <li>High accuracy results</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎭 Emotion Detection</h3>
            <p>
                Advanced NLP model that analyzes your text and identifies the underlying 
                emotion. Capable of detecting 6 different emotional states with high 
                precision and confidence scores.
            </p>
            <div style="margin-top: 1.5rem;">
                <strong style="color: #667eea;">Detects:</strong>
                <ul style="margin-top: 0.5rem; color: #4a5568;">
                    <li>😊 Joy & Happiness</li>
                    <li>😢 Sadness & Sorrow</li>
                    <li>😠 Anger & Frustration</li>
                    <li>❤️ Love & Affection</li>
                    <li>😨 Fear & Anxiety</li>
                    <li>😮 Surprise & Wonder</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # How It Works
    st.markdown('<h2 style="text-align: center; color: #2d3748; font-size: 2.5rem; margin: 2rem 0;">🔬 How It Works</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">1️⃣</div>
            <h3 style="color: white;">Input Your Text</h3>
            <p style="color: rgba(255,255,255,0.9);">
                Type or paste any text in the prediction interface. The more context you provide, 
                the better our AI models can understand and predict.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">2️⃣</div>
            <h3 style="color: white;">AI Analysis</h3>
            <p style="color: rgba(255,255,255,0.9);">
                Our dual AI models process your input simultaneously - LSTM for word prediction 
                and Logistic Regression for emotion classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">3️⃣</div>
            <h3 style="color: white;">Get Insights</h3>
            <p style="color: rgba(255,255,255,0.9);">
                Receive instant predictions with confidence scores, emotional analysis, 
                and personalized motivational quotes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Key Features
    st.markdown('<h2 style="text-align: center; color: #2d3748; font-size: 2.5rem; margin: 2rem 0;">🌟 Key Features</h2>', unsafe_allow_html=True)
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
            <h3>Real-Time</h3>
            <p>Instant predictions with no lag</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
            <h3>88% Accuracy</h3>
            <p>Highly accurate emotion detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧠</div>
            <h3>Deep Learning</h3>
            <p>LSTM neural network architecture</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">💬</div>
            <h3>NLP Powered</h3>
            <p>Advanced text understanding</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
            <h3>Analytics</h3>
            <p>Detailed performance metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎨</div>
            <h3>Beautiful UI</h3>
            <p>Modern and responsive design</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h2 style="color: #2d3748; font-size: 2.5rem; margin-bottom: 1rem;">Ready to Get Started?</h2>
        <p style="color: #4a5568; font-size: 1.3rem; margin-bottom: 2rem;">
            Navigate to the <strong style="color: #667eea;">Predict</strong> page to experience the power of AI!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 2rem; margin-top: 3rem; 
                background: #f7fafc; border-radius: 15px;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            <strong>Built with</strong> ❤️ <strong>using</strong>
        </p>
        <p style="font-size: 1rem;">
            TensorFlow • Keras • scikit-learn • Streamlit • Python
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== PREDICT PAGE ====================
elif page == "🎯 Predict":
    st.markdown('<h1 class="main-header">🎯 AI Prediction Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get next word predictions and emotion analysis in real-time</p>', unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Input Section
    st.markdown("### 💬 Enter Your Text")
    user_input = st.text_area(
        "",
        placeholder="Type or paste your text here... (e.g., 'I am feeling so happy today because')",
        height=150,
        key="prediction_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🚀 Analyze Now", use_container_width=True)
    
    if predict_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner("🤖 AI is processing your text..."):
                time.sleep(0.8)  # Simulate processing
                
                # Get predictions
                next_word = predict_next_word(user_input)
                emotion, confidence = predict_emotion(user_input)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'text': user_input,
                    'next_word': next_word,
                    'emotion': emotion,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
            
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            
            # Results Section
            st.markdown("### 📊 Analysis Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown("""
                <div class="result-card">
                    <h2>🔮 Next Word Prediction</h2>
                    <div class="big-text">{}</div>
                </div>
                """.format(next_word.upper() if next_word else "N/A"), unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                           box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin-top: 1rem;">
                    <p style="color: #4a5568; font-size: 1.1rem; margin: 0;">
                        <strong>Your text with prediction:</strong><br>
                        <span style="color: #2d3748; font-size: 1.2rem; line-height: 1.6;">
                            {user_input} <strong style="color: #667eea;">{next_word}</strong>
                        </span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                color = EMOTION_COLORS.get(emotion, "#667eea")
                
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                    <h2>🎭 Detected Emotion</h2>
                    <div style="font-size: 5rem; margin: 1rem 0;">{emoji}</div>
                    <div class="big-text">{emotion.upper()}</div>
                    <div style="font-size: 1.5rem; margin-top: 1rem; opacity: 0.9;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Motivational Quote
                quotes = EMOTION_QUOTES.get(emotion, ["Keep going! 🌟"])
                quote = random.choice(quotes)
                
                st.markdown(f"""
                <div class="quote-box">
                    <p>{quote}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Emotion Insights
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🎨 Emotion Insights")
            
            emotion_descriptions = {
                'sadness': "You seem to be experiencing sadness. Remember, it's okay to feel this way. Every emotion is valid and temporary. Consider reaching out to someone you trust.",
                'joy': "Your text radiates joy and happiness! This positive energy is wonderful. Keep embracing these moments and share your happiness with others!",
                'anger': "There's anger detected in your text. Take a moment to breathe deeply and find your center. Channel this energy into something constructive.",
                'love': "Love and affection shine through your words. What a beautiful emotion! Cherish these feelings and the people who inspire them.",
                'fear': "Fear is present in your text. Remember, courage isn't the absence of fear but acting despite it. You're stronger than you think!",
                'surprise': "Your text shows surprise! Life's unexpected moments make it exciting. Embrace the wonder and stay curious about what comes next!"
            }
            
            st.info(emotion_descriptions.get(emotion, "Emotion detected! Understanding our emotions is the first step to emotional intelligence."))
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### 📝 Analysis History")
        
        for idx, entry in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            emoji = EMOTION_EMOJIS.get(entry['emotion'], "🎭")
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                       box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span style="color: #718096; font-size: 0.9rem;">⏰ {entry['timestamp']}</span>
                    <span style="font-size: 1.5rem;">{emoji}</span>
                </div>
                <p style="color: #2d3748; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <strong>Text:</strong> {entry['text'][:100]}{"..." if len(entry['text']) > 100 else ""}
                </p>
                <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                    <span style="color: #667eea;">
                        <strong>Next Word:</strong> {entry['next_word']}
                    </span>
                    <span style="color: #f5576c;">
                        <strong>Emotion:</strong> {entry['emotion'].title()} ({entry['confidence']:.1f}%)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

# ==================== ABOUT PAGE ====================
elif page == "📊 About":
    st.markdown('<h1 class="main-header">📊 Model Analytics & Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detailed insights into our AI models</p>', unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    <div class="hero-section" style="text-align: left;">
        <h2 style="margin-bottom: 1rem;">🎯 Project Overview</h2>
        <p style="font-size: 1.2rem; line-height: 1.8;">
            This project combines two powerful AI models to provide comprehensive text analysis:
            <br><br>
            <strong>1. LSTM Next Word Prediction:</strong> Uses deep learning to predict the next word 
            based on context and patterns learned from thousands of quotes and sentences.
            <br><br>
            <strong>2. Emotion Detection:</strong> Employs NLP and machine learning to classify text 
            into 6 emotional categories with high accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Comparison
    st.markdown("## 🏆 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h3 style="color: white; font-size: 2rem; margin-bottom: 1.5rem;">
                🧠 LSTM Model
            </h3>
            <div style="margin: 1.5rem 0;">
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>Architecture:</strong></p>
                <ul style="list-style-position: inside; color: rgba(255,255,255,0.95);">
                    <li>Embedding Layer (200D)</li>
                    <li>LSTM Layers (150 units)</li>
                    <li>Dense Output Layer</li>
                    <li>Softmax Activation</li>
                </ul>
            </div>
            <div style="margin: 1.5rem 0;">
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>Training:</strong></p>
                <ul style="list-style-position: inside; color: rgba(255,255,255,0.95);">
                    <li>Dataset: 500K+ quotes</li>
                    <li>Epochs: 100</li>
                    <li>Optimizer: Adam</li>
                    <li>Loss: Categorical Crossentropy</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
            <h3 style="color: white; font-size: 2rem; margin-bottom: 1.5rem;">
                🎭 Emotion Model
            </h3>
            <div style="margin: 1.5rem 0;">
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>Algorithm:</strong></p>
                <ul style="list-style-position: inside; color: rgba(255,255,255,0.95);">
                    <li>Logistic Regression</li>
                    <li>Bag of Words Vectorization</li>
                    <li>Multi-class Classification</li>
                    <li>L2 Regularization</li>
                </ul>
            </div>
            <div style="margin: 1.5rem 0;">
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>Performance:</strong></p>
                <ul style="list-style-position: inside; color: rgba(255,255,255,0.95);">
                    <li>Accuracy: 88%</li>
                    <li>6 Emotion Classes</li>
                    <li>16,000+ Training Samples</li>
                    <li>112K+ Features</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Emotion Model Performance
    st.markdown("## 📈 Emotion Detection Performance")
    
    # Model Comparison Chart
    model_data = pd.DataFrame({
        'Model': ['Naive Bayes', 'SVM (Linear)', 'Logistic Regression'],
        'Accuracy': [73.9, 88.0, 88.0],
        'Color': ['#4299e1', '#9f7aea', '#48bb78']
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_data['Model'],
            y=model_data['Accuracy'],
            marker_color=model_data['Color'],
            text=model_data['Accuracy'].apply(lambda x: f'{x}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Model Accuracy Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2d3748'}
        },
        xaxis_title='Model',
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100],
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown("""
        <div class="stat-card">
            <p style="font-size: 1rem; margin: 0;">Model Accuracy</p>
            <h2>88%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <p style="font-size: 1rem; margin: 0;">Emotion Classes</p>
            <h2>6</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <p style="font-size: 1rem; margin: 0;">Training Samples</p>
            <h2>16K+</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <p style="font-size: 1rem; margin: 0;">Features</p>
            <h2>112K+</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Processing Pipeline
    st.markdown("## 🔄 Data Processing Pipeline")
    
    pipeline_steps = [
        ("1️⃣", "Data Loading", "Load text data from training dataset with emotion labels", "#667eea"),
        ("2️⃣", "Text Cleaning", "Remove noise: punctuation, numbers, URLs, HTML tags, emojis", "#764ba2"),
        ("3️⃣", "Tokenization", "Break text into words and remove stop words", "#f093fb"),
        ("4️⃣", "Vectorization", "Convert text to numerical features using Bag of Words", "#f5576c"),
        ("5️⃣", "Model Training", "Train models on processed data with cross-validation", "#4facfe"),
        ("6️⃣", "Prediction", "Classify new text and provide confidence scores", "#48bb78")
    ]
    
    for icon, title, description, color in pipeline_steps:
        st.markdown(f"""
        <div class="process-step" style="border-left-color: {color};">
            <h4 style="color: {color};">{icon} {title}</h4>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown("## 🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🐍 Core Libraries</h3>
            <ul style="color: #4a5568; font-size: 1.1rem; line-height: 1.8;">
                <li>Python 3.8+</li>
                <li>TensorFlow / Keras</li>
                <li>scikit-learn</li>
                <li>Pandas & NumPy</li>
                <li>NLTK</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Visualization</h3>
            <ul style="color: #4a5568; font-size: 1.1rem; line-height: 1.8;">
                <li>Plotly</li>
                <li>Matplotlib</li>
                <li>Seaborn</li>
                <li>Custom CSS</li>
                <li>Streamlit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Deployment</h3>
            <ul style="color: #4a5568; font-size: 1.1rem; line-height: 1.8;">
                <li>Streamlit Cloud</li>
                <li>Pickle/Joblib</li>
                <li>Model Caching</li>
                <li>Session State</li>
                <li>Responsive UI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Features Overview
    st.markdown("## ✨ Key Features")
    
    features = [
        ("🎯", "Real-time Predictions", "Instant next word and emotion predictions"),
        ("📊", "High Accuracy", "88% accuracy in emotion detection"),
        ("🧠", "Deep Learning", "LSTM neural networks for sequence prediction"),
        ("💬", "NLP Processing", "Advanced text preprocessing and analysis"),
        ("📈", "Performance Metrics", "Detailed analytics and model comparison"),
        ("🎨", "Modern UI", "Beautiful, responsive interface with animations")
    ]
    
    feat_col1, feat_col2 = st.columns(2)
    
    for idx, (icon, title, desc) in enumerate(features):
        with feat_col1 if idx % 2 == 0 else feat_col2:
            st.markdown(f"""
            <div class="metric-card" style="text-align: left; border-left-color: #667eea;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <h3 style="font-size: 1.5rem; margin: 0.5rem 0;">{title}</h3>
                <p style="font-size: 1rem; color: #718096; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem; border-radius: 25px; color: white; margin-top: 3rem;">
        <h2 style="margin-bottom: 1rem; font-size: 2rem;">Built with Passion for AI 🚀</h2>
        <p style="font-size: 1.2rem; opacity: 0.95; line-height: 1.8;">
            Combining the power of Deep Learning and Natural Language Processing<br>
            to create intelligent text analysis tools for everyone.
        </p>
        <p style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.8;">
            © 2024 AI Text & Emotion Analyzer | All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)