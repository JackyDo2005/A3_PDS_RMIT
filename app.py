"""
🛒 Online Shoppers Purchase Prediction & Segmentation
COSC2789 - Assignment 3 Demo Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Online Shoppers Prediction",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #f0f0f0;
        margin: 0.25rem 0;
    }
    .prediction-yes {
        color: #10b981;
        font-weight: bold;
        font-size: 1.8rem;
    }
    .prediction-no {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.8rem;
    }
    .help-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .help-box strong {
        color: #0d47a1;
    }
    .segment-card-0 {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .segment-card-0 h3, .segment-card-0 p, .segment-card-0 strong {
        color: white;
    }
    .segment-card-1 {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .segment-card-1 h3, .segment-card-1 p, .segment-card-1 strong {
        color: white;
    }
    .segment-card-2 {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .segment-card-2 h3, .segment-card-2 p, .segment-card-2 strong {
        color: white;
    }
    .segment-card-3 {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
    }
    .segment-card-3 h3, .segment-card-3 p, .segment-card-3 strong {
        color: #1a1a1a;
    }
    .segment-card-4 {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .segment-card-4 h3, .segment-card-4 p, .segment-card-4 strong {
        color: white;
    }
    .segment-card-5 {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
    }
    .segment-card-5 h3, .segment-card-5 p, .segment-card-5 strong {
        color: #1a1a1a;
    }
    .instruction-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .instruction-box strong {
        color: #b8860b;
    }
    .instruction-box h3 {
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Load models (with auto-training for cloud deployment)
@st.cache_resource
def load_models():
    """Load models from disk, or train them if they don't exist"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_files = {
        'lr': 'logistic_regression.pkl',
        'rf': 'random_forest.pkl',
        'xgb': 'xgboost.pkl',
        'kmeans': 'kmeans.pkl',
        'scaler': 'scaler.pkl',
        'label_encoders': 'label_encoders.pkl',
        'feature_names': 'feature_names.pkl'
    }
    
    # Check if all models exist
    all_exist = all((models_dir / f).exists() for f in model_files.values())
    
    if not all_exist:
        # First-time setup: train models
        st.info("🚀 **First-time setup:** Training models... This takes 3-5 minutes (only happens once!)")
        
        with st.spinner("Training AI models... Please wait..."):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from xgboost import XGBClassifier
                from sklearn.cluster import KMeans
                import warnings
                warnings.filterwarnings('ignore')
                
                # Load and prepare data
                st.write("📂 Loading data...")
                df = pd.read_csv('online_shoppers_intention.csv')
                df_clean = df.drop_duplicates()
                
                X = df_clean.drop('Revenue', axis=1)
                y = df_clean['Revenue']
                
                # Encode categorical variables
                st.write("🔄 Encoding categorical features...")
                categorical_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 
                                   'TrafficType', 'VisitorType', 'Weekend']
                
                label_encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                st.write("📊 Scaling features...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Train models
                st.write("🤖 Training Logistic Regression...")
                lr_model = LogisticRegression(random_state=42, max_iter=1000)
                lr_model.fit(X_train_scaled, y_train)
                
                st.write("🌲 Training Random Forest...")
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train_scaled, y_train)
                
                st.write("⚡ Training XGBoost...")
                xgb_model = XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
                xgb_model.fit(X_train_scaled, y_train)
                
                st.write("🎯 Training K-Means Clustering...")
                kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
                kmeans_model.fit(X_train_scaled)
                
                # Save models
                st.write("💾 Saving trained models...")
                joblib.dump(lr_model, models_dir / 'logistic_regression.pkl')
                joblib.dump(rf_model, models_dir / 'random_forest.pkl')
                joblib.dump(xgb_model, models_dir / 'xgboost.pkl')
                joblib.dump(kmeans_model, models_dir / 'kmeans.pkl')
                joblib.dump(scaler, models_dir / 'scaler.pkl')
                joblib.dump(label_encoders, models_dir / 'label_encoders.pkl')
                joblib.dump(X.columns.tolist(), models_dir / 'feature_names.pkl')
                
                st.success("✅ Models trained and saved successfully! App is ready to use.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during model training: {e}")
                return None
    
    # Load models from disk
    try:
        models = {
            'lr': joblib.load(models_dir / 'logistic_regression.pkl'),
            'rf': joblib.load(models_dir / 'random_forest.pkl'),
            'xgb': joblib.load(models_dir / 'xgboost.pkl'),
            'kmeans': joblib.load(models_dir / 'kmeans.pkl'),
            'scaler': joblib.load(models_dir / 'scaler.pkl'),
            'label_encoders': joblib.load(models_dir / 'label_encoders.pkl'),
            'feature_names': joblib.load(models_dir / 'feature_names.pkl')
        }
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# Header
st.markdown('<h1 class="main-header">🛒 Online Shoppers Purchase Predictor</h1>', unsafe_allow_html=True)
st.markdown("### COSC2789: Practical Data Science - Assignment 3")
st.markdown("**Team UG2** | Do Duy Hung (s3991053) | Ho Dinh Gia Bao (s4028938)")
st.markdown("---")

# Welcome Guide
with st.expander("📖 How to Use This App - Quick Start Guide", expanded=False):
    st.markdown("""
    ### Welcome! This app helps you predict online shopping behavior 🎯
    
    **What does this app do?**
    - Predicts if a website visitor will make a purchase
    - Groups customers into different segments for targeted marketing
    - Shows how accurate our predictions are
    
    **How to get started:**
    
    #### 1️⃣ Choose a Task (Left Sidebar)
    - **🔮 Purchase Prediction** - Predict if a visitor will buy something
    - **👥 Customer Segmentation** - See different types of customers
    - **📊 Model Performance** - View prediction accuracy metrics
    
    #### 2️⃣ For Purchase Prediction:
    - Fill in visitor information (page views, time spent, etc.)
    - Click "Predict Purchase Probability" button
    - See if they're likely to purchase and why
    
    #### 3️⃣ For Customer Segmentation:
    - Explore 6 different customer types
    - Learn marketing strategies for each segment
    - Understand customer behavior patterns
    
    **Don't worry!** All fields have default values. Just click "Predict" to see an example!
    """)

st.markdown("---")

# Check if models exist
models = load_models()
if models is None:
    st.error("⚠️ **Models not found!** The AI models need to be trained first.")
    
    st.markdown("""
    <div class="instruction-box">
        <h3>🚀 Quick Setup Instructions</h3>
        <p>Don't worry! This is easy to fix. Just follow these steps:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### For Windows Users:
        1. Open Command Prompt or PowerShell
        2. Navigate to project folder: `cd path/to/project`
        3. Run: `python save_models.py`
        4. Wait ~3 minutes for training
        5. Refresh this page
        """)
    with col2:
        st.markdown("""
        ### For Mac/Linux Users:
        1. Open Terminal
        2. Navigate to project folder: `cd path/to/project`
        3. Run: `python save_models.py`
        4. Wait ~3 minutes for training
        5. Refresh this page
        """)
    
    st.info("💡 This only needs to be done once! The models will be saved and reused.")
    st.stop()

# Sidebar - Model Selection
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; color: #1a1a1a;">
    <strong style="color: #0d47a1;">👋 Start Here!</strong><br>
    Choose what you want to do below ⬇️
</div>
""", unsafe_allow_html=True)

task_mode = st.sidebar.radio(
    "Select Task:",
    ["🔮 Purchase Prediction", "👥 Customer Segmentation", "📊 Model Performance"],
    help="Choose which analysis tool you want to use"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Quick Tips
- **Prediction**: Forecast if visitors will buy
- **Segmentation**: Understand customer types  
- **Performance**: See model accuracy
""")

# =============================================================================
# PREDICTION MODE
# =============================================================================
if task_mode == "🔮 Purchase Prediction":
    st.header("🔮 Predict Purchase Probability")
    
    st.markdown("""
    <div class="help-box">
        <strong>💡 What is this?</strong><br>
        Enter information about how a visitor browsed your website, and we'll predict 
        if they're likely to make a purchase. This helps you identify hot leads!
    </div>
    """, unsafe_allow_html=True)
    
    # Select classification model
    model_choice = st.sidebar.selectbox(
        "Classification Model:",
        ["Random Forest (Best)", "XGBoost", "Logistic Regression"],
        help="Random Forest gives the most accurate predictions (89% accuracy)"
    )
    
    st.markdown("### 📝 Step 1: Enter Visitor Information")
    
    # Add example presets
    with st.expander("💡 Try Example Scenarios", expanded=False):
        st.markdown("""
        **Quick Examples to Try:**
        - **Likely Buyer**: Product pages=20, Product duration=1200 sec, Page value=50, Returning visitor
        - **Window Shopper**: Product pages=8, Product duration=400 sec, Page value=10, New visitor
        - **Quick Browser**: Product pages=2, Product duration=30 sec, High bounce rate (0.2), New visitor
        
        *Tip: Use the default values and click Predict to see an example, then adjust the values to experiment!*
        """)
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📄 Page Visit Behavior")
        administrative = st.number_input(
            "Administrative Pages", 0, 50, 0,
            help="Number of account/admin pages visited (e.g., login, profile, settings)"
        )
        administrative_duration = st.number_input(
            "Admin Duration (sec)", 0.0, 5000.0, 0.0,
            help="Total time spent on administrative pages"
        )
        informational = st.number_input(
            "Informational Pages", 0, 50, 0,
            help="Number of 'About Us', 'FAQ', 'Contact' pages visited"
        )
        informational_duration = st.number_input(
            "Info Duration (sec)", 0.0, 5000.0, 0.0,
            help="Total time spent on informational pages"
        )
        product_related = st.number_input(
            "Product Pages", 0, 200, 1,
            help="Number of product/catalog pages viewed (most important!)"
        )
        product_related_duration = st.number_input(
            "Product Duration (sec)", 0.0, 10000.0, 0.0,
            help="Total time spent browsing products"
        )
    
    with col2:
        st.subheader("📊 Engagement Metrics")
        bounce_rates = st.slider(
            "Bounce Rate", 0.0, 0.5, 0.0, 0.01,
            help="% of visits where user left after viewing only one page (lower is better)"
        )
        exit_rates = st.slider(
            "Exit Rate", 0.0, 0.5, 0.2, 0.01,
            help="% of pageviews that were the last in the session (lower is better)"
        )
        page_values = st.slider(
            "Page Value", 0.0, 100.0, 0.0, 1.0,
            help="Average value of pages visited (higher = more purchase intent)"
        )
        special_day = st.slider(
            "Special Day Proximity", 0.0, 1.0, 0.0, 0.1,
            help="How close to a special day/holiday (0=not close, 1=very close)"
        )
    
    with col3:
        st.subheader("👤 Visitor Info")
        month = st.selectbox(
            "Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            help="Month of the visit"
        )
        visitor_type = st.selectbox(
            "Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"],
            help="Has this person visited your site before?"
        )
        weekend = st.checkbox(
            "Weekend Visit",
            help="Did they visit on Saturday or Sunday?"
        )
        operating_systems = st.selectbox(
            "Operating System", list(range(1, 9)),
            help="Visitor's operating system (1-8)"
        )
        browser = st.selectbox(
            "Browser", list(range(1, 14)),
            help="Visitor's web browser (1-13)"
        )
        region = st.selectbox(
            "Region", list(range(1, 10)),
            help="Geographic region (1-9)"
        )
        traffic_type = st.selectbox(
            "Traffic Type", list(range(1, 21)),
            help="How they arrived at your site (ads, direct, search, etc.)"
        )
    
    # Predict button
    st.markdown("### 🚀 Step 2: Get Prediction")
    if st.button("🚀 Predict Purchase Probability", type="primary", use_container_width=True):
        # Prepare input
        input_data = pd.DataFrame({
            'Administrative': [administrative],
            'Administrative_Duration': [administrative_duration],
            'Informational': [informational],
            'Informational_Duration': [informational_duration],
            'ProductRelated': [product_related],
            'ProductRelated_Duration': [product_related_duration],
            'BounceRates': [bounce_rates],
            'ExitRates': [exit_rates],
            'PageValues': [page_values],
            'SpecialDay': [special_day],
            'Month': [month],
            'OperatingSystems': [operating_systems],
            'Browser': [browser],
            'Region': [region],
            'TrafficType': [traffic_type],
            'VisitorType': [visitor_type],
            'Weekend': [weekend]
        })
        
        # Encode categorical
        categorical_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 
                           'TrafficType', 'VisitorType', 'Weekend']
        for col in categorical_cols:
            le = models['label_encoders'][col]
            try:
                input_data[col] = le.transform(input_data[col].astype(str))
            except:
                input_data[col] = 0  # Handle unseen categories
        
        # Scale
        input_scaled = models['scaler'].transform(input_data)
        
        # Predict
        model_map = {
            "Random Forest (Best)": models['rf'],
            "XGBoost": models['xgb'],
            "Logistic Regression": models['lr']
        }
        model = model_map[model_choice]
        
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        # Cluster assignment
        cluster = models['kmeans'].predict(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Step 3: View Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.markdown("### 🎯 Purchase Prediction")
            if prediction:
                st.markdown('<p class="prediction-yes">✅ WILL PURCHASE</p>', unsafe_allow_html=True)
                st.success("This visitor is likely to make a purchase!")
            else:
                st.markdown('<p class="prediction-no">❌ WON\'T PURCHASE</p>', unsafe_allow_html=True)
                st.warning("This visitor is unlikely to purchase at this time.")
        
        with result_col2:
            st.markdown("### 📈 Confidence Score")
            confidence = max(proba) * 100
            st.metric("Probability", f"{confidence:.1f}%", 
                     help="How confident the model is in this prediction")
            st.progress(confidence / 100)
            if confidence > 80:
                st.caption("🔥 Very confident prediction!")
            elif confidence > 60:
                st.caption("✅ Good confidence level")
            else:
                st.caption("⚠️ Moderate confidence")
        
        with result_col3:
            st.markdown("### 👥 Customer Segment")
            segment_names = ["Quick Browsers", "Window Shoppers", "Casual Visitors", 
                           "Active Shoppers", "Deal Hunters", "Premium Buyers"]
            segment_emojis = ["⚡", "👀", "📖", "🛍️", "🏷️", "💎"]
            st.metric("Segment", f"{segment_emojis[cluster]} {segment_names[cluster]}")
            st.caption(f"Belongs to customer segment {cluster + 1} of 6")
        
        # Add actionable recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommended Actions")
        
        recommendations = {
            (True, 0): ["Send immediate purchase incentive", "Show related products", "Offer time-limited discount"],
            (True, 1): ["Provide product comparisons", "Send personalized recommendations", "Highlight customer reviews"],
            (True, 2): ["Share valuable content", "Build brand trust", "Offer free resources"],
            (True, 3): ["Send cart reminder", "Offer free shipping", "Provide live chat support"],
            (True, 4): ["Promote special deals", "Highlight discounts", "Send exclusive offers"],
            (True, 5): ["Offer VIP benefits", "Premium product access", "Personalized service"],
            (False, 0): ["Use exit-intent popup", "Retarget with ads", "Simplify navigation"],
            (False, 1): ["Send follow-up email", "Offer comparison tools", "Show social proof"],
            (False, 2): ["Content marketing", "Newsletter signup", "Educational resources"],
            (False, 3): ["Address concerns", "Show guarantees", "Reduce friction"],
            (False, 4): ["Announce upcoming sales", "Create urgency", "Loyalty program signup"],
            (False, 5): ["Check for issues", "Provide premium support", "Ask for feedback"]
        }
        
        actions = recommendations.get((bool(prediction), cluster), ["Continue monitoring visitor behavior"])
        
        for idx, action in enumerate(actions, 1):
            st.markdown(f"**{idx}.** {action}")
        
        # Probability breakdown
        st.markdown("---")
        st.markdown("### 🎯 Detailed Probability Breakdown")
        
        col_viz1, col_viz2 = st.columns([2, 1])
        
        with col_viz1:
            prob_df = pd.DataFrame({
                'Outcome': ['No Purchase', 'Purchase'],
                'Probability': proba
            })
            fig = px.bar(prob_df, x='Outcome', y='Probability', 
                         color='Outcome',
                         color_discrete_map={'No Purchase': '#ef4444', 'Purchase': '#10b981'},
                         title=f"Prediction Confidence by {model_choice}",
                         text='Probability')
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_title="Probability", yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col_viz2:
            st.markdown("#### 📊 Summary")
            st.metric("Will Purchase", f"{proba[1]*100:.1f}%", 
                     help="Probability that this visitor will buy")
            st.metric("Won't Purchase", f"{proba[0]*100:.1f}%",
                     help="Probability that this visitor won't buy")
            st.markdown("---")
            st.caption(f"Model used: **{model_choice}**")
            st.caption(f"Accuracy: **89%**" if "Random Forest" in model_choice else 
                      f"Accuracy: **88%**" if "XGBoost" in model_choice else 
                      f"Accuracy: **87%**")

# =============================================================================
# SEGMENTATION MODE
# =============================================================================
elif task_mode == "👥 Customer Segmentation":
    st.header("👥 Customer Segmentation Analysis")
    
    st.markdown("""
    <div class="help-box">
        <strong>💡 What is this?</strong><br>
        We grouped your customers into 6 different types based on their browsing behavior. 
        Each group needs a different marketing approach to maximize sales!
    </div>
    """, unsafe_allow_html=True)
    
    # Load sample data for visualization
    @st.cache_data
    def load_sample_data():
        df = pd.read_csv('online_shoppers_intention.csv')
        return df.drop_duplicates().head(1000)  # Sample for speed
    
    df_sample = load_sample_data()
    
    st.markdown("""
    <div class="instruction-box">
        <strong>📊 Understanding Customer Segments</strong><br>
        We used AI (K-Means clustering) to identify 6 distinct customer types. 
        Each segment has unique behavior patterns and responds differently to marketing strategies.
    </div>
    """, unsafe_allow_html=True)
    
    # Cluster characteristics (from your analysis)
    cluster_profiles = {
        0: {"name": "Quick Browsers", "size": "~18%", "behavior": "Low engagement, high bounce rates", "emoji": "⚡"},
        1: {"name": "Window Shoppers", "size": "~15%", "behavior": "Medium engagement, exploring products", "emoji": "👀"},
        2: {"name": "Casual Visitors", "size": "~20%", "behavior": "Low product interaction, information seekers", "emoji": "📖"},
        3: {"name": "Active Shoppers", "size": "~17%", "behavior": "High page views, moderate purchase intent", "emoji": "🛍️"},
        4: {"name": "Deal Hunters", "size": "~16%", "behavior": "Special day visitors, price-conscious", "emoji": "🏷️"},
        5: {"name": "Premium Buyers", "size": "~14%", "behavior": "High engagement, high conversion rate", "emoji": "💎"}
    }
    
    # Display cluster cards with distinct colors
    st.markdown("### 🎯 The 6 Customer Segments")
    cols = st.columns(3)
    for i, (cluster_id, profile) in enumerate(cluster_profiles.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="segment-card-{cluster_id}">
                <h3>{profile['emoji']} Segment {cluster_id + 1}: {profile['name']}</h3>
                <p><strong>Size:</strong> {profile['size']} of all customers</p>
                <p><strong>Behavior:</strong> {profile['behavior']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📈 Detailed Segment Analysis")
    
    # Interactive cluster exploration
    selected_cluster = st.selectbox(
        "Select a segment to see detailed marketing strategies:", 
        [f"{cluster_profiles[i]['emoji']} Segment {i+1}: {cluster_profiles[i]['name']}" 
         for i in range(6)],
        help="Choose a customer segment to learn how to market to them effectively"
    )
    
    cluster_idx = int(selected_cluster.split("Segment")[1].split(":")[0].strip()) - 1
    
    # Define detailed strategies for each segment
    strategies = {
        0: {
            "title": "⚡ Quick Browsers",
            "description": "These visitors leave quickly without much interaction.",
            "strategies": [
                "Send retargeting ads with urgency messaging (\"Limited time offer!\")",
                "Use exit-intent popups with discount codes",
                "Show compelling headlines and clear calls-to-action",
                "Simplify navigation to reduce bounce rate"
            ],
            "goal": "Convert them before they leave"
        },
        1: {
            "title": "👀 Window Shoppers",
            "description": "These visitors browse multiple products but don't commit.",
            "strategies": [
                "Display product recommendation widgets",
                "Send personalized email campaigns with viewed products",
                "Offer comparison tools to help decision-making",
                "Provide customer reviews and social proof"
            ],
            "goal": "Help them make a purchase decision"
        },
        2: {
            "title": "📖 Casual Visitors",
            "description": "These visitors are researching or seeking information.",
            "strategies": [
                "Create valuable content (blog posts, guides, tutorials)",
                "Build email newsletter with helpful tips",
                "Offer free resources in exchange for email signup",
                "Focus on brand awareness, not immediate sales"
            ],
            "goal": "Build trust and stay top-of-mind"
        },
        3: {
            "title": "🛍️ Active Shoppers",
            "description": "These visitors are seriously considering purchases.",
            "strategies": [
                "Send cart abandonment reminder emails",
                "Offer free shipping or small discounts to close the deal",
                "Provide live chat support for questions",
                "Show limited stock indicators to create urgency"
            ],
            "goal": "Remove barriers to purchase"
        },
        4: {
            "title": "🏷️ Deal Hunters",
            "description": "These visitors are price-sensitive and wait for sales.",
            "strategies": [
                "Promote flash sales and limited-time discounts",
                "Send holiday and special occasion offers",
                "Create a loyalty program with exclusive deals",
                "Highlight \"best value\" products"
            ],
            "goal": "Provide the deals they're looking for"
        },
        5: {
            "title": "💎 Premium Buyers",
            "description": "These are your best customers with high purchase intent.",
            "strategies": [
                "Offer premium tier or VIP membership programs",
                "Provide early access to new products",
                "Give personalized shopping experiences",
                "Focus on excellent customer service and retention"
            ],
            "goal": "Maximize lifetime value and loyalty"
        }
    }
    
    strategy_info = strategies[cluster_idx]
    
    st.markdown(f"""
    ### {strategy_info['title']}
    
    **📊 Customer Profile:**  
    {strategy_info['description']}
    
    **💡 Marketing Goal:**  
    {strategy_info['goal']}
    
    **🎯 Recommended Strategies:**
    """)
    
    for idx, strategy in enumerate(strategy_info['strategies'], 1):
        st.markdown(f"{idx}. {strategy}")
    
    # Add visual separator
    st.markdown("---")
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Segment Size", cluster_profiles[cluster_idx]['size'], help="Percentage of total customers in this segment")
    with col2:
        st.metric("Priority Level", ["Medium", "Medium", "Low", "High", "Medium", "Very High"][cluster_idx], help="How important this segment is for revenue")
    with col3:
        st.metric("Purchase Rate", ["Low", "Medium", "Low", "Medium-High", "Medium", "Very High"][cluster_idx], help="How likely they are to make a purchase")

# =============================================================================
# PERFORMANCE MODE
# =============================================================================
else:
    st.header("📊 Model Performance Metrics")
    
    st.markdown("""
    <div class="help-box">
        <strong>💡 What is this?</strong><br>
        This shows how accurate our prediction models are. Higher numbers = better predictions!
        We tested 3 different AI models and show you which one performs best.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Classification Models", "🔍 Clustering Models"])
    
    with tab1:
        st.subheader("🎯 Purchase Prediction Model Comparison")
        
        st.markdown("""
        <div class="instruction-box">
            <strong>📚 Understanding the Metrics:</strong><br>
            • <strong>Accuracy:</strong> How often the model is correct overall<br>
            • <strong>Precision:</strong> Of predicted purchases, how many actually purchased<br>
            • <strong>Recall:</strong> Of actual purchases, how many did we catch<br>
            • <strong>F1-Score:</strong> Balance between precision and recall<br>
            • <strong>ROC-AUC:</strong> Overall ability to distinguish buyers from non-buyers (higher is better)
        </div>
        """, unsafe_allow_html=True)
        
        # Performance metrics (from your notebook)
        perf_data = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Logistic Regression'],
            'Accuracy': [0.89, 0.88, 0.87],
            'Precision': [0.74, 0.72, 0.68],
            'Recall': [0.63, 0.61, 0.58],
            'F1-Score': [0.68, 0.66, 0.62],
            'ROC-AUC': [0.93, 0.92, 0.89]
        })
        
        metric_choice = st.selectbox(
            "Select metric to visualize:", 
            ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            help="Choose which performance metric to compare across models"
        )
        
        fig = px.bar(perf_data, x='Model', y=metric_choice, 
                     color='Model',
                     color_discrete_map={
                         'Random Forest': '#10b981',
                         'XGBoost': '#3b82f6', 
                         'Logistic Regression': '#8b5cf6'
                     },
                     title=f"Model Comparison: {metric_choice}")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1], yaxis_title=metric_choice + " Score")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(perf_data.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='lightgreen'), use_container_width=True)
        
        st.success("✅ **Recommendation:** Random Forest is the best model with 89% accuracy and 93% ROC-AUC score!")
        
        # Add interpretation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", "Random Forest", help="Highest overall performance")
        with col2:
            st.metric("Accuracy", "89%", help="Correct predictions out of 100")
        with col3:
            st.metric("ROC-AUC", "0.93", help="Excellent discrimination ability")
    
    with tab2:
        st.subheader("🔍 Customer Segmentation Model Comparison")
        
        st.markdown("""
        <div class="instruction-box">
            <strong>📚 Understanding Clustering Metrics:</strong><br>
            • <strong>Silhouette Score:</strong> How well-separated the clusters are (higher is better)<br>
            • <strong>Davies-Bouldin Index:</strong> Cluster separation quality (lower is better)<br>
            • <strong>Business Utility:</strong> How useful for real marketing decisions (higher is better)<br>
            • <strong>Cluster Balance:</strong> How evenly sized the customer groups are (higher is better)
        </div>
        """, unsafe_allow_html=True)
        
        cluster_perf = pd.DataFrame({
            'Model': ['K-Means (k=6)', 'DBSCAN', 'Agglomerative (k=2)'],
            'Silhouette Score': [0.227, 0.657, 0.646],
            'Davies-Bouldin Index': [1.335, 0.364, 0.559],
            'Business Utility': [0.95, 0.20, 0.40],
            'Cluster Balance': [0.90, 0.10, 0.60]
        })
        
        st.dataframe(cluster_perf, use_container_width=True)
        
        st.success("""
        ✅ **Winner: K-Means with 6 Clusters**
        
        **Why we chose K-Means:**
        - ✅ Creates 6 distinct customer segments (perfect for marketing teams)
        - ✅ Balanced group sizes - each segment has enough customers
        - ✅ Highest business utility (95%) - directly actionable insights
        - ✅ Each segment has unique characteristics and needs different strategies
        
        **Why not the others?**
        - ❌ DBSCAN & Agglomerative: Only 2 clusters (too simple for real marketing)
        - ❌ Not enough granularity to create targeted campaigns
        """)
        
        # Visual comparison
        st.markdown("### 📊 Business Value Comparison")
        fig = px.bar(cluster_perf, x='Model', y='Business Utility',
                     color='Model',
                     color_discrete_map={
                         'K-Means (k=6)': '#10b981',
                         'DBSCAN': '#ef4444',
                         'Agglomerative (k=2)': '#f59e0b'
                     },
                     title="Which Model Provides Most Business Value?")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 2rem; border-radius: 1rem; margin-top: 2rem;">
    <h3 style="color: white; margin-bottom: 0.5rem;">🎓 COSC2789: Practical Data Science - Assignment 3</h3>
    <p style="color: #f0f0f0; margin: 0.5rem 0;">
        <strong>RMIT University Vietnam</strong> | Team UG2 | January 2026
    </p>
    <p style="color: #f0f0f0; margin: 0.5rem 0;">
        Do Duy Hung (s3991053) | Ho Dinh Gia Bao (s4028938)
    </p>
    <p style="color: #e0e0e0; font-size: 0.9rem; margin-top: 1rem;">
        Built with ❤️ using Streamlit, Scikit-learn, and XGBoost
    </p>
</div>
""", unsafe_allow_html=True)
