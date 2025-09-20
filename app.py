import pickle, os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ========================
# Load model & encoders
# ========================
@st.cache_resource(show_spinner="📦 Loading model and encoders...")
def load_model_and_encoders(dir):
    with open(os.path.join(dir, 'model.pkl'), 'rb') as file:
        model = pickle.load(file)
    with open(os.path.join(dir, 'encoder_X.pkl'), 'rb') as file:
        X_encoder = pickle.load(file)
    with open(os.path.join(dir, 'encoder_y.pkl'), 'rb') as file:
        y_encoder = pickle.load(file)
    return model, X_encoder, y_encoder

# ========================
# Main App
# ========================
def main(dir):
    # --------- PAGE CONFIG ---------
    st.set_page_config(
        page_title="ML Asset Forecast",
        page_icon="🏠",
        layout="wide",
    )
    # --------- CUSTOM CSS ---------
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #fff;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3, h4 {
            color: #f5f5f5 !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1e88e5, #1565c0);
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.6rem 1.2rem;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #42a5f5, #1e88e5);
            transform: scale(1.02);
        }
        .info-box {
            background: rgba(255,255,255,0.08);
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # --------- SIDEBAR ---------
    st.sidebar.title("⚙️ Options")
    st.sidebar.markdown("Upload data and run predictions with the ML model.")
    # --------- HERO TITLE ---------
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem; 
                    background: rgba(255,255,255,0.05); 
                    border-radius: 16px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h1>🏠 Real Estate Asset Prediction with Machine Learning</h1>
            <p style="font-size:27px;">Predict asset outcomes using a pre-trained Machine Learning model</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # --------- ABOUT APP ---------
    st.markdown(
        """
        <div class="info-box" style="margin-top:2rem; font-size:16px; font-weight:400;">
            <h3>📖 About This App</h3>
            <p>
            This application predicts asset outcomes using a pre-trained Machine Learning model.<br>
            Please prepare your CSV input data with the correct structure.<br>
            Below are the expected columns and examples of their possible groups/values.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # --------- SHOW FEATURE GROUPS FROM SAMPLE FILE ---------
    try:
        sample_input = pd.read_csv(os.path.join(dir, "x_test.csv"))
        st.subheader("🔎 Input Feature Groups")
        # Show sample expected format (example, adjust columns as needed)
        sample_df = pd.DataFrame({
            "age_group": ["20-29", "20-29", "50-59"],
            "household_type": ["Single", "Single", "Couple with children"],
            "num_children": ["1-2", "1-2", "0"],
            "income_bracket": ["Over 1,500,000 NOK", "500,000–1,000,000 NOK", "Under 500,000 NOK"],
            "tenure": ["Owner", "Owner", "Renter"],
            "property_pref": ["Detached house", "Apartment", "Detached house"],
            "urbanicity": ["Suburb", "City", "City"],
            "lifestyle_pref": ["Public transport", "Schools/Safety", "Culture/Lifestyle"],
            "education": ["Higher education - Bachelor", "Secondary school", "Higher education - Bachelor"],
            "employment": ["Self-employed", "Full-time job", "Full-time job"]
        })
        with st.expander("👀 Show expected columns and sample groups"):
            col1, col2 = st.columns(2)
            cols = list(sample_input.columns)
            half = len(cols) // 2
            with col1:
                for c in cols[:half]:
                    uniques = sample_input[c].dropna().unique()[:len(sample_input)]  # show up examples
                    st.markdown(
                        f"""
                        <div style="margin-bottom:1rem; padding:0.5rem; background:rgba(255,255,255,0.05); border-radius:8px;">
                            <b>{c}</b><br>
                            <span style="font-size:13px; color:#ddd;">Groups: {", ".join(map(str, uniques))}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            with col2:
                for c in cols[half:]:
                    uniques = sample_input[c].dropna().unique()[:5]
                    st.markdown(
                        f"""
                        <div style="margin-bottom:1rem; padding:0.5rem; background:rgba(255,255,255,0.05); border-radius:8px;">
                            <b>{c}</b><br>
                            <span style="font-size:13px; color:#ddd;">Groups: {", ".join(map(str, uniques))}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            st.subheader("📖 Example data")
            st.dataframe(sample_df, width="stretch")
    except Exception as e:
        st.warning("⚠️ Could not load sample input file to display feature groups.")
    # --------- LOAD MODEL ---------
    model, X_encoder, y_encoder = load_model_and_encoders(dir)
    # --------- FILE UPLOAD ---------
    st.subheader("📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file for prediction",
        type=["csv"]
    )
    if uploaded_file is not None:
        X_real = pd.read_csv(uploaded_file)
        st.markdown("### 📊 Input Data (Preview)")
        st.dataframe(X_real, width="stretch")
        if st.button("🚀 Run Prediction", type="primary"):
            with st.spinner("⏳ Processing..."):
                try:
                    # Encode & predict
                    x_convert = X_encoder.transform(X_real)
                    y_pred = model.predict(x_convert)
                    proba = model.predict_proba(x_convert)
                    confidence = np.max(proba, axis=1)
                    # Decode results
                    y_predict = y_encoder.inverse_transform(
                        y_pred.reshape(-1, 1)
                    ).flatten()

                    prediction = pd.DataFrame({
                        'Predicted': y_predict,
                        'Confidence': confidence.tolist()
                    })
                    st.success("✅ Prediction completed!")
                    # --------- SHOW RESULTS ---------
                    st.markdown("### 📈 Prediction Results")
                    st.dataframe(prediction, width="stretch")
                    # --------- CHART ---------
                    fig = px.bar(
                        prediction,
                        x=prediction.index[:len(prediction)],
                        y="Confidence",
                        color="Predicted",
                        text="Predicted",
                        title="🔍 Confidence of Predictions",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # --------- DOWNLOAD ---------
                    csv = prediction.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Results (.csv)",
                        data=csv,
                        file_name="prediction.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.info("⬆️ Please upload a CSV file to start.")

# ========================
# Run App
# ========================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    main(BASE_DIR)
