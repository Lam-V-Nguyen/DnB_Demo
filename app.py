import pickle, os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ========================
# Load model & encoders
# ========================
@st.cache_resource
def load_model_and_encoders(BASE_DIR):
    with open(os.path.join(BASE_DIR,'model.pkl'), 'rb') as file:
        model = pickle.load(file)
    with open(os.path.join(BASE_DIR,'encoder_X.pkl'), 'rb') as file:
        X_encoder = pickle.load(file)
    with open(os.path.join(BASE_DIR,'encoder_y.pkl'), 'rb') as file:
        y_encoder = pickle.load(file)
    return model, X_encoder, y_encoder
# ========================
# Main App
# ========================
def main(BASE_DIR):
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
            <h1>🏠 Machine Learning Asset Forecast</h1>
            <p style="font-size:18px;">Predict asset outcomes using a pre-trained Machine Learning model</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # --------- LOAD MODEL ---------
    model, X_encoder, y_encoder = load_model_and_encoders(BASE_DIR)
    # --------- INPUT FORMAT INFO ---------
    st.markdown(
        """
        <div class="info-box" style="margin-top: 2rem;">
            <h3>ℹ️ Input Data Format</h3>
            <p>Please upload a CSV file with the same structure as follows.<br>
            Ensure that column names and types match the expected format.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
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
    with st.expander("👀 Example of valid input data"):
        st.dataframe(sample_df, width="stretch")
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
                    styled_prediction = prediction.style.set_table_styles(
                        [
                            {
                                "selector": "th",
                                "props": [("font-weight", "bold"), ("font-size", "14px"), ("text-align", "center")]
                            }
                        ]
                    )
                    st.dataframe(styled_prediction, width="stretch")
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
                    st.plotly_chart(fig, width='stretch')
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

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    main(BASE_DIR)
