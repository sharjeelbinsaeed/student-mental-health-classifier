
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Mental Health Classifier",
                   page_icon="🧠", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load("model_assets/smh_model.joblib")


artifact = load_artifact()
model = artifact["model"]
enc_map = artifact["encoders"]
feature_cols = artifact["feature_cols"]
label_map = artifact["label_map"]  # {0:"Low",1:"Medium",2:"High"}

st.title("🧠 Student Mental Health Classifier")
st.write("This tool predicts **Low / Medium / High** mental-health category based on study, sleep, stress and related factors.")
st.caption("Educational demo — not a medical device.")

with st.expander("ℹ️ About the inputs"):
    st.markdown("""
    - Keep values realistic (e.g., sleep 4–10 hours, screen time 0–12 hours).
    - Some fields are categorical; we use the same fitted encoders from training.
    """)

with st.form("prediction_form"):
    # Numeric inputs
    age = st.number_input("Age (years)", min_value=5.0,
                          max_value=60.0, value=18.0, step=1.0)
    study_hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=16.0, value=2.0, step=0.5)
    pressure = st.number_input(
        "Academic pressure (0–10)", min_value=0.0, max_value=10.0, value=5.0, step=1.0)
    screen_hours = st.number_input(
        "Screen time per day (hours)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    sleep_hours = st.number_input(
        "Sleep per night (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    env_rating = st.number_input(
        "Rate school environment (0–10)", min_value=0.0, max_value=10.0, value=6.0, step=1.0)

    # Categorical inputs (use classes_ from fitted encoders to avoid mismatch)
    gender_raw = st.selectbox("Gender", enc_map['Gender'].classes_.tolist())
    school_type_raw = st.selectbox(
        "School/College Type", enc_map['School Type'].classes_.tolist())
    tuition_raw = st.selectbox(
        "Attend tuition/coaching?", enc_map['Tuition'].classes_.tolist())
    physical_raw = st.selectbox(
        "Physical activity (hrs/week, categorical)", enc_map['PhysicalActivity'].classes_.tolist())
    struggle_raw = st.selectbox(
        "Family struggles with fees/books?", enc_map['StruggleWithFees'].classes_.tolist())
    parent_edu_raw = st.selectbox(
        "Parents' highest education", enc_map['ParentEducation'].classes_.tolist())
    stress_raw = st.selectbox(
        "Stress frequency (past month)", enc_map['StressLevel'].classes_.tolist())

    submitted = st.form_submit_button("Predict")

if submitted:
    # Apply the same encoders used in training
    gender = enc_map['Gender'].transform([gender_raw])[0]
    school_type = enc_map['School Type'].transform([school_type_raw])[0]
    tuition = enc_map['Tuition'].transform([tuition_raw])[0]
    physical = enc_map['PhysicalActivity'].transform([physical_raw])[0]
    struggle = enc_map['StruggleWithFees'].transform([struggle_raw])[0]
    parent_edu = enc_map['ParentEducation'].transform([parent_edu_raw])[0]
    stress = enc_map['StressLevel'].transform([stress_raw])[0]

    # Build a single-row DataFrame with EXACT feature names & order
    row = pd.DataFrame([{
        'What is your age? ': age,
        'Gender': gender,
        'School Type': school_type,
        '  How many hours do you study per day?  ': study_hours,
        'Tuition': tuition,
        'Do you feel academic pressure from your family? ': pressure,
        'How many hours do you spend on a screen (mobile/laptop) daily?  ': screen_hours,
        '  How many hours do you sleep per night (on average)?  ': sleep_hours,
        'Do you take part in any physical activity or sports (hrs/week)?  ': physical,
        '  Rate your school environment.  ': env_rating,
        'StruggleWithFees': struggle,
        'ParentEducation': parent_edu,
        'StressLevel': stress
    }])[feature_cols]  # enforce same order as training

    pred = int(model.predict(row)[0])
    st.success(f"Predicted Category: **{label_map[pred]}**")

    # Optional confidence proxy (SVC without probability=True doesn't give probabilities)
    # If you later retrain with probability=True, you can show predicted_proba here.

st.markdown("---")
st.subheader("Batch Prediction (optional)")
st.write("Upload a CSV with the original raw columns (same as your training data). We'll apply the training encoders and predict for each row.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])


def preprocess_uploaded(df_raw):
    """Apply the SAME preprocessing as training (must mirror train_model.py)."""
    df = df_raw.copy()
    df = df.fillna(df.mean(numeric_only=True))
    df['What is your age? '] = df['What is your age? '].astype(
        str).str.extract(r'(\d+)').astype(float)

    df['Gender'] = enc_map['Gender'].transform(df['Gender   '])
    df['School Type'] = enc_map['School Type'].transform(
        df['School/College Type '])
    df['Tuition'] = enc_map['Tuition'].transform(
        df['  Do you attend tuition/coaching after school/College?  '])
    df['StruggleWithFees'] = enc_map['StruggleWithFees'].transform(
        df['Does your family struggle to afford your education (fees/books)?  '])
    df['ParentEducation'] = enc_map['ParentEducation'].transform(
        df['Highest education level of your parents?  '])
    df['Do you take part in any physical activity or sports (hrs/week)?  '] = enc_map['PhysicalActivity'].transform(
        df['Do you take part in any physical activity or sports (hrs/week)?  ']
    )
    df['StressLevel'] = enc_map['StressLevel'].transform(
        df['In the past month, how often did you feel stressed, anxious, or mentally tired?']
    )

    X_batch = df[feature_cols].copy()
    return X_batch


if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        X_batch = preprocess_uploaded(df_raw)
        preds = model.predict(X_batch)
        labels = [label_map[int(p)] for p in preds]
        out = df_raw.copy()
        out["Prediction"] = labels
        st.dataframe(out.head(20))
        st.download_button("Download predictions CSV",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv",
                           mime="text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
