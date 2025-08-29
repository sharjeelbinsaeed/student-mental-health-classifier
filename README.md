# 🧠 Student Mental Health Classifier (SVM + Streamlit)

Predicts **Low / Medium / High** mental health category from student study, sleep, stress & context features.

## Demo
- Streamlit Cloud: https://sharjeelbinsaeed-student-mental-health-classifier.streamlit.app/<your-live-link

## How it works
- Target grouped: 1–3=Low, 4–7=Medium, 8–10=High
- Model: StandardScaler + SVC (rbf, class_weight='balanced')
- Files:
  - `train_model.py`: trains and saves `model_assets/smh_model.joblib`
  - `app_streamlit.py`: web UI to predict
  - `requirements.txt`: dependencies

## Run locally
```bash
conda create -n smh python=3.11 -y
conda activate smh
pip install -r requirements.txt
python train_model.py
streamlit run app_streamlit.py
