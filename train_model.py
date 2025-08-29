import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Load your data
df = pd.read_csv('student_mental_health.csv')  

# Print first few rows
print("Sample Data:\n", df.head())

df.isnull().sum()
df = df.fillna(df.mean(numeric_only=True))
print(df.columns)
numeric_df = df.select_dtypes(include=['number'])
target_col = 'On a scale of 010, how would you rate your current mental health?  '
correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
print(correlations)

from sklearn.preprocessing import LabelEncoder

# make a copy to avoid overwriting raw df
df = df.copy()

# dict to store encoders
encoders = {}

# Gender
encoders['Gender'] = LabelEncoder()
df['Gender'] = encoders['Gender'].fit_transform(df['Gender   '])

# School Type
encoders['School Type'] = LabelEncoder()
df['School Type'] = encoders['School Type'].fit_transform(df['School/College Type '])

# Tuition
encoders['Tuition'] = LabelEncoder()
df['Tuition'] = encoders['Tuition'].fit_transform(
    df['  Do you attend tuition/coaching after school/College?  '])

# Struggle with Fees
encoders['StruggleWithFees'] = LabelEncoder()
df['StruggleWithFees'] = encoders['StruggleWithFees'].fit_transform(
    df['Does your family struggle to afford your education (fees/books)?  '])

# Parents' Education
encoders['ParentEducation'] = LabelEncoder()
df['ParentEducation'] = encoders['ParentEducation'].fit_transform(
    df['Highest education level of your parents?  '])

# Physical Activity
encoders['PhysicalActivity'] = LabelEncoder()
df['Do you take part in any physical activity or sports (hrs/week)?  '] = \
    encoders['PhysicalActivity'].fit_transform(
        df['Do you take part in any physical activity or sports (hrs/week)?  '])

# Stress Level
encoders['StressLevel'] = LabelEncoder()
df['StressLevel'] = encoders['StressLevel'].fit_transform(
    df['In the past month, how often did you feel stressed, anxious, or mentally tired?'])

# Age numeric clean-up
df['What is your age? '] = df['What is your age? '].astype(str).str.extract(r'(\d+)').astype(float)

y = df['On a scale of 010, how would you rate your current mental health?  ']

# Relabel target into 3 groups
y_grouped = y.copy()
y_groupedd = y_grouped.replace({1: 0, 2: 0, 3: 0,    # Low
                               4: 1, 5: 1, 6: 1, 7: 1,  # Medium
                               8: 2, 9: 2, 10: 2})     # High

print("Class distribution after grouping:\n", y_groupedd.value_counts())

feature_cols = [
    'What is your age? ',
    'Gender',
    'School Type',
    '  How many hours do you study per day?  ',
    'Tuition',
    'Do you feel academic pressure from your family? ',
    'How many hours do you spend on a screen (mobile/laptop) daily?  ',
    '  How many hours do you sleep per night (on average)?  ',
    'Do you take part in any physical activity or sports (hrs/week)?  ',
    '  Rate your school environment.  ',
    'StruggleWithFees',
    'ParentEducation',
    'StressLevel'
]
X = df[feature_cols].copy()


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_groupedd, test_size=0.2, stratify=y_groupedd, random_state=42)

# Create pipeline: scaling + SVM
model = make_pipeline(StandardScaler(), SVC(
    kernel='rbf', C=1, gamma='scale', random_state=42))

# Train
model.fit(X_train, y_train)

#  SAVE TRAINED MODEL
os.makedirs("model_assets", exist_ok=True)
artifact = {
    "model": model,
    "encoders": encoders,
    "feature_cols": feature_cols,
    "label_map": {0: "Low", 1: "Medium", 2: "High"}
}

joblib.dump(artifact, "model_assets/smh_model.joblib")
print("Model saved to model_assets/smh_model.joblib")

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=["Low", "Medium", "High"]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[
            "Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Grouped SVM Classification")
plt.show()
