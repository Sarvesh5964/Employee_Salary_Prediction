import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("Employee Income Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    return df

df = load_data()
st.subheader("Raw Data")
st.write(df.head())

# Preprocessing
def preprocess_data(df):
    df = df.dropna()
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

df_clean = preprocess_data(df)

# Train-test split
X = df_clean.drop("income", axis=1)
y = df_clean["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Model trained with accuracy: {acc:.2f}")

# EDA - Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# User Input
st.subheader("Enter Employee Details for Prediction")
user_input = {}
for col in X.columns:
    val = st.number_input(f"Enter {col}:", value=float(X[col].mean()))
    user_input[col] = val

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.write("Predicted Income Category:", ">50K" if prediction == 1 else "<=50K")