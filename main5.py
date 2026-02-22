import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("🚢 Titanic Survival Predictor")

# -----------------------
# Upload Dataset
# -----------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Titanic CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    # -----------------------
    # Dataset Overview
    # -----------------------
    st.subheader("Dataset Overview")
    st.write("Total Passengers:", df.shape[0])
    st.write("Survived:", df["Survived"].sum())
    st.write("Did Not Survive:", df.shape[0] - df["Survived"].sum())

    st.write("First 5 Rows")
    st.dataframe(df.head())

    # -----------------------
    # Data Preprocessing
    # -----------------------
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1, errors="ignore")

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    label_encoders = {}
    for col in ["Sex", "Embarked"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -----------------------
    # Train Model
    # -----------------------
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.subheader("Model Performance")
    st.info(f"Model Accuracy: {accuracy:.3f}")

    # -----------------------
    # Prediction Section
    # -----------------------
    st.subheader("Predict Survival")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        age = st.slider("Age", 1, 80, 30)
        sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)

    with col2:
        fare = st.number_input("Fare", 0.0, 500.0, 50.0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])

    if st.button("Predict"):
        sex_encoded = label_encoders["Sex"].transform([sex])[0]
        embarked_encoded = label_encoders["Embarked"].transform([embarked])[0]

        input_data = np.array([[pclass, sex_encoded, age, sibsp, 0, fare, embarked_encoded]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.success(f"Prediction: SURVIVED ({probability*100:.1f}%)")
        else:
            st.error(f"Prediction: DID NOT SURVIVE ({probability*100:.1f}%)")

else:
    st.warning("Please upload a Titanic dataset CSV file.")
