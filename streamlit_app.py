# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load NLTK resources (ensure these are downloaded)
nltk.download('stopwords')  # Comment this line after the first run

# Load CountVectorizer
with open("count_vectorizer.pkl", "rb") as file:
    cv = pickle.load(file)

# Sidebar for model selection
st.sidebar.title("Choose Classifier")
classifier_name = st.sidebar.selectbox(
    "Select a model:",
    ['Decision Tree', 'K-Nearest Neighbors', 'Logistic Regression', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'Support Vector Classifier']
)

# Load selected model
try:
    with open(f"{classifier_name}_model.pkl", "rb") as file:
        classifier = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file for {classifier_name} not found.")
    st.stop()

# Title and input
st.title("Sentiment Analysis App")
st.write("Enter a review to predict whether it is positive or negative.")

# Text input
user_review = st.text_input("Your Review:")

# Preprocess user input
if user_review:
    # Preprocessing
    review = re.sub('[^a-zA-Z]', ' ', user_review)
    review = review.lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)

    # Transform the input into Bag of Words format
    input_data = cv.transform([review]).toarray()

    # Predict sentiment
    prediction = classifier.predict(input_data)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    # Display sentiment prediction
    st.write(f"**Prediction ({classifier_name}):**", sentiment)

    # Load test and train data to calculate metrics
    try:
        with open("X_test.pkl", "rb") as file:
            X_test = pickle.load(file)
        with open("y_test.pkl", "rb") as file:
            y_test = pickle.load(file)
        with open("X_train.pkl", "rb") as file:
            X_train = pickle.load(file)
        with open("y_train.pkl", "rb") as file:
            y_train = pickle.load(file)
    except FileNotFoundError:
        st.error("Error loading train/test data files.")
        st.stop()

    # Check for accuracy, bias, and variance
    y_pred = classifier.predict(X_test)  # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    bias = classifier.score(X_train, y_train)  # Model score on training data (bias)
    variance = classifier.score(X_test, y_test)  # Model score on test data (variance)

    # Debugging output
    st.write(f"**Test Accuracy (Model Accuracy)**: {accuracy:.2f}")  # Display model accuracy first
    st.write(f"**Training Accuracy (Bias)**: {bias:.2f}")
    st.write(f"**Test Accuracy (Variance)**: {variance:.2f}")

    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Display confusion matrix as DataFrame for better readability
    cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    st.write("**Confusion Matrix:**")
    st.dataframe(cm_df)

    st.write("### Classification Report:")
    st.text(class_report)  # Display classification report
