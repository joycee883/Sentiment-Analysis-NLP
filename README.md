# 🎯Project Spotlight: Sentiment Analysis Application

## 📋Project Overview

The Sentiment Analysis Application classifies user-entered reviews as either positive or negative. Built with Streamlit for an interactive experience, this application uses pre-trained machine learning models to predict sentiment and includes multiple evaluation metrics to assess model performance in real-time.

## 🛠️Tools Used

Programming Language: Python <br> Libraries: <br>
* Pandas & NumPy: For efficient data handling and preprocessing <br>
* Scikit-learn: For model training, testing, and performance evaluation <br>
* NLTK: Text cleaning and stemming for natural language processing <br>
* Streamlit: Developing an intuitive user interface <br>

## 🔍Key Steps

1. Data Preprocessing & Feature Engineering <br>
* Text Cleaning: Removed special characters, converted text to lowercase, and applied stemming. <br>
* Stopword Removal: Filtered out common stopwords to retain essential words. <br>
* CountVectorizer: Converted cleaned text into numerical data for training and predictions. <br>

2. Model Selection & Training <br>
Trained various classifiers to identify the most accurate model for sentiment analysis: <br>
* Decision Tree <br>
* K-Nearest Neighbors <br>
* Logistic Regression <br>
* Random Forest <br>
* AdaBoost <br>
* Gradient Boosting <br>
* Support Vector Classifier <br>
Saved models as .pkl files for easy integration and selection. <br>

3. Model Evaluation & Metrics <br>
Used several metrics to evaluate model performance: <br>
* Accuracy: Measures overall prediction correctness. <br>
* Bias & Variance: Training and test accuracies to assess generalizability. <br>
* Confusion Matrix: Visualizes true and false positive/negative rates. <br>
* Classification Report: Includes Precision, Recall, and F1-score for each class. <br>

4. Frontend with Streamlit <br>
* Interactive Elements: Users can enter a review and select a classifier for real-time predictions. <br>
* Model Comparison: Displays accuracy and bias-variance analysis, helping users understand model strengths. <br>
* Visualizations: Confusion matrix and classification report give insights into model performance. <br>

## 📊Key Findings

Best Performing Model: Logistic Regression achieved reliable accuracy, balancing efficiency and simplicity. <br>
Additional Insights: Decision Tree and Random Forest performed well but tended to overfit training data. <br>

## 🌐Applications

The Sentiment Analysis App is versatile and can be applied in multiple contexts: <br>
* Customer Review Analysis: Helps businesses gauge sentiment in customer feedback. <br>
* Social Media Monitoring: Assists social media managers in tracking sentiment trends. <br>
* Market Research: Aids in assessing public opinion on products or events. <br>

## 🔮Future Improvements

* Advanced Preprocessing: Adding techniques like TfidfVectorizer for improved feature extraction. <br>
* Expanded Sentiment Classes: Including neutral sentiment for more nuanced classification. <br>
* Visualization Over Time: Enabling insights into evolving sentiment trends. <br>


Try out the Sentiment Analysis App and explore different models for real-time sentiment predictions!
