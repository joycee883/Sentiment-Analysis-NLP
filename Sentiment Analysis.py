# Importing necessary libraries
import pandas as pd  # For data manipulation and handling
import numpy as np  # For numerical computations
import re  # For regular expressions to clean text
import nltk  # Natural Language Toolkit for NLP-related operations
import pickle  # For saving models and data

# Load the dataset
dataset = pd.read_csv(r"C:\Users\91939\Desktop\AI&DS\Data science projects\NLP Projects\SentimentAnalysis-NLP,ML\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)


# Downloading English stopwords
nltk.download('stopwords')  
from nltk.corpus import stopwords  # List of common stopwords
from nltk.stem.porter import PorterStemmer  # For stemming words

# Text Preprocessing - Cleaning the text data
corpus = []  # To store preprocessed text data
for i in range(len(dataset)):  # Process the entire dataset
    # Removing any characters except alphabets and converting text to lowercase
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()  # Splitting text into individual words
    
    # Initializing stemmer
    ps = PorterStemmer()
    # Stemming words and removing stopwords
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    # Joining words back into a single string
    review = ' '.join(review)
    corpus.append(review)  # Adding preprocessed review to the corpus

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()  # Initializing the CountVectorizer
X = cv.fit_transform(corpus).toarray()  # Converting text data into numerical format
y = dataset.iloc[:, 1].values  # Target variable (Liked) indicating positive/negative reviews

# Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Import necessary libraries for classification models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Dictionary to hold different classifiers including ensemble methods
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Hyperparameter tuning parameters
param_grids = {
    'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'Logistic Regression': {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'saga']},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
    'Support Vector Classifier': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
    'AdaBoost': {'n_estimators': [50, 100, 200]},
    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Loop through each classifier for hyperparameter tuning and print evaluation metrics
for name, classifier in classifiers.items():
    print(f"\n{name} Results:")
    
    # Hyperparameter tuning
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grids[name], cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Best parameters and estimator
    best_classifier = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    
    # Predict the test set results
    y_pred = best_classifier.predict(X_test)
    
    # Evaluate the model with a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Calculate the accuracy score
    ac = accuracy_score(y_test, y_pred)
    print("Accuracy:", ac)

    # Additional metrics: Precision, Recall, F1-score
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Check for bias and variance
    bias = best_classifier.score(X_train, y_train)  # Model score on training data
    print("Bias (Train Accuracy):", bias)

    variance = best_classifier.score(X_test, y_test)  # Model score on test data
    print("Variance (Test Accuracy):", variance)

    # Save the best classifier
    with open(f"{name}_model.pkl", "wb") as file: 
        pickle.dump(best_classifier, file)

# Save CountVectorizer to use in the app
with open("count_vectorizer.pkl", "wb") as file:
    pickle.dump(cv, file)

print("Models and vectorizer saved successfully.")

# Save the train and test data splits as pickle files
with open("X_train.pkl", "wb") as file:
    pickle.dump(X_train, file)
with open("X_test.pkl", "wb") as file:
    pickle.dump(X_test, file)
with open("y_train.pkl", "wb") as file:
    pickle.dump(y_train, file)
with open("y_test.pkl", "wb") as file:
    pickle.dump(y_test, file)

print("Train and test data splits saved successfully.")






