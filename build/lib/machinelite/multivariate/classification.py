import pandas as pd

# Import necessary libraries for various classification models

# Linear Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Support Vector Machines
from sklearn.svm import SVC

# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Tree-Based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Ensemble Methods
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

# Neural Networks
from sklearn.neural_network import MLPClassifier

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Specialized Libraries
from catboost import CatBoostClassifier

# Multi-output Classification Wrapper
from sklearn.multioutput import MultiOutputClassifier

# Import metrics for evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Initialize a list of classification models with default or specified hyperparameters
models = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs'),
    RidgeClassifier(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, solver='adam', random_state=42),
    MultinomialNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    CatBoostClassifier()
]

# Define the classification function
def Classification(X_train, X_test, y_train, y_test, models=models):
    """
    Function to perform multi-output classification using various models
    Args:
    X_train: Training data features
    y_train: Training data labels
    X_test: Test data features
    y_test: Test data labels
    models: List of models to be used for classification
    
    Returns:
    Prints the evaluation metrics for each model
    """
    # Loop through each model, fit on training data, predict on test data, and compute metrics
    for model in models:
        # Initialize MultiOutputClassifier with the current model
        multi_output_classifier = MultiOutputClassifier(model)

        # Fit the model to the training data
        multi_output_classifier.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = multi_output_classifier.predict(X_test)
        
        # Print model evaluation results
        print("\n------------------------------------------------------------------------------------\n")
        print("Model:", model.__class__.__name__)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Precision Score:", precision_score(y_test, y_pred, average='weighted'))
        print("Recall Score:", recall_score(y_test, y_pred, average='weighted'))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\n------------------------------------------------------------------------------------")

# Example usage:
# X_train, X_test, y_train, y_test = ... # Define your data
# Classification(X_train, y_train, X_test, y_test)
