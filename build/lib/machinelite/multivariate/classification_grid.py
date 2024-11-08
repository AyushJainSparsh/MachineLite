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

# Import metrics for evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Import tool for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier

# Initialize a list of classification models with default or specified hyperparameters
models = [
    LogisticRegression(),
    RidgeClassifier(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    MLPClassifier(),
    MultinomialNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    CatBoostClassifier()
]

# Define parameter grids for hyperparameter tuning of each model
param_grids = {
    'LogisticRegression': {
        'multi_class': ['multinomial'],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs'],
        'max_iter': [100, 200, 300]
    },
    'RidgeClassifier': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    'SVC': {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1, 10],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'BaggingClassifier': {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'max_iter': [200, 300, 400]
    },
    'MultinomialNB': {
        'alpha': [0.01, 0.1, 1, 10],
        'fit_prior': [True, False]
    },
    'LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
    },
    'QuadraticDiscriminantAnalysis': {
        'reg_param': [0.0, 0.1, 0.5, 0.9]
    },
    'CatBoostClassifier': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'bagging_temperature': [0, 0.5, 1, 2]
    }
}

# Function to perform classification using multiple models and hyperparameter tuning
def ClassificationGrid(X_train, X_test, y_train, y_test, models=models, param_grids=param_grids):
    """
    Function to perform classification using multiple models with hyperparameter tuning
    Args:
    X_train: Training data features
    y_train: Training data labels
    X_test: Test data features
    y_test: Test data labels
    models: List of models to be used for classification
    param_grids: Parameter grids for hyperparameter tuning
    
    Returns:
    Prints the evaluation metrics for each model
    """
    # Loop through each model, fit on training data, predict on test data, and compute metrics
    for i, model in enumerate(models):
        
        # Initialize MultiOutputClassifier with the current model
        multi_output_classifier = MultiOutputClassifier(model)

        # Initialize GridSearchCV with the current model and its parameter grid
        grid = GridSearchCV(multi_output_classifier, param_grid=param_grids[model.__class__.__name__], cv=5, scoring='accuracy', n_jobs=-1)
        
        # Fit the model on the training data
        grid.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = grid.predict(X_test)
        
        # Print model evaluation results
        print("\n------------------------------------------------------------------------------------\n")
        print("Model:", model.__class__.__name__)
        print("Best Params:", grid.best_params_)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Precision Score:", precision_score(y_test, y_pred, average='weighted'))
        print("Recall Score:", recall_score(y_test, y_pred, average='weighted'))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\n------------------------------------------------------------------------------------")
