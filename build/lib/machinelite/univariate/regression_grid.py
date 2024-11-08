import pandas as pd

# Import necessary libraries for various regression models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , root_mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

# Initialize a list of regression models with default or specified hyperparameters
models = [
    LinearRegression(),
    Lasso(),
    Ridge(),
    ElasticNet(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    SVR(),
    KNeighborsRegressor(),
    CatBoostRegressor()
]

# Define parameter grids for hyperparameter tuning of each model
param_grids = {
    'LinearRegression': {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'max_iter': [1000, 2000, 3000]
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'max_iter': [1000, 2000, 3000]
    },
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'max_iter': [1000, 2000, 3000]
    },
    'DecisionTreeRegressor': {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    },
    'RandomForestRegressor': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    },
    'AdaBoostRegressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1, 10],
        'loss': ['linear', 'square', 'exponential']
    },
    'SVR': {
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    },
    'KNeighborsRegressor': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'CatBoostRegressor': {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [100, 200, 300],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'bagging_temperature': [0, 0.5, 1, 2],
        'random_strength': [0.5, 1, 1.5, 2]
    }
}

# Define the regression function
def RegressionGrid(X_train, X_test, y_train, y_test, models=models, param_grids=param_grids):
    """
    Function to perform multi-output regression using various models with hyperparameter tuning
    Args:
    X_train: Training data features
    y_train: Training data labels
    X_test: Test data features
    y_test: Test data labels
    models: List of models to be used for regression
    param_grids: Parameter grids for hyperparameter tuning
    
    Returns:
    A DataFrame containing the evaluation metrics for each model
    """
    # Initialize lists to store evaluation metrics
    mse = []
    mae = []
    rmse = []
    score = []
    best_params_ = []
    
    # Loop through each model, fit on training data, predict on test data, and compute metrics
    for model in models:
        
        # Initialize GridSearchCV with the current model and its parameter grid
        grid = GridSearchCV(estimator=model, param_grid=param_grids[model.__class__.__name__], cv=5, scoring='r2')
        
        # Fit the model on the training data
        grid.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = grid.predict(X_test)
        
        # Compute and store the evaluation metrics
        mse.append(mean_squared_error(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        rmse.append(root_mean_squared_error(y_test, y_pred))
        score.append(r2_score(y_test, y_pred))
        best_params_.append(grid.best_params_)
    
    # Create a DataFrame to store and display the results
    output = {
        "Model": [model.__class__.__name__ for model in models],
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "r2_score": score,
        "best_params_": best_params_
    }
    
    return pd.DataFrame(output)

# Example usage:
# X_train, X_test, y_train, y_test = ... # Define your data
# results = RegressionGrid(X_train, y_train, X_test, y_test)
# print(results)
