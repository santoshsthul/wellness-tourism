
# for data manipulation
import pandas as pd
import numpy as np # Import numpy for sqrt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("https://abhorrently-uncongregated-glayds.ngrok-free.dev") # Use the public_url variable
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

Xtrain_path = "https://huggingface.co/datasets/SantoshS23/TourismPackage/X_train.csv"
Xtest_path = "https://huggingface.co/datasets/SantoshS23/TourismPackage/X_test.csv"
ytrain_path = "https://huggingface.co/datasets/SantoshS23/TourismPackage/y_train.csv"
ytest_path = "https://huggingface.co/datasets/SantoshS23/TourismPackage/y_test.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)


# Define numeric and categorical features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch',
    'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
   'TypeofContact','Occupation', 'Gender', 'ProductPitched', 'MaritalStatus','Designation'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__subsample': [0.7, 0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbregressor__reg_lambda': [0.1, 1, 10]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "train_R2": train_r2,
        "test_R2": test_r2
    })

    # Save the model locally
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "SantoshS23/tourism_package_prediction_model" # Updated repo_id
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_package_prediction_model_v1.joblib", # Updated filename
        path_in_repo="best_tourism_package_prediction_model_v1.joblib",   # Updated filename
        repo_id=repo_id,
        repo_type=repo_type,
    )
