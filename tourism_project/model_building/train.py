import os
import pandas as pd
import numpy as np
from io import StringIO

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import mlflow

# huggingface_hub imports
from huggingface_hub import hf_hub_download, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# MLflow tracking URI (use env or default)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("mlops-training-experiment")

HF_TOKEN = os.getenv("HF_TOKEN")

def download_csv_from_dataset(repo_id: str, filename: str, token: str | None = None, repo_type: str = "dataset") -> pd.DataFrame:
    """
    Download a file from a Hugging Face dataset repo using hf_hub_download and return a DataFrame.
    Uses the token if provided, and benefits from local caching.
    """
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token=token
        )
    except Exception as e:
        # Provide actionable error messages for auth issues
        err = str(e)
        if "401" in err or "Unauthorized" in err or "403" in err:
            raise RuntimeError(
                f"Could not download {filename} from {repo_id}: Unauthorized (401/403). "
                "Set HF_TOKEN (Actions secret) with a token that has access to the dataset."
            ) from e
        raise

    return pd.read_csv(local_path)

# Dataset repo info (map from your previous URLs)
dataset_repo = "SantoshS23/TourismPackage"  # repo_id for hf_hub_download
X_train = download_csv_from_dataset(dataset_repo, "X_train.csv", token=HF_TOKEN)
X_test = download_csv_from_dataset(dataset_repo, "X_test.csv", token=HF_TOKEN)
y_train = download_csv_from_dataset(dataset_repo, "y_train.csv", token=HF_TOKEN)
y_test = download_csv_from_dataset(dataset_repo, "y_test.csv", token=HF_TOKEN)

# (rest of training pipeline unchanged)
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch',
    'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
   'TypeofContact','Occupation', 'Gender', 'ProductPitched', 'MaritalStatus','Designation'
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__subsample': [0.7, 0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbregressor__reg_lambda': [0.1, 1, 10]
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    mlflow.log_metrics({
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "train_R2": train_r2,
        "test_R2": test_r2
    })

    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face model repo (only if HF_TOKEN provided)
    if not HF_TOKEN:
        print("HF_TOKEN not found in environment; skipping upload to Hugging Face. "
              "Set the HF_TOKEN secret in your repository if you want automatic uploads.")
    else:
        api = HfApi()
        try:
            who = api.whoami(token=HF_TOKEN)
            hf_user = who.get("name") or who.get("user", {}).get("name") or who.get("username") or who.get("login")
            repo_id = f"{hf_user}/tourism_package_prediction_model" if hf_user else "SantoshS23/tourism_package_prediction_model"
        except Exception:
            repo_id = "SantoshS23/tourism_package_prediction_model"

        repo_type = "model"

        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type, token=HF_TOKEN)
            print(f"Repository '{repo_id}' already exists. Using it.")
        except RepositoryNotFoundError:
            try:
                create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
                print(f"Repository '{repo_id}' created.")
            except HfHubHTTPError as e:
                raise RuntimeError(
                    f"Failed to create repository '{repo_id}' on Hugging Face Hub: {e}. "
                    "Check that the token has 'repo' scope and that the account has permission."
                ) from e

        try:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=repo_id,
                repo_type=repo_type,
                token=HF_TOKEN,
            )
            print(f"Uploaded {model_path} to {repo_id}")
        except HfHubHTTPError as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                raise RuntimeError(
                    "Upload failed with HTTP 401 Unauthorized. Ensure HF_TOKEN is valid "
                    "and has 'repo' (write) permissions. Recreate token at https://huggingface.co/settings/tokens"
                ) from e
            raise
