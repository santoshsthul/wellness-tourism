import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# Validate HF token
token = os.getenv("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN environment variable not set. Add it as a GitHub Actions secret (Settings → Secrets and variables → Actions) named HF_TOKEN.")

# Initialize API client and verify token
api = HfApi(token=token)
try:
    api.whoami()
except HfHubHTTPError as e:
    sys.exit(f"HF_TOKEN failed authentication: {e}")

# Constants
repo_id = "SantoshS23/TourismPackage"  # change if your dataset repo is different
dataset_filename = "tourism.csv"

# Download dataset file from Hugging Face dataset repo
try:
    local_dataset_path = hf_hub_download(
        repo_id=repo_id,
        filename=dataset_filename,
        repo_type="dataset",
        token=token,
    )
except Exception as e:
    sys.exit(f"Failed to download {dataset_filename} from {repo_id}: {e}")

# Read dataset
try:
    df = pd.read_csv(local_dataset_path)
    print("Dataset loaded successfully from:", local_dataset_path)
except Exception as e:
    sys.exit(f"Failed to read dataset CSV: {e}")

# Basic preprocessing (existing logic)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

label_encoder = LabelEncoder()
for col in ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

target_col = 'ProdTaken'
if target_col not in df.columns:
    sys.exit(f"Target column '{target_col}' not found in dataset")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save files to workspace
workspace = os.getenv("GITHUB_WORKSPACE", ".")
out_dir = os.path.join(workspace, "tourism_project", "model_building", "output")
os.makedirs(out_dir, exist_ok=True)

X_train_path = os.path.join(out_dir, "X_train.csv")
X_test_path = os.path.join(out_dir, "X_test.csv")
y_train_path = os.path.join(out_dir, "y_train.csv")
y_test_path = os.path.join(out_dir, "y_test.csv")

X_train.to_csv(X_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_train.to_csv(y_train_path, index=False)
y_test.to_csv(y_test_path, index=False)

print("Saved split files to:", out_dir)

# Upload generated files back to the dataset repo
files = [
    (X_train_path, "X_train.csv"),
    (X_test_path, "X_test.csv"),
    (y_train_path, "y_train.csv"),
    (y_test_path, "y_test.csv"),
]

for local_path, remote_name in files:
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"Uploaded {remote_name} to {repo_id}")
    except Exception as e:
        sys.exit(f"Failed to upload {remote_name} to {repo_id}: {e}")
