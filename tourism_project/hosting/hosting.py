import os
import sys
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Validate HF token
token = os.getenv("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN environment variable not set. Add it as a GitHub Actions secret (Settings → Secrets and variables → Actions) named HF_TOKEN.")

# Init API client and verify token
api = HfApi(token=token)
try:
    api.whoami()
except HfHubHTTPError as e:
    sys.exit(f"HF_TOKEN failed authentication: {e}")

# Build deployment folder path relative to repository workspace
workspace = os.getenv("GITHUB_WORKSPACE", ".")
deployment_folder_path = os.path.join(workspace, 'tourism_project', 'deployment')

if not os.path.isdir(deployment_folder_path):
    sys.exit(f"Deployment folder not found at {deployment_folder_path}")

try:
    api.upload_folder(
        folder_path=deployment_folder_path,
        repo_id="SantoshS23/wellness-tourism-model",
        repo_type="space",
        path_in_repo="",
        token=token,
    )
    print("Deployment folder uploaded successfully.")
except Exception as e:
    sys.exit(f"Failed to upload deployment folder to Hugging Face: {e}")
