from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

repo_id = "SantoshS23/TourismPackage"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Build relative workspace path
workspace = Path(os.getenv("GITHUB_WORKSPACE", "."))
data_dir = workspace / "tourism_project" / "data"

api.upload_folder(
    folder_path= os.path.join(tourism_project, 'data'),
    repo_id=repo_id,
    repo_type=repo_type,
)
