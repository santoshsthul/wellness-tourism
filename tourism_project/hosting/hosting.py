
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

deployment_folder_path = os.path.join(tourism_project , 'deployment')

api.upload_folder(
    folder_path=deployment_folder_path,     # the local folder containing your files
    repo_id="SantoshS23/wellness-tourism-model",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
