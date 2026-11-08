import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not set.")
api = HfApi(token=token)
user = api.whoami()
print("Remote whoami:", user["name"], user.get("email"))

