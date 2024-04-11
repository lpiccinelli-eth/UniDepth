from unidepth.models import UniDepthV1HF, UniDepthV1
from huggingface_hub import hf_hub_download, upload_file
import torch
import json

with open("./configs/config_v1_cnvnxtl.json") as f:
    config = json.load(f)
print(config)

model = UniDepthV1HF(config)

filepath = hf_hub_download(repo_id="lpiccinelli/UniDepth", filename="unidepth_v1_cnvnxtl.bin", repo_type="model")
info = model.load_state_dict(torch.load(filepath, map_location="cpu"), strict=False)
print("UniDepthV1_ConvNextL is loaded with:")
print(f"\tmissing keys: {info.missing_keys}\n\tadditional keys: {info.unexpected_keys}")

# model.push_to_hub("lpiccinelli/unidepth-v1-cnvnxtl", config=config, commit_message="Initial commit")
upload_file(
    path_or_fileobj=filepath,
    path_in_repo="pytorch_model.bin",
    repo_id="lpiccinelli/unidepth-v1-cnvnxtl",
    commit_message="Initial commit"
)
reloaded_model = UniDepthV1HF.from_pretrained(backbone="ConvNextL")