import os
from huggingface_hub import hf_hub_url
os.system("mkdir -p out/threedfront_objfeat_vqvae/checkpoints")
url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="threedfront_objfeat_vqvae_epoch_01999.pth", repo_type="dataset")
os.system(f"wget {url} -O out/threedfront_objfeat_vqvae/checkpoints/epoch_01999.pth")
url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="objfeat_bounds.pkl", repo_type="dataset")
os.system(f"wget {url} -O out/threedfront_objfeat_vqvae/objfeat_bounds.pkl")