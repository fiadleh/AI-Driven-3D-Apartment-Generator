import hashlib
import os
import argparse
import random
import pickle
from copy import deepcopy

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

# to solve no src issue
from pathlib import Path
import sys
#sys.path.append('/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/')
sys.path.append(str(Path(__file__).parent.parent)+'/')


from src.utils.util import *
from src.utils.visualize import *
from src.data import filter_function, get_dataset_raw_and_encoded, get_encoded_dataset
from src.data.threed_future_dataset import ThreedFutureDataset
from src.data.threed_front_dataset_base import trs_to_corners
from src.data.utils_text import compute_loc_rel, reverse_rel, fill_templates
from src.models import model_from_config, ObjectFeatureVQVAE
from src.models.sg2sc_diffusion import Sg2ScDiffusion
from src.models.sg_diffusion_vq_objfeat import scatter_trilist_to_matrix
from src.models.clip_encoders import CLIPTextEncoder




print('gen 3d start')
parser = argparse.ArgumentParser(
    description="Generate scenes using two previously trained models"
)


parser.add_argument(
    "--tag",
    type=str,
    required=False,
    help="Tag that refers to the current experiment"
)
parser.add_argument(
    "--fvqvae_tag",
    type=str,
    required=False,
    help="Tag that refers to the fVQ-VAE experiment"
)
parser.add_argument(
    "--fvqvae_epoch",
    type=int,
    default=1999, #1999
    help="Epoch of the pretrained fVQ-VAE"
)
parser.add_argument(
    "--sg2sc_tag",
    type=str,
    required=False,
    help="Tag that refers to the Sg2Sc experiment"
)
parser.add_argument(
    "--sg2sc_epoch",
    type=int,
    default=0, #1999
    help="Epoch of the pretrained Sg2Sc model"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="out",
    help="Path to the output directory"
)
parser.add_argument(
    "--checkpoint_epoch",
    type=int,
    default=None,
    help="The epoch to load the checkpoint from"
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=4,
    help="The number of processed spawned by the batch provider (default=0)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for the PRNG (default=0)"
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=1,
    help="The number of epochs for evaluation"  # descriptions are sampled randomly each epoch
)
parser.add_argument(
    "--n_scenes",
    type=int,
    default=9999, #100
    help="The number of scenes to be generated"
)
parser.add_argument(
    "--condition_type",
    type=str,
    default="text",
    choices=["text", "none"],
    help="The type of the CLIP embedded conditioning (choices: `text`, `none`)"
)
parser.add_argument(
    "--draw_scene_graph",
    action="store_true",
    help="Draw generated scene graphs"
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="Visualize the generated scenes"
)
parser.add_argument(
    "--eight_views",
    action="store_true",
    help="Render 8 views of the scene"
)
parser.add_argument(
    "--resolution",
    type=int,
    default=256,
    help="Resolution of the rendered image"
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print more information"
)
parser.add_argument(
    "--device",
    type=str,
    default="0",
    help="Device to use for training"
)
parser.add_argument(
    "--cfg_scale",
    type=float,
    default=1.,
    help="scale for the classifier-free guidance"
)
parser.add_argument(
    "--sg2sc_cfg_scale",
    type=float,
    default=1.,
    help="scale for the classifier-free guidance in sg2sc model"
)

parser.add_argument(
    "--room_prompt",
    type=str,
    help="Textual description of the wanted room"
)

args = parser.parse_args()

# Set the random seed
if args.seed is not None and args.seed >= 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"You have chosen to seed([{args.seed}]) the experiment")

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
else:
    device = torch.device("cpu")
print(f"Run code on device [{device}]\n")

args.fvqvae_tag = "threedfront_objfeat_vqvae" 

args.tag = "bedroom_sgdiffusion_vq_objfeat"
args.sg2sc_tag = "bedroom_sg2scdiffusion_objfeat" 
args_config_file = "/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/configs/bedroom_sg_diffusion_vq_objfeat.yaml"


#args.tag = "livingroom_sgdiffusion_vq_objfeat"
#args.sg2sc_tag = "livingroom_sg2scdiffusion_objfeat" 
#args_config_file = "/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/configs/livingroom_sg_diffusion_vq_objfeat.yaml"

args.tag = "diningroom_sgdiffusion_vq_objfeat"
args.sg2sc_tag = "diningroom_sg2scdiffusion_objfeat" 
args_config_file = "/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/configs/diningroom_sg_diffusion_vq_objfeat.yaml"


args_output_dir ="/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/out"
exp_dir = os.path.join(args_output_dir, args.tag)
ckpt_dir = os.path.join(exp_dir, "checkpoints")

# Check if `ckpt_dir` exists
config = load_config(args_config_file)

# Build the dataset of 3D models
objects_dataset = ThreedFutureDataset.from_pickled_dataset(
    config["data"]["path_to_pickled_3d_futute_models"]
)
print(f"Load [{len(objects_dataset)}] 3D-FUTURE models")

# Compute the bounds for this experiment, save them to a file in the
# experiment directory and pass them to the validation dataset
if not os.path.exists(os.path.join(exp_dir, "bounds.npz")):
    train_dataset = get_encoded_dataset(
        config["data"],  # same encoding type as validation dataset
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=None,  # no need for prior statistics computation
        split=config["training"].get("splits", ["train", "val"])
    )
    np.savez(
        os.path.join(exp_dir, "bounds.npz"),
        translations=train_dataset.bounds["translations"],
        sizes=train_dataset.bounds["sizes"],
        angles=train_dataset.bounds["angles"]
    )
    print(f"Training set has bounds: {train_dataset.bounds}")
    print(f"Load [{len(train_dataset)}] training scenes with [{train_dataset.n_object_types}] object types\n")

config["data"]["encoding_type"] += "_sincos_angle"  # for sg2sc diffusion postprocessing
if "eval" not in config["data"]["encoding_type"]: config["data"]["encoding_type"] += "_eval"

raw_dataset, dataset = get_dataset_raw_and_encoded(
    config["data"],
    filter_fn=filter_function(
        config["data"],
        split=config["validation"].get("splits", ["test"])
    ),
    path_to_bounds=os.path.join(exp_dir, "bounds.npz"),
    augmentations=None,
    split=config["validation"].get("splits", ["test"])
)
print(f"Load [{len(dataset)}] validation scenes with [{dataset.n_object_types}] object types\n")

if args.n_scenes == 0:  # use all scenes
    B = config["validation"]["batch_size"]
else:
    B = args.n_scenes
dataloader = DataLoader(
    dataset,
    batch_size=B,
    num_workers=args.n_workers,
    pin_memory=False,
    collate_fn=dataset.collate_fn,
    shuffle=False
)

print(f"Load pretrained text encoder [{config['model']['text_encoder']}]\n")
if "clip" in config["model"]["text_encoder"]:
    text_encoder = CLIPTextEncoder(config["model"]["text_encoder"], device=device)
else:
    raise ValueError(f"Invalid text encoder name: [{config['model']['text_encoder']}]")

# Load pretrained VQ-VAE codebook weights
print("Load pretrained VQ-VAE\n")
with open(f"{args.output_dir}/{args.fvqvae_tag}/objfeat_bounds.pkl", "rb") as f:
    kwargs = pickle.load(f)
vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
ckpt_path = f"{args.output_dir}/{args.fvqvae_tag}/checkpoints/epoch_{args.fvqvae_epoch:05d}.pth"
vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
vqvae_model = vqvae_model.to(device)
vqvae_model.eval()

# Initialize the model
model = model_from_config(
    config["model"],
    dataset.n_object_types,
    dataset.n_predicate_types,
    text_emb_dim=text_encoder.text_emb_dim
).to(device)

# Create EMA for the model
ema_config = config["training"]["ema"]
if ema_config["use_ema"]:
    ema_states = EMAModel(model.parameters())
    ema_states.to(device)
else:
    ema_states: EMAModel = None

# Load the weights from a checkpoint
load_epoch = load_checkpoints(model, ckpt_dir, ema_states, epoch=args.checkpoint_epoch, device=device)

# Evaluate with the EMA parameters if specified
if ema_states is not None:
    print(f"Copy EMA parameters to the model\n")
    ema_states.copy_to(model.parameters())
model.eval()

# Initialize the sg2sc model
sg2sc_model = Sg2ScDiffusion(
    dataset.n_object_types,
    dataset.n_predicate_types,
    use_objfeat="objfeat" in config["model"]["name"]
).to(device)

# Create EMA for the sg2sc model
ema_config = config["training"]["ema"]
if ema_config["use_ema"]:
    sg2sc_ema_states = EMAModel(model.parameters())
    sg2sc_ema_states.to(device)
else:
    sg2sc_ema_states: EMAModel = None

sg2sc_load_epoch = load_checkpoints(
    sg2sc_model,
    f"{args.output_dir}/{args.sg2sc_tag}/checkpoints",
    sg2sc_ema_states,
    epoch=args.sg2sc_epoch, device=device
)

# Evaluate with the sg2sc EMA parameters if specified
if sg2sc_ema_states is not None:
    print(f"Copy EMA parameters to the sg2sc model\n")
    sg2sc_ema_states.copy_to(sg2sc_model.parameters())
sg2sc_model.eval()

# Check if `save_dir` exists and if it doesn't create it
#save_dir = os.path.join(exp_dir, "generated_scenes", f"epoch_{load_epoch:05d}")
save_dir = os.path.join(exp_dir, "generated_scenes", str(hashlib.md5("args_room_prompt".encode('utf-8')).hexdigest()))

os.makedirs(save_dir, exist_ok=True)

# Generate the scene graphs and then boxes
classes = np.array(dataset.object_types)
rel_counts, correct_rel_counts, correct_rel_counts_sg = 1e-9, 0, 0
correct_easy_rel_counts, correct_easy_rel_counts_sg = 0, 0
print("Sample scene graphs with the generative model")
args.n_epochs = 20
for epoch in range(args.n_epochs):
    #print(dataloader)
    #exit()
    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        desc=f"[{epoch:2d}/{args.n_epochs:2d}] Process each batch",
        total=len(dataloader), ncols=125,
        disable=True
    ):
        # Prepare CLIP text embeddings
        descriptions = batch["descriptions"]
        texts = []
        batch_selected_relations, batch_selected_descs = [], []
        for desc_idx, desc in enumerate(descriptions):
            text, selected_relations, selected_descs = fill_templates(desc,
                dataset.object_types,
                dataset.predicate_types,
                batch["object_descs"][desc_idx],
                seed=epoch * len(dataset) + batch_idx * B + desc_idx
            )
            texts.append(text)  # a batch of texts
            batch_selected_relations.append(selected_relations)
            batch_selected_descs.append(selected_descs)
        print(texts, '\n******************************************')
    exit()