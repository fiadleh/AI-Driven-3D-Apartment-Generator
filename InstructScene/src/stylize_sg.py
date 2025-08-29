import os
import argparse
import random
import pickle

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

from src.utils.util import *
from src.utils.visualize import *
from src.data import filter_function, get_dataset_raw_and_encoded, get_encoded_dataset
from src.data.threed_future_dataset import ThreedFutureDataset
from src.models import model_from_config, ObjectFeatureVQVAE
from src.models.sg_diffusion_vq_objfeat import scatter_trilist_to_matrix
from src.models.clip_encoders import CLIPTextEncoder


def main():
    parser = argparse.ArgumentParser(
        description="Stylize scenes using two previously trained models"
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--fvqvae_tag",
        type=str,
        required=True,
        help="Tag that refers to the fVQ-VAE experiment"
    )
    parser.add_argument(
        "--fvqvae_epoch",
        type=int,
        default=1999,
        help="Epoch of the pretrained fVQ-VAE"
    )
    parser.add_argument(
        "--sg2sc_tag",
        type=str,
        required=True,
        help="Tag that refers to the Sg2Sc experiment"
    )
    parser.add_argument(
        "--sg2sc_epoch",
        type=int,
        default=1999,
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
        default=5,
        help="The number of scenes to be generated"
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
        "--visualize_original",
        action="store_true",
        help="Visualize the original scenes"
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

    # Check if `ckpt_dir` exists
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    assert os.path.exists(ckpt_dir), f"Checkpoint directory {ckpt_dir} does not exist"

    config = load_config(args.config_file)

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

    # Check if `save_dir` exists and if it doesn't create it
    save_dir = os.path.join(exp_dir, "generated_scenes", f"epoch_{load_epoch:05d}")
    os.makedirs(save_dir, exist_ok=True)

    # Generate the scene graphs and then boxes
    classes = np.array(dataset.object_types)
    templates = [
        "Make the room {} style.", "Make objects in the room {}", "Let the room be in {} style",
        "Make the room style {}.", "Make the room {}.", "Let objects be in {}.",
    ]
    styles = ["black", "white", "gray", "brown"]
    clip_cossims = []
    print("Stylize scene graphs with the generative model")
    for epoch in range(args.n_epochs):
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc=f"[{epoch:2d}/{args.n_epochs:2d}] Process each batch",
            total=len(dataloader), ncols=125,
            disable=args.verbose
        ):
            # Move everything to the device
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device)
            # Prepare CLIP text embeddings
            np.random.seed(epoch * len(dataloader) + batch_idx)
            selected_styles = list(np.random.choice(styles, size=len(batch["scene_uids"])))
            texts = [np.random.choice(templates).format(style) for style in selected_styles]
            text_last_hidden_state, text_embeds = text_encoder(texts)

            # 1. Generate the graph nodes and edges
            with torch.no_grad():
                objs, edges, objfeat_vq_indices = model.stylize(
                    batch, text_last_hidden_state, text_embeds,
                    filter_ratio=0.25,
                    cfg_scale=args.cfg_scale
                )
                if objfeat_vq_indices is not None:
                    # Replace the empty token with a random token within the vocabulary
                    objfeat_vq_indices_rand = torch.randint_like(objfeat_vq_indices, 0, 64)
                    objfeat_vq_indices[objfeat_vq_indices == 64] = objfeat_vq_indices_rand[objfeat_vq_indices == 64]

            # 2. Generate boxes from the scene graphs
            # Prepare the scene graph for the sg2sc model
            objs = objs.argmax(dim=-1)  # (bs, n)
            obj_masks = (objs != dataset.n_object_types).long()  # (bs, n)
            # Mask and symmetrize edges
            edges = edges.argmax(dim=-1)  # (bs, n*(n-1)/2)
            edges = F.one_hot(edges, num_classes=dataset.n_predicate_types+1).float()
            edges = scatter_trilist_to_matrix(edges, objs.shape[-1])  # (bs, n, n, n_pred_types+1)
            e_mask1 = obj_masks.unsqueeze(1).unsqueeze(-1)  # (bs, 1, n, 1)
            e_mask2 = obj_masks.unsqueeze(2).unsqueeze(-1)  # (bs, n, 1, 1)
            edges = edges * e_mask1 * e_mask2  # mask out edges to non-existent objects
            edges_negative = edges[..., 
                [*range(dataset.n_predicate_types//2, dataset.n_predicate_types)] + \
                [*range(0, dataset.n_predicate_types//2)] + \
                [*range(dataset.n_predicate_types, edges.shape[-1])]
            ]  # (bs, n, n, n_pred_types+1)
            edges = edges + edges_negative.permute(0, 2, 1, 3)
            edge_mask = torch.eye(objs.shape[-1], device=device).bool().unsqueeze(0).unsqueeze(-1)  # (1, n, n, 1)
            edge_mask = ((~edge_mask).float() * e_mask1 * e_mask2).squeeze(-1)  # (bs, n, n)
            assert torch.all(edges.sum(dim=-1) == edge_mask)  # every edge is one-hot encoded, except for the diagonal and empty nodes
            edges_empty = edges[edges.sum(dim=-1) == 0]
            edges_empty[..., -1] = 1.
            edges[edges.sum(dim=-1) == 0] = edges_empty  # set the empty edges to the last class
            edges = torch.argmax(edges, dim=-1)  # (bs, n, n)

            if objfeat_vq_indices is not None:
                # Decode objfeat indices to objfeat embeddings
                BB, N = objfeat_vq_indices.shape[:2]
                objfeats = vqvae_model.reconstruct_from_indices(objfeat_vq_indices.reshape(BB*N, -1)).reshape(BB, N, -1)  # (BB, N, D)
                objfeats = objfeats.cpu().numpy()
                # Replace the empty token with a random token within the vocabulary
                objfeat_vq_indices_original = batch["objfeat_vq_indices"].reshape(BB, N, -1)
                objfeat_vq_indices_rand = torch.randint_like(objfeat_vq_indices, 0, 64)
                objfeat_vq_indices_original[objfeat_vq_indices_original == 64] = objfeat_vq_indices_rand[objfeat_vq_indices_original == 64]
                objfeats_original = vqvae_model.reconstruct_from_indices(objfeat_vq_indices_original.reshape(BB*N, -1)).reshape(BB, N, -1)  # (BB, N, D)
                objfeats_original = objfeats_original.cpu().numpy()
            else:
                objfeats = None

            objs, edges = objs.cpu(), edges.cpu()
            obj_masks = obj_masks.cpu()
            # boxes_pred = boxes_pred.cpu()
            boxes_pred = batch["boxes"].cpu()  # use the ground truth boxes

            bbox_params = {
                "class_labels": F.one_hot(objs, num_classes=dataset.n_object_types+1).float(),  # +1 for empty node
                "translations": boxes_pred[..., :3],
                "sizes": boxes_pred[..., 3:6],
                "angles": boxes_pred[..., 6:]
            }
            boxes = dataset.post_process(bbox_params)
            bbox_params_t = torch.cat([
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"]
            ], dim=-1).numpy()
            assert bbox_params_t.shape[-1] == 7 + dataset.n_object_types+1

            # Evaluate (and visualize) each scene in the batch
            progress_bar = tqdm(
                total=len(bbox_params_t),
                desc="Visualize each scene",
                ncols=125,
                disable=args.verbose
            )
            for i in range(len(bbox_params_t)):
                # Get the textured objects by retrieving the 3D models
                trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids = get_textured_objects(
                    bbox_params_t[i],
                    objects_dataset, classes,
                    objfeats[i] if objfeats is not None else None,
                    "openshape_vitg14",  # TODO: make this configurable
                    verbose=args.verbose
                )
                trimesh_meshes_original, bbox_meshes_original, \
                obj_classes_original, obj_sizes_original, obj_ids_original = get_textured_objects(
                    bbox_params_t[i],
                    objects_dataset, classes,
                    objfeats_original[i] if objfeats_original is not None else None,
                    "openshape_vitg14",  # TODO: make it configurable
                    verbose=args.verbose
                )

                # Compute the CLIP cosine similarity between the retrieved objects and the text
                style_texts = [
                    f"a {selected_styles[i].lower()[:-1]} {obj_class.lower()}"
                    for obj_class in obj_classes if obj_class != None
                ]
                class_texts = [
                    f"a {obj_class.lower()}"
                    for obj_class in obj_classes if obj_class != None
                ]
                _, obj_clip_style_features = text_encoder(style_texts)  # (n_objs, D); already normalized
                _, obj_clip_class_features = text_encoder(class_texts)  # (n_objs, D); already normalized
                obj_clip_image_features = [
                    torch.from_numpy(np.load(
                        f"dataset/3D-FRONT/3D-FUTURE-model/{obj_id}/clip_vitb32.npy"
                    )).float().to(device)  # already normalized
                    for obj_id in obj_ids if obj_id != None
                ]
                obj_clip_image_features = torch.stack(obj_clip_image_features, dim=0)  # (n_objs, D)
                cossim = ((obj_clip_style_features - obj_clip_class_features) * obj_clip_image_features).sum(dim=-1)  # (n_objs,)
                clip_cossims.append(cossim.mean().item())

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "clip_cossim": "{:.6f}".format(np.mean(clip_cossims))
                })

                # Whether to visualize the scene by blender rendering
                if not args.visualize:
                    continue

                # To get the manually created floor plan, which includes vertices of all meshes in the scene
                all_vertices = np.concatenate([
                    tr_mesh.vertices for tr_mesh in trimesh_meshes
                ], axis=0)
                x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
                z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()

                tr_floor, _ = floor_plan_from_scene(
                    raw_dataset[0], config["data"]["path_to_floor_plan_textures"],  # `raw_dataset[0]` is not really used
                    without_room_mask=True,
                    rectangle_floor=True, room_size=[x_min, z_min, x_max, z_max]
                )
                trimesh_meshes.append(tr_floor)

                # Create a trimesh scene and export it to a temporary directory
                ii = epoch * len(dataset) + batch_idx * B + i
                export_dir = os.path.join(save_dir, f"{ii:04d}@{batch['scene_uids'][i]}_cfg{args.cfg_scale:.1f}_{args.sg2sc_cfg_scale:.1f}")
                tmp_dir = os.path.join(export_dir, "tmp")
                os.makedirs(export_dir, exist_ok=True)
                os.makedirs(tmp_dir, exist_ok=True)
                export_scene(tmp_dir, trimesh_meshes, bbox_meshes)

                if args.visualize_original:
                    # Export original scenes
                    trimesh_meshes_original.append(tr_floor)
                    tmp_dir_original = os.path.join(export_dir, "tmp_original")
                    os.makedirs(tmp_dir_original, exist_ok=True)
                    export_scene(tmp_dir_original, trimesh_meshes_original, bbox_meshes_original)

                # Render the exported scene by calling blender
                blender_render_scene(
                    tmp_dir,
                    export_dir,
                    top_down_view=(not args.eight_views),
                    resolution_x=args.resolution,
                    resolution_y=args.resolution
                )
                if args.visualize_original:
                    blender_render_scene(
                        tmp_dir_original,
                        export_dir,
                        output_suffix="_original",
                        top_down_view=(not args.eight_views),
                        resolution_x=args.resolution,
                        resolution_y=args.resolution
                    )
                # Delete the temporary directory
                os.system(f"rm -rf {tmp_dir}")
                if args.verbose:
                    print(f"Save the scene to {export_dir}\n")
                if args.visualize_original:
                    os.system(f"rm -rf {tmp_dir_original}")
                    if args.verbose:
                        print(f"Save the scene to {export_dir}\n")

                # Save conditioned text
                if texts is not None:
                    with open(os.path.join(export_dir, "description.txt"), "w") as f:
                        f.write(texts[i])

                # Visualize the generated scene graph
                if args.draw_scene_graph:
                    obj_ids = objs[i]  # (N,)
                    vis_objs, vis_obj_mapping, _vis_count = [], {}, 0
                    for idx, obj_id in enumerate(obj_ids):
                        if obj_id != dataset.n_object_types:
                            vis_objs.append(obj_id)
                            vis_obj_mapping[idx] = _vis_count
                            _vis_count += 1
                    edge_ids = edges[i]  # (N, N)
                    vis_triples = []
                    for idx1 in range(edge_ids.shape[0]):
                        for idx2 in range(idx1+1, edge_ids.shape[1]):
                            if edge_ids[idx1, idx2] != dataset.n_predicate_types:
                                vis_triples.append([
                                    vis_obj_mapping[idx1], edge_ids[idx1, idx2], vis_obj_mapping[idx2]
                                ])
                    vis_objs, vis_triples = torch.tensor(vis_objs), torch.tensor(vis_triples)
                    image = draw_scene_graph(
                        vis_objs, vis_triples,
                        object_types=dataset.object_types,
                        predicate_types=dataset.predicate_types
                    )
                    Image.fromarray(image).save(
                        os.path.join(export_dir, f"scene_graph.png")
                    )

                # Save the scene graph
                # graph = {
                #     "objs": vis_objs.numpy(),
                #     "triples": vis_triples.numpy()
                # }
                # np.savez(
                #     os.path.join(export_dir, f"scene_graph.npz"),
                #     **graph
                # )

            # Not generate all scenes
            if args.n_scenes != 0:  # only generate the first `n_scenes` scenes
                break

    # Save the evaluation results
    eval_info = ""
    eval_info += f"CLIP cosine similarity: {np.mean(clip_cossims):.6f}\n"

    if eval_info != "":
        with open(os.path.join(save_dir, f"eval_cfg{args.cfg_scale:.1f}_{args.sg2sc_cfg_scale:.1f}.txt"), "w") as f:
            f.write(eval_info)

if __name__ == "__main__":
    main()
