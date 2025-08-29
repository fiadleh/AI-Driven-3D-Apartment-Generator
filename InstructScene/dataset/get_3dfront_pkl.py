from src.data import filter_function
from src.data.threed_front import ThreedFront

config = {
    "filter_fn":                 "threed_front_bedroom",
    "min_n_boxes":               -1,
    "max_n_boxes":               -1,
    "path_to_invalid_scene_ids": "configs/invalid_threed_front_rooms.txt",
    "path_to_invalid_bbox_jids": "configs/black_list.txt",
    "annotation_file":           "configs/bedroom_threed_front_splits.csv"
}

dataset = ThreedFront.from_dataset_directory(
    dataset_directory="/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/dataset/3D-FRONT/3D-FRONT",
    path_to_models="/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/dataset/3D-FRONT/3D-FUTURE-model",
    path_to_model_info="/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/dataset/3D-FRONT/3D-FUTURE-model/model_info.json",
    filter_fn=filter_function(
        config, ["train", "val", "test"], False
    )
)
