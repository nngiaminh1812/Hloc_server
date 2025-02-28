
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

src_path = os.path.abspath("Hierarchical-Localization-Core/")
if src_path not in sys.path:
    sys.path.append(src_path)

from hloc import extractors,logger
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names, read_image
from hloc.utils.parsers import parse_image_lists
from hloc import (
    extract_features
)
#Thư viện cho việc xem các mô hình 3D đã xây dựng
import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from hloc.visualization import plot_images, read_image,plot_keypoints

from hloc.extract_features import ImageDataset
import  matplotlib.pyplot as plt
import cv2
def get_conf_local_feature():
    conf_local_feature=extract_features.confs["superpoint_aachen"]
    return conf_local_feature


def load_model(conf):
    """Load the model only once and return it."""
    logger.info("Loading model.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, conf["model"]["name"])
    
    print(f"Loaded Model Type: {type(Model)}")  # Debugging
    
    model = Model(conf["model"]).eval().to(device)  # <-- Error might be here
    logger.info(f"Model loaded on {device}.")
    return model

def get_keypoints_utils(image_query,image_name,conf_local_feature,model_local):
        logger.info(f"Creating Image Dataset")
        image_query=Path(image_query)
        dataset = ImageDataset(image_query, conf_local_feature["preprocessing"], [image_name])

        logger.info(f"Creating loader")
        loader = torch.utils.data.DataLoader(
                dataset, num_workers=1, shuffle=False, pin_memory=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pred_rs=None

        logger.info(f"Stating extract image to get keypoints")
        for idx, data in enumerate(loader):
                name = dataset.names[idx]
                pred = model_local({"image": data["image"].to(device, non_blocking=True)})
                pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}

                pred["image_size"] = original_size = data["original_size"][0].numpy()
                if "keypoints" in pred:
                        size = np.array(data["image"].shape[-2:][::-1])
                        scales = (original_size / size).astype(np.float32)
                        pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
                        if "scales" in pred:
                                pred["scales"] *= scales.mean()
                                # add keypoint uncertainties scaled to the original resolution
                                uncertainty = getattr(model_local, "detection_noise", 1) * scales.mean()
                pred_rs=pred
        logger.info(f"Extracted image and get keypoints successfully")
        return pred_rs

